import json
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2
import swanlab

import torch.nn.functional as F
import operator
import torch.nn as nn
from collections import defaultdict

from utils import *
from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model

# ====================== 方案一：层级化KAM ======================
class HierarchicalKAM(nn.Module):
    def __init__(self, text_feats, attr_names, obj_names, comp_to_prim, device, lambda1=0.5, lambda2=0.5):
        super(HierarchicalKAM, self).__init__()
        self.device = device
        self.num_comps, self.feat_dim = text_feats.shape
        self.attr_names = attr_names
        self.obj_names = obj_names
        self.comp_to_prim = comp_to_prim
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.comp_residual = nn.Parameter(
            torch.zeros([self.num_comps, self.feat_dim], dtype=text_feats.dtype).to(device),
            requires_grad=True
        )
        self.attr_residual = nn.Parameter(
            torch.zeros([len(attr_names), self.feat_dim], dtype=text_feats.dtype).to(device),
            requires_grad=True
        )
        self.obj_residual = nn.Parameter(
            torch.zeros([len(obj_names), self.feat_dim], dtype=text_feats.dtype).to(device),
            requires_grad=True
        )

    def forward(self, text_feats, weight):
        weight_reshaped = weight.view(-1, 1)
        residual = self.comp_residual.clone()
        for comp_idx in range(self.num_comps):
            attr_idx, obj_idx = self.comp_to_prim[comp_idx]
            residual[comp_idx] += self.lambda1 * self.attr_residual[attr_idx] + self.lambda2 * self.obj_residual[obj_idx]
        
        updated_feats = text_feats + weight_reshaped * residual
        updated_feats = F.normalize(updated_feats, dim=-1)
        return updated_feats

    def get_orthogonality_loss(self):
        orth_loss = 0.0
        num_attr = len(self.attr_names)
        num_obj = len(self.obj_names)
        
        for attr_idx in range(num_attr):
            for obj_idx in range(num_obj):
                attr_res = F.normalize(self.attr_residual[attr_idx], dim=-1)
                obj_res = F.normalize(self.obj_residual[obj_idx], dim=-1)
                sim = torch.dot(attr_res, obj_res)
                orth_loss += torch.square(sim)
        
        return orth_loss / (num_attr * num_obj)

# ====================== 方案二：稳健记忆库======================

class RobustPriorityQueue:
    def __init__(self, shot_capacity, sim_threshold, correction_interval, device, num_total_classes, use_robust_cache):
        self.shot_capacity = shot_capacity
        self.sim_threshold = sim_threshold
        self.correction_interval = correction_interval
        self.device = device
        self.num_total_classes = num_total_classes  # 总类别数（文本特征数）
        self.cache = defaultdict(list)
        self.step = 0
        # 新增：改进二启用开关（唯一新增的初始化参数）
        self.use_robust_cache = use_robust_cache

    def cosine_similarity(self, feat1, feat2):
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        return torch.dot(feat1, feat2).item()

    def enqueue(self, pred_cls, img_feat, entropy, text_feats):
        # 核心：关闭改进二时，执行原始Tomcat入队逻辑（无相似度过滤、无step累加）
        if not self.use_robust_cache:
            item = (img_feat.detach(), entropy.detach())
            if pred_cls in self.cache:
                if len(self.cache[pred_cls]) < self.shot_capacity:
                    self.cache[pred_cls].append(item)
                elif entropy < self.cache[pred_cls][-1][1]:
                    self.cache[pred_cls][-1] = item
                self.cache[pred_cls] = sorted(self.cache[pred_cls], key=lambda x: x[1])
            else:
                self.cache[pred_cls] = [item]
            return
        
        # 开启改进二时，执行原有逻辑（无任何修改）
        self.step += 1
        target_prototype = text_feats[pred_cls]
        sim = self.cosine_similarity(img_feat, target_prototype)
        
        if sim < self.sim_threshold:
            return
        
        item = (img_feat.detach(), entropy.detach())
        if pred_cls in self.cache:
            if len(self.cache[pred_cls]) < self.shot_capacity:
                self.cache[pred_cls].append(item)
            elif entropy < self.cache[pred_cls][-1][1]:
                self.cache[pred_cls][-1] = item
            self.cache[pred_cls] = sorted(self.cache[pred_cls], key=lambda x: x[1])
        else:
            self.cache[pred_cls] = [item]

    def periodic_correction(self, model, text_feats):
        # 关闭改进二时，直接跳过周期性修正（原始Tomcat无此逻辑）
        if not self.use_robust_cache:
            return
        
        # 开启改进二时，执行原有逻辑（无任何修改）
        if self.step % self.correction_interval != 0:
            return
        
        with torch.no_grad():
            for cls_idx in list(self.cache.keys()):
                valid_items = []
                for item in self.cache[cls_idx]:
                    img_feat, _ = item
                    clip_logits, new_entropy, new_pred = get_clip_logits(img_feat.unsqueeze(0), model, text_feats)
                    if new_pred == cls_idx and new_entropy < self.cache[cls_idx][-1][1]:
                        valid_items.append((img_feat, new_entropy))
                self.cache[cls_idx] = sorted(valid_items, key=lambda x: x[1])[:self.shot_capacity]

    def get_cache_data(self, feat_dim):
        """
        空缓存时返回维度为[0, num_total_classes]的cache_values，避免越界
        """
        cache_keys = []
        cache_values = []
        all_classes = []
        
        for cls_idx in sorted(self.cache.keys()):
            items = self.cache[cls_idx]
            if len(items) == 0:
                continue
            
            prototype = torch.zeros(feat_dim, device=self.device)
            for img_feat, _ in items:
                prototype += img_feat.to(self.device) / len(items)
            
            cache_keys.append(prototype)
            cache_values.append(cls_idx)
            all_classes.append(cls_idx)
        
        # 空缓存时返回维度兼容的空张量
        if len(cache_keys) == 0:
            # cache_keys: [0, feat_dim], cache_values: [0, num_total_classes]
            cache_keys = torch.empty((0, feat_dim), device=self.device)
            cache_values = torch.empty((0, self.num_total_classes), device=self.device)
            return cache_keys, cache_values, []
        
        cache_keys = torch.stack(cache_keys)
        # 确保cache_values维度为[num_cache_cls, num_total_classes]
        cache_values = F.one_hot(
            torch.tensor(cache_values, device=self.device, dtype=torch.int64),
            num_classes=self.num_total_classes
        ).to(cache_keys.dtype)
        
        return cache_keys, cache_values, all_classes
# ====================== 工具函数（核心修复：compute_cache_logits空值处理） ======================
def contrastive_loss(x: torch.Tensor, y: torch.Tensor, temperature):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    batch_size = x.shape[0]

    similarity_matrix = torch.mm(x, y.t()) * temperature.to(dtype=x.dtype)
    labels = torch.arange(batch_size, device=x.device)

    loss_x = F.cross_entropy(similarity_matrix, labels)
    loss_y = F.cross_entropy(similarity_matrix.t(), labels)

    return (loss_x + loss_y) / 2

def self_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_clip_logits(img_feats, model, text_feats):
    clip_logits = model.cos_sim_func_4com_cls(img_feats, text_feats)
    entropy = self_entropy(clip_logits)
    pred = clip_logits.argmax(dim=1)[0].item()
    return clip_logits, entropy, pred

def adaptive_update_weight(img_feats, text_feats, alpha=10):
    if text_feats.numel() == 0:
        return torch.zeros((img_feats.shape[0], 0), device=img_feats.device)
    
    if len(img_feats.shape) == 1:
        img_feats = img_feats.unsqueeze(0)
    if len(text_feats.shape) == 1:
        text_feats = text_feats.unsqueeze(0)
    
    similarity = img_feats @ text_feats.T
    weight = 1 / (1 + torch.exp(alpha * similarity))
    return weight

def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, device):
    """
    1. 空缓存时，根据image_features的batch_size和总类别数返回零张量
    2. 避免访问空张量的shape[1]
    """
    batch_size = image_features.shape[0]
    # 空缓存判断：cache_keys为空 或 cache_values为空
    if len(cache_keys) == 0 or len(cache_values) == 0:
        # 关键：获取总类别数（cache_values的第二维）
        num_classes = cache_values.shape[1] if len(cache_values.shape) >= 2 else 0
        # 返回维度为[batch_size, num_classes]的零张量
        return torch.zeros((batch_size, num_classes), device=device)
    
    # 非空缓存时正常计算
    affinity = image_features @ cache_keys.T
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits.to(device)

# ====================== 核心预测函数 ======================
def predict_logits_text_first_with_hitomcat(model, dataset, config):
    model.eval()
    device = config.device
    cpu_cache = config.cpu_cache
    use_cache = config.use_img_cache

    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []
    all_logits = torch.Tensor().cpu()
    
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in pairs_dataset]).to(device)
    num_total_classes = len(pairs_dataset)  # 总类别数=组合数
    
    # 构建组合→基元映射
    attr_names = list(attr2idx.keys())
    obj_names = list(obj2idx.keys())
    comp_to_prim = {
        comp_idx: (attr2idx[pairs_dataset[comp_idx][0]], obj2idx[pairs_dataset[comp_idx][1]])
        for comp_idx in range(len(pairs_dataset))
    }

    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # 加载文本特征
    with torch.no_grad():
        text_feats = []
        num_text_batch = pairs.shape[0] // config.text_encoder_batch_size
        for i_batch in range(num_text_batch):
            batch_pairs = pairs[i_batch*config.text_encoder_batch_size : (i_batch+1)*config.text_encoder_batch_size]
            batch_feats = model.encode_text_for_open(batch_pairs)
            text_feats.append(batch_feats)
        if pairs.shape[0] % config.text_encoder_batch_size != 0:
            batch_pairs = pairs[num_text_batch*config.text_encoder_batch_size :]
            batch_feats = model.encode_text_for_open(batch_pairs)
            text_feats.append(batch_feats)
        text_feats = torch.cat(text_feats, dim=0)
        text_feats = F.normalize(text_feats, dim=-1)
    model.release_text_encoder()

    # 初始化参数
    pos_params = {
        'shot_capacity': config.shot_capacity,
        'alpha': config.alpha,
        'beta': config.beta,
    }

    # 初始化层级化KAM
    hier_kam = HierarchicalKAM(
        text_feats=text_feats,
        attr_names=attr_names,
        obj_names=obj_names,
        comp_to_prim=comp_to_prim,
        device=device,
        lambda1=0.5,
        lambda2=0.5
    )
    optimizer_t = torch.optim.AdamW(
        [{'params': hier_kam.parameters(), 'lr': config.text_lr, 'eps': config.eps, 'weight_decay': config.wd}]
    )

    # 初始化稳健记忆库（传入总类别数）
    if use_cache:
        robust_queue = RobustPriorityQueue(
            shot_capacity=config.shot_capacity,
            sim_threshold=config.sim_threshold,
            correction_interval=config.correction_interval,
            device=device,
            num_total_classes=num_total_classes,  # 关键：传入总类别数
            use_robust_cache=config.use_robust_cache  # 新增：传入开关
        )
        optimizer_i = torch.optim.AdamW(
            [{'params': hier_kam.parameters(), 'lr': config.image_lr, 'eps': config.eps, 'weight_decay': config.wd}]
        )

    # 测试主循环
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
        img = data[0].to(device)
        attr_gt, obj_gt, pair_gt = data[1], data[2], data[3]

        # 图像特征编码
        with torch.no_grad():
            img_feats, _ = model.encode_image(img.type(model.clip.dtype))
            img_feats = F.normalize(img_feats, dim=-1)
            text_weight = adaptive_update_weight(img_feats, text_feats, config.hier_theta)

        # 层级化KAM更新文本特征
        new_text_feats = hier_kam(text_feats, text_weight)
        clip_logits, entropy, pred = get_clip_logits(img_feats, model, new_text_feats)

        # 稳健记忆库更新
        if use_cache:
            robust_queue.enqueue(pred, img_feats[0], entropy[0], new_text_feats)
            robust_queue.periodic_correction(model, new_text_feats)
            # 获取缓存数据（维度兼容）
            pos_cache_keys, pos_cache_values, all_classes = robust_queue.get_cache_data(img_feats.shape[-1])
            
            # 缓存权重计算
            with torch.no_grad():
                if pos_cache_keys.numel() == 0:
                    cache_weight = torch.zeros((img_feats.shape[0], 0), device=device)
                else:
                    cache_weight = adaptive_update_weight(img_feats, pos_cache_keys, config.hier_theta)
            
            # 缓存特征更新
            new_pos_cache_keys = pos_cache_keys.clone()
            if len(all_classes) > 0 and cache_weight.numel() > 0:
                cache_weight_reshaped = cache_weight.view(-1, 1)
                for i, cls_idx in enumerate(all_classes):
                    attr_idx, obj_idx = comp_to_prim[cls_idx]
                    residual = hier_kam.comp_residual[cls_idx] + \
                               hier_kam.lambda1 * hier_kam.attr_residual[attr_idx] + \
                               hier_kam.lambda2 * hier_kam.obj_residual[obj_idx]
                    new_pos_cache_keys[i] = new_pos_cache_keys[i] + cache_weight_reshaped[i] * residual
                new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=-1)
            
            # 计算缓存logits（无越界）
            cache_logits = compute_cache_logits(
                img_feats, new_pos_cache_keys, pos_cache_values,
                pos_params['alpha'], pos_params['beta'], device
            )
            # 确保cache_logits维度和clip_logits一致
            if cache_logits.shape[1] != clip_logits.shape[1]:
                cache_logits = torch.zeros_like(clip_logits, device=device)
            clip_logits = clip_logits + cache_logits
            entropy = self_entropy(clip_logits)

        # 损失计算
        loss = entropy
        if config.use_align_loss and use_cache and len(all_classes) > 0:
            image2text_loss = contrastive_loss(
                new_pos_cache_keys.to(device), new_text_feats[all_classes, :],
                model.clip.logit_scale.exp()
            )
            loss = loss + config.align_loss_weight * image2text_loss
        orth_loss = hier_kam.get_orthogonality_loss()
        loss = loss + config.lambda_orth * orth_loss

        # 反向传播
        optimizer_t.zero_grad()
        if use_cache:
            optimizer_i.zero_grad()
        loss.backward()
        optimizer_t.step()
        if use_cache:
            optimizer_i.step()

        # 日志记录
        if config.use_wandb:
            swanlab.log({
                'total_loss': loss.item(),
                'orth_loss': orth_loss.item(),
                'entropy_loss': entropy.mean().item(),
                'cache_size': len(robust_queue.cache) if use_cache else 0
            })

        # 收集结果
        clip_logits = clip_logits.detach().cpu()
        all_logits = torch.cat([all_logits, clip_logits], dim=0)
        all_attr_gt.append(attr_gt)
        all_obj_gt.append(obj_gt)
        all_pair_gt.append(pair_gt)

    # 结果整合
    all_attr_gt = torch.cat(all_attr_gt).cpu()
    all_obj_gt = torch.cat(all_obj_gt).cpu()
    all_pair_gt = torch.cat(all_pair_gt).cpu()

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt