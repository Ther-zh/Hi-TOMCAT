import os
import torch
import numpy as np
import random
import os
import yaml
import json

from tools.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def write_json(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_optimizer(model, config):
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def get_scheduler(optimizer, config, num_batches=-1):
    if not hasattr(config, 'scheduler'):
        return None
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'linear_w_warmup' or config.scheduler == 'cosine_w_warmup':
        assert num_batches != -1
        num_training_steps = num_batches * config.epochs
        num_warmup_steps = int(config.warmup_proportion * num_training_steps)
        if config.scheduler == 'linear_w_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
        if config.scheduler == 'cosine_w_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler


def step_scheduler(scheduler, config, bid, num_batches):
    if config.scheduler in ['StepLR']:
        if bid + 1 == num_batches:    # end of the epoch
            scheduler.step()
    elif config.scheduler in ['linear_w_warmup', 'cosine_w_warmup']:
        scheduler.step()

    return scheduler


# 新增：余弦相似度计算（方案二需要）
def cosine_similarity_mean(feat1, feat2):
    """
    计算两个特征矩阵的平均余弦相似度
    Args:
        feat1: [batch_size, feat_dim]
        feat2: [batch_size, feat_dim]
    Returns:
        mean_sim: 平均余弦相似度
    """
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)
    sim = torch.bmm(feat1.unsqueeze(1), feat2.unsqueeze(2)).squeeze()
    return torch.mean(sim).item()

# 新增：样本熵计算（方案二周期性修正需要）
def compute_sample_entropy(model, img_feat, text_feats):
    """
    计算单样本的预测熵
    Args:
        model: TOMCAT模型
        img_feat: [1, feat_dim] 单样本图像特征
        text_feats: [num_comps, feat_dim] 文本原型特征
    Returns:
        entropy: 预测熵（越小越可信）
        pred: 预测类别索引
    """
    with torch.no_grad():
        clip_logits = model.cos_sim_func_4com_cls(img_feat, text_feats)
        entropy = -(clip_logits.softmax(1) * clip_logits.log_softmax(1)).sum(1)
        pred = clip_logits.argmax(dim=1)[0].item()
    return entropy.item(), pred