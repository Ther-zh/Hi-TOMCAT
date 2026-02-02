import os
import sys
import csv
import json
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import swanlab
from itertools import product
from datetime import datetime
from torch.cuda.amp import autocast

# ====================== 全局配置：仅指定【当前阶段要调的2个参数范围】+ 基础适配 ======================
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- 仅需修改这里！指定当前要调的2个参数范围 ----------------------
# 注意：调参范围二选一，根据yml中use_robust_cache开关匹配
# 1. STAGE1（yml中use_robust_cache=False）：仅改进一，调以下2个参数
TUNE_PARAMS_STAGE1 = {
    "param_names": ["lambda_orth", "hier_theta"],
    "ranges": [
        [0.01,0.05,0.1,0.4,0.7,1.0,1.3,1.6,2],  # lambda_orth范围
        [0.9, 1.0, 1.1]                        # hier_theta范围
    ]
}

# 2. STAGE2（yml中use_robust_cache=True）：改进一+二，调以下2个参数（改进一固定）
TUNE_PARAMS_STAGE2 = {
    "param_names": ["correction_interval", "sim_threshold"],
    "ranges": [
        [10, 20, 30, 40],                          # correction_interval范围
        [0.10, 0.15, 0.20, 0.25, 0.30]             # sim_threshold范围
    ],
    "improve1_best_params_path": "tune_improve1_only_results/best_params.json"  # 阶段1最优参数路径
}

# ---------------------- 固定配置（无需修改） ----------------------
CFG_PATH = "config/ut-zappos.yml"  # 你的yml路径
CORE_METRICS = ["AUC", "best_hm", "attr_acc", "best_seen", "best_unseen", "obj_acc", "biasterm"]
SAVE_DIR_PREFIX = "tune_results_"  # 结果保存目录前缀（自动加阶段名）

# ====================== 对齐swan_test2.py的工具类（无修改）======================
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
# class Evaluator:
#     def __init__(self, dset, model, device):
#         self.dset = dset
#         self.device = device
#         pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
#         self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
#         self.pairs = torch.LongTensor(pairs)

#         if dset.phase == 'train':
#             test_pair_set = set(dset.train_pairs)
#             test_pair_gt = set(dset.train_pairs)
#         elif dset.phase == 'val':
#             test_pair_set = set(dset.val_pairs + dset.train_pairs)
#             test_pair_gt = set(dset.val_pairs)
#         else:
#             test_pair_set = set(dset.test_pairs + dset.train_pairs)
#             test_pair_gt = set(dset.test_pairs)

#         self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
#         self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)
#         for attr, obj in test_pair_gt:
#             pair_val = dset.pair2idx[(attr, obj)]
#             key = (dset.attr2idx[attr], dset.obj2idx[obj])
#             self.test_pair_dict[key] = [pair_val, 0, 0]

#         masks = [1 for _ in dset.pairs] if dset.open_world else [1 if pair in test_pair_set else 0 for pair in dset.pairs]
#         self.closed_mask = torch.BoolTensor(masks)
#         seen_mask = [1 if pair in set(dset.train_pairs) else 0 for pair in dset.pairs]
#         self.seen_mask = torch.BoolTensor(seen_mask)

#         oracle_obj_mask = []
#         for _obj in dset.objs:
#             oracle_obj_mask.append(torch.BoolTensor([1 if _obj == obj else 0 for attr, obj in dset.pairs]))
#         self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)
#         self.score_model = self.score_manifold_model

#     def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):
#         def get_pred(_s):
#             _, pred = _s.topk(topk, dim=1)
#             pred = pred.view(-1)
#             return self.pairs[pred][:,0].view(-1,topk), self.pairs[pred][:,1].view(-1,topk)
#         orig = scores.clone()
#         scores[~self.seen_mask.repeat(scores.shape[0],1)] += bias
#         return {
#             "open": get_pred(scores),
#             "unbiased_open": get_pred(orig),
#             "closed": get_pred(scores.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10)),
#             "unbiased_closed": get_pred(orig.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10))
#         }

#     def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
#         scores = torch.stack([scores[(a,o)] for a,o in self.dset.pairs], 1)
#         return {**self.generate_predictions(scores, obj_truth, bias, topk), "scores": scores.clone()}

#     def evaluate_predictions(self, preds, attr_gt, obj_gt, pair_gt, allpred, topk=1):
#         from scipy.stats import hmean
#         attr_gt, obj_gt, pair_gt = attr_gt.cpu(), obj_gt.cpu(), pair_gt.cpu()
#         seen_ind = torch.tensor([i for i,(a,o) in enumerate(zip(attr_gt.numpy(), obj_gt.numpy())) if (a,o) in self.train_pairs])
#         unseen_ind = torch.tensor([i for i,(a,o) in enumerate(zip(attr_gt.numpy(), obj_gt.numpy())) if (a,o) not in self.train_pairs])

#         def process(s):
#             a_match = (attr_gt.unsqueeze(1).repeat(1,topk) == s[0][:,:topk]).any(1).float()
#             o_match = (obj_gt.unsqueeze(1).repeat(1,topk) == s[1][:,:topk]).any(1).float()
#             match = (a_match * o_match).float()
#             return a_match, o_match, match, match[seen_ind], match[unseen_ind]
#         stats = {}
#         for k in ["closed", "unbiased_closed"]:
#             a,o,m,s,u = process(preds[k])
#             stats[f"{k}_attr_match"] = a.mean().item()
#             stats[f"{k}_obj_match"] = o.mean().item()
#             stats[f"{k}_match"] = m.mean().item()
#             stats[f"{k}_seen_match"] = s.mean().item() if len(s) else 0.0
#             stats[f"{k}_unseen_match"] = u.mean().item() if len(u) else 0.0

#         scores = preds["scores"]
#         correct_scores = scores[torch.arange(len(scores)), pair_gt][unseen_ind]
#         max_seen = scores[unseen_ind][:, self.seen_mask].topk(topk,1)[0][:,topk-1]
#         diff = max_seen - correct_scores
#         valid_diff = diff[stats["closed_unseen_match"]>0] - 1e-4
#         biaslist = valid_diff[::max(len(valid_diff)//20,1)] if len(valid_diff) else [0.0]

#         seen_acc, unseen_acc = [stats["closed_seen_match"]], [stats["closed_unseen_match"]]
#         base_scores = torch.stack([allpred[(a,o)] for a,o in self.dset.pairs], 1)
#         for b in biaslist:
#             s,u = process(self.score_fast_model(base_scores.clone(), obj_gt, b, topk))[3:]
#             seen_acc.append(s.mean().item() if len(s) else 0.0)
#             unseen_acc.append(u.mean().item() if len(u) else 0.0)

#         seen_acc, unseen_acc = np.array(seen_acc), np.array(unseen_acc)
#         hm = hmean([seen_acc, unseen_acc], axis=0) if len(seen_acc) else 0.0
#         return {
#             **stats,
#             "AUC": np.trapz(seen_acc, unseen_acc),
#             "best_hm": np.max(hm) if len(hm) else 0.0,
#             "best_seen": np.max(seen_acc),
#             "best_unseen": np.max(unseen_acc),
#             "biasterm": biaslist[np.argmax(hm)] if len(hm) else 1e3
#         }

#     def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
#         scores[~self.seen_mask.repeat(scores.shape[0],1)] += bias
#         closed = scores.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10)
#         _, pred = closed.topk(topk,1)
#         pred = pred.view(-1)
#         return (self.pairs[pred][:,0].view(-1,topk), self.pairs[pred][:,1].view(-1,topk))
# ====================== 对齐swan_test2.py的工具类（终极修复：解决unseen计算维度展平）======================
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
class Evaluator:
    def __init__(self, dset, model, device):
        self.dset = dset
        self.device = device
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        if dset.phase == 'train':
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        masks = [1 for _ in dset.pairs] if dset.open_world else [1 if pair in test_pair_set else 0 for pair in dset.pairs]
        self.closed_mask = torch.BoolTensor(masks)
        seen_mask = [1 if pair in set(dset.train_pairs) else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(seen_mask)
        # 预计算seen pair的数量（用于维度校验）
        self.seen_pair_num = self.seen_mask.sum().item()
        print(f"【Evaluator初始化】seen_mask长度：{len(self.seen_mask)} | seen pair数：{self.seen_pair_num}")

        oracle_obj_mask = []
        for _obj in dset.objs:
            oracle_obj_mask.append(torch.BoolTensor([1 if _obj == obj else 0 for attr, obj in dset.pairs]))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)
        self.score_model = self.score_manifold_model

    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):
        def get_pred(_s):
            _, pred = _s.topk(topk, dim=1)
            pred = pred.view(-1)
            return self.pairs[pred][:,0].view(-1,topk), self.pairs[pred][:,1].view(-1,topk)
        orig = scores.clone()
        scores[~self.seen_mask.repeat(scores.shape[0],1)] += bias
        return {
            "open": get_pred(scores),
            "unbiased_open": get_pred(orig),
            "closed": get_pred(scores.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10)),
            "unbiased_closed": get_pred(orig.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10))
        }

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        scores = torch.stack([scores[(a,o)] for a,o in self.dset.pairs], 1)
        # 维度校验
        assert scores.shape == (len(obj_truth), len(self.dset.pairs)), \
            f"score_manifold_model: scores维度异常 {scores.shape}，预期({len(obj_truth)}, {len(self.dset.pairs)})"
        return {**self.generate_predictions(scores, obj_truth, bias, topk), "scores": scores.clone()}

    def evaluate_predictions(self, preds, attr_gt, obj_gt, pair_gt, allpred, topk=1):
        from scipy.stats import hmean
        attr_gt, obj_gt, pair_gt = attr_gt.cpu(), obj_gt.cpu(), pair_gt.cpu()
        # 优化seen/unseen索引计算，避免循环，提升速度+稳定性
        pair_comb = torch.stack([attr_gt, obj_gt], dim=1).numpy()
        train_pair_set = set(tuple(p) for p in self.train_pairs)
        seen_mask = np.array([tuple(p) in train_pair_set for p in pair_comb])
        seen_ind = torch.where(torch.BoolTensor(seen_mask))[0]
        unseen_ind = torch.where(~torch.BoolTensor(seen_mask))[0]
        self.unseen_num = len(unseen_ind)
        print(f"【evaluate_predictions】总样本：{len(attr_gt)} | seen样本：{len(seen_ind)} | unseen样本：{self.unseen_num}")

        def process(s):
            a_match = (attr_gt.unsqueeze(1).repeat(1,topk) == s[0][:,:topk]).any(1).float()
            o_match = (obj_gt.unsqueeze(1).repeat(1,topk) == s[1][:,:topk]).any(1).float()
            match = (a_match * o_match).float()
            return a_match, o_match, match, match[seen_ind], match[unseen_ind]

        stats = {}
        for k in ["closed", "unbiased_closed"]:
            a,o,m,s,u = process(preds[k])
            stats[f"{k}_attr_match"] = a.mean().item()
            stats[f"{k}_obj_match"] = o.mean().item()
            stats[f"{k}_match"] = m.mean().item()
            stats[f"{k}_seen_match"] = s.mean().item() if len(s) else 0.0
            stats[f"{k}_unseen_match"] = u.mean().item() if len(u) else 0.0

        scores = preds["scores"]
        # ====================== 核心修复：unseen样本scores计算（解决96162维度展平）======================
        if self.unseen_num == 0:
            biaslist = [0.0]
            print("【unseen计算】无unseen样本，跳过bias计算")
        else:
            # 1. 计算unseen样本的正确pair得分（维度[1891]，强制一维）
            correct_scores = scores[torch.arange(len(scores)), pair_gt][unseen_ind].squeeze()
            # 强制reshape为一维，避免隐性维度问题
            correct_scores = correct_scores.reshape(-1)
            print(f"【unseen计算】correct_scores维度：{correct_scores.shape}（预期[{self.unseen_num}]）")

            # 2. 计算unseen样本的seen pair最大得分（核心修复：避免展平，强制一维）
            # 先索引unseen样本，再取seen mask，得到[1891,33]
            scores_unseen_seen = scores[unseen_ind][:, self.seen_mask]
            print(f"【unseen计算】scores_unseen_seen维度：{scores_unseen_seen.shape}（预期[{self.unseen_num},{self.seen_pair_num}]）")
            # topk取最大值，得到[1891,1]，再squeeze+reshape为[1891]
            max_seen, _ = scores_unseen_seen.topk(topk, dim=1)
            max_seen = max_seen.squeeze(dim=1).reshape(-1)
            print(f"【unseen计算】max_seen维度：{max_seen.shape}（预期[{self.unseen_num}]）")

            # 3. 维度强制校验（核心！确保两个张量都是[1891]）
            assert correct_scores.shape == max_seen.shape == (self.unseen_num,), \
                f"维度不匹配：correct_scores{correct_scores.shape} | max_seen{max_seen.shape}，预期均为({self.unseen_num},)"

            # 4. 计算差值，后续操作均基于一维张量
            diff = max_seen - correct_scores
            diff = diff.reshape(-1)
            print(f"【unseen计算】diff维度：{diff.shape}（预期[{self.unseen_num}]）")

            # 5. 过滤有效差值（修复mask广播错误，原代码用标量索引的bug）
            # 原错误：stats["closed_unseen_match"]是标量，用标量索引会导致广播
            # 正确：取unseen样本的match结果，生成mask
            unseen_match = process(preds["closed"])[4]  # 取unseen_ind的match结果
            valid_mask = (unseen_match > 0).cpu()
            valid_diff = diff[valid_mask] - 1e-4
            valid_diff = valid_diff.reshape(-1)
            print(f"【unseen计算】valid_diff维度：{valid_diff.shape} | 有效样本数：{len(valid_diff)}")

            # 6. 生成biaslist（避免步长导致的维度膨胀）
            if len(valid_diff) == 0:
                biaslist = [0.0]
            else:
                step = max(len(valid_diff) // 20, 1)
                biaslist = valid_diff[::step].tolist()
            print(f"【unseen计算】biaslist长度：{len(biaslist)}")

        # ====================== bias循环计算（修复后）======================
        seen_acc, unseen_acc = [stats["closed_seen_match"]], [stats["closed_unseen_match"]]
        base_scores = torch.stack([allpred[(a,o)] for a,o in self.dset.pairs], 1)
        # 维度校验
        assert base_scores.shape == scores.shape, f"base_scores维度异常 {base_scores.shape}，预期{scores.shape}"

        for b in biaslist:
            # 调用score_fast_model，取seen/unseen准确率
            s,u = process(self.score_fast_model(base_scores.clone(), obj_gt, b, topk))[3:]
            seen_acc.append(s.mean().item() if len(s) else 0.0)
            unseen_acc.append(u.mean().item() if len(u) else 0.0)

        seen_acc, unseen_acc = np.array(seen_acc), np.array(unseen_acc)
        hm = hmean([seen_acc, unseen_acc], axis=0) if len(seen_acc) and len(unseen_acc) else 0.0
        # 最终指标返回
        return {
            **stats,
            "AUC": np.trapz(seen_acc, unseen_acc) if len(seen_acc) > 1 else 0.0,
            "best_hm": np.max(hm) if len(hm) else 0.0,
            "best_seen": np.max(seen_acc) if len(seen_acc) else 0.0,
            "best_unseen": np.max(unseen_acc) if len(unseen_acc) else 0.0,
            "biasterm": biaslist[np.argmax(hm)] if len(hm) and len(biaslist) else 1e3
        }

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        scores[~self.seen_mask.repeat(scores.shape[0],1)] += bias
        closed = scores.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10)
        _, pred = closed.topk(topk,1)
        pred = pred.view(-1)
        return (self.pairs[pred][:,0].view(-1, topk), self.pairs[pred][:,1].view(-1, topk))
# def test(test_dset, evaluator, logits, attr_gt, obj_gt, pair_gt, config):
#     preds = {p: logits[:,i] for i,p in enumerate(test_dset.pairs)}
#     all_pred = torch.stack([preds[(a,o)] for a,o in test_dset.pairs], 1)
#     res = evaluator.score_model(preds, obj_gt, 1e3, 1)
#     attr_acc = (res['unbiased_closed'][0].squeeze(-1) == attr_gt).float().mean().item()
#     obj_acc = (res['unbiased_closed'][1].squeeze(-1) == obj_gt).float().mean().item()
#     stats = evaluator.evaluate_predictions(res, attr_gt, obj_gt, pair_gt, preds, 1)
#     return {**stats, "attr_acc": attr_acc, "obj_acc": obj_acc}
def test(test_dset, evaluator, logits, attr_gt, obj_gt, pair_gt, config):

    preds = {p: logits[:,i] for i,p in enumerate(test_dset.pairs)}
    # 🔴 新增：打印preds值的维度（确认每一列维度正确）
    pred_vals = list(preds.values())
    
    # 原代码：构造all_pred张量
    all_pred = torch.stack([preds[(a,o)] for a,o in test_dset.pairs], 1)
    
    # 原代码：调用score_model
    res = evaluator.score_model(preds, obj_gt, 1e3, 1)

    
    # 原代码：计算准确率
    attr_acc = (res['unbiased_closed'][0].squeeze(-1) == attr_gt).float().mean().item()
    obj_acc = (res['unbiased_closed'][1].squeeze(-1) == obj_gt).float().mean().item()
    
    # 原代码：评估预测结果
    stats = evaluator.evaluate_predictions(res, attr_gt, obj_gt, pair_gt, preds, 1)
    return {**stats, "attr_acc": attr_acc, "obj_acc": obj_acc}
# ====================== 核心工具函数（适配yml读取+调参逻辑）======================
# ====================== 工具函数：加载/修改配置（完全对齐旧脚本，适配分阶段调参）======================
def load_config(cfg_path):
    """完全复用旧脚本的配置加载逻辑：直接加载yml为字典，避免Namespace格式问题"""
    import yaml
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """旧脚本配套的配置保存函数，保证yml格式正确"""
    import yaml
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)

def modify_config(original_cfg, tune_params, param_values):
    """
    核心修复：完全对齐旧脚本的配置修改逻辑
    1. lambda_orth/hier_theta 强制存入`tta`嵌套节点（代码预期读取位置）
    2. correction_interval/sim_threshold 存入顶层（改进二参数默认位置）
    3. 保留原始yml所有配置，仅修改调参参数
    4. 自动识别阶段，阶段2固定改进一最优参数
    """
    # 深拷贝原始配置，避免修改原文件
    cfg = copy.deepcopy(original_cfg)
    # 确保核心嵌套节点存在（旧脚本逻辑，避免键不存在报错）
    if "tta" not in cfg:
        cfg["tta"] = {}
    if "test" not in cfg:
        cfg["test"] = {}
    
    # 1. 从yml读取阶段开关，判断当前调参阶段
    use_robust_cache = cfg.get("use_robust_cache", False)
    param1, param2 = tune_params["param_names"]
    val1, val2 = param_values
    
    # 2. 阶段1：仅改进一（use_robust_cache=False）→ 改tta节点下的lambda_orth/hier_theta
    if not use_robust_cache:
        cfg["tta"][param1] = val1
        cfg["tta"][param2] = val2
        # 强制关闭改进二，对齐阶段1需求（旧脚本FIXED_PARAMS逻辑）
        cfg["tta"]["use_img_cache"] = False
    # 3. 阶段2：改进一+二（use_robust_cache=True）→ 改顶层的correction_interval/sim_threshold，固定改进一
    else:
        # 加载阶段1最优参数，固定到tta节点（核心：和旧脚本一致，存在tta下）
        improve1_best = load_improve1_best_params(tune_params["improve1_best_params_path"])
        cfg["tta"]["lambda_orth"] = improve1_best["best_lambda_orth"]
        cfg["tta"]["hier_theta"] = improve1_best["best_hier_theta"]
        # 开启改进二，对齐阶段2需求
        cfg["tta"]["use_img_cache"] = True
        # 修改改进二的调参参数（顶层，和yml配置一致）
        cfg[param1] = val1
        cfg[param2] = val2
    
    # 4. 固化基础参数（对齐旧脚本FIXED_PARAMS，分配到对应节点）
    fixed_params = {
        "open_world": False, "text_first": True, "use_wandb": True, "seed": 42,
        "eval_batch_size_wo_tta": 1, "num_workers": 0, "threshold_trials": 6,
        "shot_capacity": 3, "use_tta": True
    }
    test_params = ["open_world", "text_first", "use_wandb", "seed", "eval_batch_size_wo_tta", "num_workers", "threshold_trials"]
    tta_params = ["shot_capacity", "use_tta", "use_img_cache"]
    for k, v in fixed_params.items():
        if k in test_params:
            cfg["test"][k] = v
        elif k in tta_params and k in cfg["tta"]:
            cfg["tta"][k] = v
    
    # 5. 生成临时配置文件名（格式化，避免浮点数/特殊字符问题）
    val1_fmt = f"{val1:.4f}" if isinstance(val1, float) else str(val1)
    val2_fmt = f"{val2:.4f}" if isinstance(val2, float) else str(val2)
    temp_cfg_path = f"temp_tune_{param1}_{val1_fmt}_{param2}_{val2_fmt}.yml"
    # 保存临时配置（完全对齐旧脚本格式）
    save_config(cfg, temp_cfg_path)
    return temp_cfg_path


# def load_improve1_best_params(params_path):
#     """加载阶段1最优改进一参数（阶段2用）"""
#     if not os.path.exists(params_path):
#         raise FileNotFoundError(f"请先运行阶段1（use_robust_cache=False）生成{params_path}")
#     with open(params_path, 'r') as f:
#         return json.load(f)

# ====================== 实验运行+记录+可视化（适配动态调参）======================
# ====================== 核心函数：运行单次实验（100%复用旧脚本可行逻辑）======================
def run_experiment(temp_cfg_path):
    """
    完全复用旧脚本的实验运行逻辑，无任何修改！
    配置解析→数据集加载→模型实例化→预测→指标计算，和旧脚本完全一致
    """
    sys.path.append(DIR_PATH)
    from parameters import parser
    from utils import load_args, set_seed
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from swan_test_hitomcat import predict_logits_text_first_with_hitomcat  # 你的改进一函数

    try:
        # 1. 配置解析（和旧脚本/ swan_test2.py完全一致）
        args = parser.parse_args(["--cfg", temp_cfg_path])
        load_args(args.cfg, args)
        config = args
        set_seed(config.seed)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  📌 使用设备：{config.device}")

        # 2. 实例化测试数据集（完全对齐旧脚本）
        print(f"  📌 加载数据集：{config.dataset}")
        test_dataset = CompositionDataset(
            config.dataset_path,
            phase='test',
            split='compositional-split-natural',
            open_world=config.open_world
        )

        # 3. 模型加载（完全对齐旧脚本，参数处理一致）
        print(f"  📌 加载模型：{config.load_model}")
        allattrs = test_dataset.attrs
        allobj = test_dataset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)
        model = get_model(config, attributes=attributes, classes=classes, offset=offset).to(config.device)
        if config.load_model and os.path.exists(config.load_model):
            model.load_state_dict(torch.load(config.load_model, map_location='cpu'))
        model.eval()

        # 4. 选择预测函数（和旧脚本一致，启用改进一）
        predict_logits_func = predict_logits_text_first_with_hitomcat
        print(f"  📌 使用预测函数：Hi-TOMCAT（改进一）")

        # 5. 运行预测（带autocast，对齐旧脚本）
        print(f"  📌 开始预测...")
        with autocast(dtype=torch.bfloat16):
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(model, test_dataset, config)

        # 6. 计算指标（复用旧脚本的Evaluator+test函数，保证指标一致）
        print(f"  📌 计算指标...")
        evaluator = Evaluator(test_dataset, model=None, device=config.device)
        test_stats = test(test_dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)

        # 清理临时配置文件
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)

        # 提取核心指标
        res = {k: test_stats.get(k, 0.0) for k in CORE_METRICS}
        print(f"  ✅ 实验完成 | AUC: {res['AUC']:.4f} | Best HM: {res['best_hm']:.4f}")
        return res

    except Exception as e:
        # 失败时清理临时文件
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)
        raise Exception(f"运行异常：{str(e)}")

def load_improve1_best_params(params_path):
    """加载阶段1最优参数，适配阶段2固定需求"""
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"请先运行阶段1（yml中use_robust_cache=False）生成最优参数文件！\n缺失文件：{params_path}"
        )
    with open(params_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    # 兼容旧脚本的参数名，确保能正确读取
    if "best_lambda_orth" not in best_params or "best_hier_theta" not in best_params:
        raise KeyError("阶段1最优参数文件缺少核心键：best_lambda_orth / best_hier_theta")
    return best_params
def init_record(save_dir, param_names):
    """初始化CSV和SwanLab"""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "tune_metrics.csv")
    headers = param_names + CORE_METRICS
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()
    # SwanLab项目名=保存目录名
    swanlab.init(project=os.path.basename(save_dir), config={"tune_params": param_names})
    return csv_path

def record_data(csv_path, param_names, param_values, metrics):
    """记录数据到CSV+SwanLab"""
    row = dict(zip(param_names, param_values))
    row.update(metrics)
    with open(csv_path, 'a+', newline='') as f:
        csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
    swanlab.log({**metrics, **dict(zip(param_names, param_values))})

def visualize(save_dir, csv_path, param_names):
    """可视化调参结果"""
    import pandas as pd
    df = pd.read_csv(csv_path).dropna(subset=["AUC"]).query("AUC>0")
    if len(df) == 0:
        print("无有效数据，跳过可视化")
        return

    # 热力图（AUC）
    for value in ["AUC", "best_hm", "attr_acc"]:
        plt.figure(figsize=(10,8))
        pivot = df.pivot(index=param_names[0], columns=param_names[1], values=value)
        im = plt.imshow(pivot, cmap="YlGnBu", aspect="auto")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                plt.text(j, i, f"{pivot.iloc[i,j]:.4f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, label=value)
        plt.xlabel(param_names[1], fontsize=14, fontweight="bold")
        plt.ylabel(param_names[0], fontsize=14, fontweight="bold")
        plt.title(f"{value} Heatmap (Higher is Better)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{value}_heatmap.png"), dpi=300)
        plt.close()

    # 折线图（核心指标）
    for metric in ["AUC", "best_hm", "attr_acc"]:
        plt.figure(figsize=(12,6))
        for p2 in df[param_names[1]].unique():
            data = df[df[param_names[1]] == p2].sort_values(param_names[0])
            plt.plot(data[param_names[0]], data[metric], marker="o", linewidth=2, label=f"{param_names[1]}={p2}")
        plt.xlabel(param_names[0], fontsize=14, fontweight="bold")
        plt.ylabel(metric, fontsize=14, fontweight="bold")
        plt.title(f"{metric} vs {param_names[0]}", fontsize=16, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_lineplot.png"), dpi=300)
        plt.close()

    # 保存最优参数
    best_row = df.loc[df["AUC"].idxmax()]
    best_params = {**dict(zip(param_names, best_row[param_names])), **best_row[CORE_METRICS].to_dict()}
    with open(os.path.join(save_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=4)

    # 打印结果
    print("="*80)
    print(f"✅ 调参完成！最优参数：")
    for k,v in best_params.items():
        if k in param_names:
            print(f"📌 {k}: {v:.4f}" if isinstance(v, float) else f"📌 {k}: {v}")
        elif k in ["AUC", "best_hm"]:
            print(f"📊 {k}: {v:.4f}")
        else:
            print(f"📊 {k}: {v:.2%}")
    print("="*80)

# ====================== 主函数（自动识别阶段+网格搜索）======================

# ====================== 主函数（自动识别阶段+网格搜索，修复字典访问错误）======================
def main():
    # 1. 加载yml配置（字典），从tta节点读取阶段开关，修复字典访问错误
    original_cfg = load_config(CFG_PATH)
    # 核心修复：从tta嵌套节点读取use_robust_cache，字典用[]访问，加默认值避免键不存在
    use_robust_cache = original_cfg.get("tta", {}).get("use_robust_cache", False)
    
    if not use_robust_cache:
        # STAGE1：仅改进一，调lambda_orth+hier_theta
        tune_params = TUNE_PARAMS_STAGE1
        save_dir = f"{SAVE_DIR_PREFIX}stage1_improve1_only"
    else:
        # STAGE2：改进一+二，调correction_interval+sim_threshold（固定改进一）
        tune_params = TUNE_PARAMS_STAGE2
        save_dir = f"{SAVE_DIR_PREFIX}stage2_improve1fixed_improve2"
        # 提前加载阶段1最优参数（仅打印用，modify_config中会实际设置到配置）
        improve1_best = load_improve1_best_params(tune_params["improve1_best_params_path"])
        print(f"📌 固定改进一最优参数：lambda_orth={improve1_best['best_lambda_orth']:.4f}, hier_theta={improve1_best['best_hier_theta']:.4f}")

    # 2. 初始化记录
    csv_path = init_record(save_dir, tune_params["param_names"])
    print("="*80)
    print(f"🚀 开始调参（阶段：{'仅改进一' if not use_robust_cache else '改进一+二'}）")
    print(f"📌 调参参数：{tune_params['param_names']}")
    print(f"📌 参数范围：{tune_params['ranges']}")
    print(f"📌 总实验组数：{len(tune_params['ranges'][0]) * len(tune_params['ranges'][1])}")
    print(f"📌 结果保存至：{save_dir}")
    print("="*80)

    # 3. 网格搜索
    param_combinations = product(*tune_params["ranges"])
    total = len(tune_params['ranges'][0]) * len(tune_params['ranges'][1])
    success = 0

    for idx, param_vals in enumerate(param_combinations, 1):
        print(f"\n{'='*60}")
        print(f"【实验 {idx}/{total}】{dict(zip(tune_params['param_names'], param_vals))}")
        print(f"{'='*60}")
        try:
            # 生成临时配置（modify_config中已处理所有参数设置，包括阶段2固定改进一）
            temp_cfg = modify_config(original_cfg, tune_params, param_vals)
            # 运行实验
            metrics = run_experiment(temp_cfg)
            # 记录数据
            record_data(csv_path, tune_params["param_names"], param_vals, metrics)
            success += 1
            print(f"✅ 成功 | AUC: {metrics['AUC']:.4f} | Best HM: {metrics['best_hm']:.4f}")
        except Exception as e:
            print(f"❌ 失败 | 错误：{str(e)}...")
            continue

    # 4. 可视化+总结
    visualize(save_dir, csv_path, tune_params["param_names"])
    swanlab.finish()
    print(f"\n📊 实验总结：共{total}组 | 成功{success}组 | 失败{total-success}组")
    if success == 0:
        print("❌ 所有实验失败，请检查：1.yml路径/参数是否正确 2.模型/数据集路径是否有效 3.阶段1最优参数文件是否存在（阶段2）")
if __name__ == "__main__":
    main()