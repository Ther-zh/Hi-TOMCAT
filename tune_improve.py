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

# ====================== 全局配置：解决Linux中文显示+固定随机种子（通用配置，无需修改） ======================
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 纯英文避字体问题
plt.rcParams['axes.unicode_minus'] = False
# 固定随机种子保证可复现（若需修改，在YML中设置seed，脚本会读取）
BASE_SEED = 42
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====================== ✅ 仅需用户修改这5项 ✅ 其余全由YML控制 ======================
CFG_PATH = "config/ut-zappos.yml"  # 你的主YML配置文件路径
SAVE_DIR = "tune_improve_on2"    # 调参结果保存目录（自动创建）
SWANLAB_PROJECT = "Tune-Improve2"  # SwanLab项目名
# 调参参数范围：按场景预留，脚本会根据YML中的use_robust_cache自动匹配
TUNE_PARAMS_SCOPE = {
    # 场景1：YML中use_robust_cache=False（仅改进一）→ 调这两个
    "lambda_orth": [2,3,4,5,6,7,8,9,10],  # 正交损失权重
    "hier_theta": [4.5,5,5.5,6,6.5,7],               # 自适应更新温度系数
    # 场景2：YML中use_robust_cache=True（改进一+二）→ 调这两个
    "sim_threshold": [0.05,0.7,0.10,0.12,0.15,0.17,0.20],           # 缓存入队相似度阈值
    "correction_interval": [10 ,15, 20,25,30,35,40]          # 缓存周期性修正步长
}
# 要记录的核心指标（和swan_test2.py输出完全一致，无需修改）
CORE_METRICS = ["AUC", "best_hm", "attr_acc", "best_seen", "best_unseen", "obj_acc", "biasterm"]

# ====================== 全局变量：加载YML后自动初始化（用户无需管） ======================
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
original_cfg = None  # 原始YML配置
use_robust_cache = False  # 从YML读取后赋值
TUNE_PARAMS = []  # 自动匹配的待调参数，如["lambda_orth", "hier_theta"]
TUNE_PARAM1, TUNE_PARAM2 = "", ""  # 待调参数1/2
TUNE_VALS1, TUNE_VALS2 = [], []    # 待调参数1/2的范围
TOTAL_EXP_NUM = 0  # 总实验组数，自动计算

# ====================== 完全保留原有正确逻辑：Evaluator类（指标计算核心，一行未改） ======================
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

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        self.score_model = self.score_manifold_model

    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):
        def get_pred_from_scores(_scores, topk):
            _, pair_pred = _scores.topk(topk, dim=1)
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias

        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update({"unbiased_open": get_pred_from_scores(orig_scores, topk)})
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update({"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)})

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        attr_pred, obj_pred = scores
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)
        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores
        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(self.device)
        scores = torch.stack([scores[(attr, obj)] for attr, obj in self.dset.pairs], 1)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        results = {}
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        _, pair_pred = closed_scores.topk(topk, dim=1)
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), self.pairs[pair_pred][:, 1].view(-1, topk)
        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk=1):
        from scipy.stats import hmean
        attr_truth, obj_truth, pair_truth = attr_truth.to("cpu"), obj_truth.to("cpu"), pair_truth.to("cpu")
        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))
        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)
        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)

        def _process(_scores):
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)
            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = ["_attr_match", "_obj_match", "_match", "_seen_match", "_unseen_match", "_ca", "_seen_ca", "_unseen_ca"]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        scores = predictions["scores"]
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]
        unseen_score_diff = max_seen_scores - correct_scores
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")
        base_scores = torch.stack([allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias=bias, topk=1)
            results = results['closed']
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats

# ====================== 完全保留原有正确逻辑：test函数（指标汇总，一行未改） ======================
def test(test_dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config):
    predictions = {pair_name: all_logits[:, i] for i, pair_name in enumerate(test_dataset.pairs)}
    all_pred = [predictions]
    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat([all_pred[i][k] for i in range(len(all_pred))]).float()
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=1)
    attr_acc = float(torch.mean((results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean((results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=1)
    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc
    return stats

# ====================== 配置工具函数：纯YML读取/保存，仅覆盖调参参数 ======================
def load_config(cfg_path):
    import yaml
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    import yaml
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

def modify_config(param1_val, param2_val):
    """
    仅修改调参参数，其余完全保留YML原始配置
    1. 从original_cfg深拷贝，不修改原文件
    2. 仅将两个调参参数写入tta节点（无则自动创建）
    3. 生成临时配置文件，实验后自动删除
    """
    cfg = copy.deepcopy(original_cfg)
    # 确保tta节点存在（YML中无则创建，不修改其他任何节点）
    if "tta" not in cfg:
        cfg["tta"] = {}
    # 仅覆盖调参参数，其余所有参数（包括use_robust_cache/use_img_cache等）均从YML读取
    cfg["tta"][TUNE_PARAM1] = param1_val
    cfg["tta"][TUNE_PARAM2] = param2_val
    # 生成临时配置文件（基于调参参数命名，避免重复）
    temp_cfg_name = f"temp_tune_{TUNE_PARAM1}_{param1_val:.4f}_{TUNE_PARAM2}_{param2_val:.4f}.yml"
    temp_cfg_path = os.path.join(DIR_PATH, temp_cfg_name)
    save_config(cfg, temp_cfg_path)
    return temp_cfg_path

# ====================== 核心初始化：从YML读取use_robust_cache，自动匹配调参参数 ======================
def init_tune_params():
    """
    关键逻辑：
    1. 从YML的tta节点读取use_robust_cache（无则默认False）
    2. 根据use_robust_cache自动匹配待调参的2个参数及范围
    3. 初始化全局调参变量，计算总实验组数
    """
    global original_cfg, use_robust_cache, TUNE_PARAMS, TUNE_PARAM1, TUNE_PARAM2, TUNE_VALS1, TUNE_VALS2, TOTAL_EXP_NUM
    # 加载原始YML
    original_cfg = load_config(CFG_PATH)
    # 从YML的tta节点读取use_robust_cache，容错处理（无则默认False）
    use_robust_cache = original_cfg.get("tta", {}).get("use_robust_cache", False)
    # 自动匹配调参参数
    if not use_robust_cache:
        # 场景1：仅改进一 → 调lambda_orth + hier_theta
        TUNE_PARAMS = ["lambda_orth", "hier_theta"]
    else:
        # 场景2：改进一+二 → 调sim_threshold + correction_interval
        TUNE_PARAMS = ["sim_threshold", "correction_interval"]
    # 初始化调参参数变量
    TUNE_PARAM1, TUNE_PARAM2 = TUNE_PARAMS[0], TUNE_PARAMS[1]
    TUNE_VALS1, TUNE_VALS2 = TUNE_PARAMS_SCOPE[TUNE_PARAM1], TUNE_PARAMS_SCOPE[TUNE_PARAM2]
    TOTAL_EXP_NUM = len(TUNE_VALS1) * len(TUNE_VALS2)
    # 打印调参场景信息（方便用户核对）
    print("="*80)
    print("📌 调参场景自动识别（从YML读取）")
    print(f"📌 use_robust_cache: {use_robust_cache}")
    print(f"📌 待调参数：{TUNE_PARAM1} × {TUNE_PARAM2}")
    print(f"📌 调参范围：{TUNE_VALS1} × {TUNE_VALS2}")
    print(f"📌 总实验组数：{TOTAL_EXP_NUM}")
    print("="*80)

# ====================== 完全保留原有正确逻辑：run_experiment（实验运行核心，一行未改） ======================
def run_experiment(temp_cfg_path):
    """完全复用swan_test2.py的运行逻辑，所有参数从临时YML读取"""
    sys.path.append(DIR_PATH)
    from parameters import parser
    from utils import load_args, set_seed
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from swan_test_hitomcat import predict_logits_text_first_with_hitomcat  # 你的改进预测函数

    try:
        # 1. 配置解析（和swan_test2.py完全一致）
        args = parser.parse_args(["--cfg", temp_cfg_path])
        load_args(args.cfg, args)
        config = args
        # 随机种子从YML读取，无则用BASE_SEED
        try:
            set_seed(config.seed)
        except AttributeError:
            set_seed(BASE_SEED)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  📌 使用设备：{config.device}")

        # 2. 实例化测试数据集（所有参数从YML读取）
        print(f"  📌 加载数据集：{config.dataset}")
        test_dataset = CompositionDataset(
            config.dataset_path,
            phase='test',
            split='compositional-split-natural',
            open_world=config.open_world
        )

        # 3. 模型加载（所有参数从YML读取，对齐swan_test2.py）
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

        # 4. 运行预测（带autocast，和swan_test2.py完全一致）
        print(f"  📌 开始预测...")
        with autocast(dtype=torch.bfloat16):
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_text_first_with_hitomcat(model, test_dataset, config)

        # 5. 计算指标（复用原有正确逻辑）
        print(f"  📌 计算指标...")
        evaluator = Evaluator(test_dataset, model=None, device=config.device)
        test_stats = test(test_dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)

        # 清理临时配置文件
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)

        # 提取核心指标，保证无缺失
        res = {k: test_stats.get(k, 0.0) for k in CORE_METRICS}
        print(f"  ✅ 实验完成 | AUC: {res['AUC']:.4f} | Best HM: {res['best_hm']:.4f}")
        return res

    except Exception as e:
        # 实验失败时强制清理临时文件
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)
        raise Exception(f"运行异常：{str(e)}")

# ====================== 数据记录工具：自适应调参参数，CSV+SwanLab ======================
def init_swanlab(project_name):
    """初始化SwanLab，记录YML中的核心配置和调参信息"""
    swanlab.init(
        project=project_name,
        config={
            "cfg_path": CFG_PATH,
            "use_robust_cache": use_robust_cache,
            "tune_params": TUNE_PARAMS,
            "total_exp_num": TOTAL_EXP_NUM
        },
        log_level="info",
        mode="online" if original_cfg.get("use_wandb", True) else "offline"  # wandb模式从YML读取
    )

def init_csv(save_dir, core_metrics):
    """初始化CSV，表头自动适配当前调参参数"""
    os.makedirs(save_dir, exist_ok=True)
    # 文件名标记调参场景，避免覆盖
    csv_suffix = "robustcache_on" if use_robust_cache else "robustcache_off"
    csv_path = os.path.join(save_dir, f"tune_metrics_{csv_suffix}.csv")
    # 表头：调参参数1 + 调参参数2 + 核心指标
    headers = [TUNE_PARAM1, TUNE_PARAM2] + core_metrics
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    return csv_path

def record_data(csv_path, param1_val, param2_val, metrics):
    """记录单次实验数据，动态适配调参参数"""
    row = {TUNE_PARAM1: param1_val, TUNE_PARAM2: param2_val}
    row.update(metrics)
    # 写入CSV
    with open(csv_path, 'a+', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
    # 同步到SwanLab（含调参参数，方便在线分析）
    swanlab.log({**metrics, TUNE_PARAM1: param1_val, TUNE_PARAM2: param2_val})

# ====================== 可视化工具：自适应调参参数，鲁棒性强 ======================
def visualize_results(save_dir, csv_path, core_metrics):
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    import os  # 补充缺失的os导入，原代码用到了os.path却没导入
    
    # 加载并过滤有效数据（剔除AUC<=0/空值的无效实验）
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["AUC"])
    df = df[df["AUC"] > 0]
    if len(df) == 0:
        print("【警告】无有效实验数据，跳过可视化！")
        return

    # 场景后缀，用于文件名
    csv_suffix = "robustcache_on" if use_robust_cache else "robustcache_off"

    # 1. 多指标热力图（AUC/best_hm/attr_acc），修复刻度+param_names+os缺失问题
    for value in ["AUC", "best_hm", "attr_acc"]:
        plt.figure(figsize=(10,8))
        # 核心修复：用全局TUNE_PARAM1/TUNE_PARAM2替换未定义的param_names
        pivot = df.pivot(index=TUNE_PARAM1, columns=TUNE_PARAM2, values=value)
        # 【可选优化】按数值排序pivot的行列，让热力图按参数大小顺序展示（避免乱序）
        pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
        im = plt.imshow(pivot, cmap="YlGnBu", aspect="auto")
        
        # ===================== 核心修改：设置实际数值刻度 =====================
        # x轴：刻度位置=列索引，刻度标签=pivot列的实际参数值（保留4位小数，可按需修改）
        plt.xticks(
            range(len(pivot.columns)),  # 刻度位置：0,1,2...
            [f"{x:.4f}" for x in pivot.columns],  # 刻度标签：实际参数值
            fontsize=10, 
            rotation=45  # 旋转45度，避免标签重叠（可根据需要改0/30/60）
        )
        # y轴：刻度位置=行索引，刻度标签=pivot行的实际参数值
        plt.yticks(
            range(len(pivot.index)), 
            [f"{x:.4f}" for x in pivot.index], 
            fontsize=10
        )
        # =====================================================================
        
        # 数值标注，保留4位小数
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                plt.text(j, i, f"{pivot.iloc[i,j]:.4f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, label=value)
        # 修复：图表轴标签替换为实际调参参数名
        plt.xlabel(TUNE_PARAM2, fontsize=14, fontweight="bold")
        plt.ylabel(TUNE_PARAM1, fontsize=14, fontweight="bold")
        plt.title(f"{value} Heatmap (Higher is Better)", fontsize=16, fontweight="bold")
        plt.tight_layout()  # 自动调整布局，适配旋转后的标签
        plt.savefig(os.path.join(save_dir, f"{value}_heatmap_{csv_suffix}.png"), dpi=300)
        plt.close()

    # 2. 核心指标折线图（AUC/best_hm/attr_acc），原有逻辑不变
    for metric in ["AUC", "best_hm", "attr_acc"]:
        plt.figure(figsize=(12, 6))
        for param2_val in TUNE_VALS2:
            param2_data = df[df[TUNE_PARAM2] == param2_val].sort_values(TUNE_PARAM1)
            if len(param2_data) == 0:
                continue
            plt.plot(param2_data[TUNE_PARAM1], param2_data[metric], marker="o", linewidth=2, label=f"{TUNE_PARAM2}={param2_val}")
        # 自适应坐标轴
        plt.xlabel(TUNE_PARAM1, fontsize=14, fontweight="bold")
        plt.ylabel(metric, fontsize=14, fontweight="bold")
        plt.title(f"Improve-Tune: {metric} vs {TUNE_PARAM1} (use_robust_cache={use_robust_cache})", fontsize=16, fontweight="bold")
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_lineplot_{csv_suffix}.png"), dpi=300)
        plt.close()

    # 3. 保存最优参数（按AUC最大化，自适应调参参数），原有逻辑不变
    best_idx = df["AUC"].idxmax()
    best_row = df.loc[best_idx]
    best_params = {
        f"best_{TUNE_PARAM1}": float(best_row[TUNE_PARAM1]),
        f"best_{TUNE_PARAM2}": float(best_row[TUNE_PARAM2]),
        **{k: float(best_row[k]) for k in core_metrics}
    }
    # 保存最优参数到JSON，全量调参数据到Excel
    best_param_path = os.path.join(save_dir, f"best_params_{csv_suffix}.json")
    excel_path = os.path.join(save_dir, f"tune_metrics_{csv_suffix}.xlsx")
    with open(best_param_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    df.to_excel(excel_path, index=False)

    # 醒目打印最优参数及核心指标，控制台直观展示
    print("\n" + "="*80)
    print(f"✅ 调参完成！use_robust_cache={use_robust_cache} 最优参数如下：")
    print("="*80)
    print(f"📌 最优{TUNE_PARAM1}：{best_params[f'best_{TUNE_PARAM1}']:.4f}")
    print(f"📌 最优{TUNE_PARAM2}：{best_params[f'best_{TUNE_PARAM2}']:.4f}")
    print("-"*80)
    for k in core_metrics:
        if k in ["AUC", "best_hm", "biasterm"]:
            print(f"📊 {k:12s}：{best_params[k]:.4f}")
        else:
            print(f"📊 {k:12s}：{best_params[k]:.2%}")
    print("="*80)
    print(f"📁 所有调参结果已保存至：{os.path.abspath(save_dir)}")
# ====================== 主函数：网格搜索主流程（全自动，无人工干预） ======================
def main():
    # 第一步：初始化调参参数（从YML读取，自动匹配）
    init_tune_params()
    # 第二步：初始化数据记录（CSV+SwanLab）
    csv_path = init_csv(SAVE_DIR, CORE_METRICS)
    init_swanlab(SWANLAB_PROJECT)

    # 第三步：网格搜索遍历所有参数组合
    param_combinations = product(TUNE_VALS1, TUNE_VALS2)
    success_num = 0
    print(f"\n🚀 开始网格搜索调参，总{TOTAL_EXP_NUM}组实验...")

    for idx, (param1_val, param2_val) in enumerate(param_combinations, 1):
        print(f"\n{'='*60}")
        print(f"【实验 {idx}/{TOTAL_EXP_NUM}】{TUNE_PARAM1}={param1_val:.4f}, {TUNE_PARAM2}={param2_val:.4f}")
        print(f"{'='*60}")
        try:
            # 生成临时YML（仅覆盖调参参数）
            temp_cfg_path = modify_config(param1_val, param2_val)
            # 运行单次实验
            metrics = run_experiment(temp_cfg_path)
            # 记录数据
            record_data(csv_path, param1_val, param2_val, metrics)
            success_num += 1
        except Exception as e:
            print(f"❌ 实验失败 | 错误详情：{e}")
            continue

    # 第四步：可视化结果+保存最优参数
    visualize_results(SAVE_DIR, csv_path, CORE_METRICS)
    # 结束SwanLab日志
    swanlab.finish()

    # 实验总结
    print(f"\n📊 调参实验总结：")
    print(f"📌 总实验组数：{TOTAL_EXP_NUM} | 成功：{success_num} | 失败：{TOTAL_EXP_NUM-success_num}")
    if success_num == 0:
        print("❌ 所有实验失败，请优先检查YML中的【dataset_path/load_model】路径是否正确！")

if __name__ == "__main__":
    main()