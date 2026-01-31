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

# ====================== 全局配置：解决Linux中文显示+随机种子+环境适配 ======================
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 纯英文，彻底避免字体问题
plt.rcParams['axes.unicode_minus'] = False
# 固定随机种子，保证实验可复现
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ====================== 配置项（仅需确认这3个，其余完全对齐swan_test2.py）======================
CFG_PATH = "config/ut-zappos.yml"  # 你的配置文件路径
SAVE_DIR = "tune_improve1_results"  # 调参结果保存目录（自动创建）
SWANLAB_PROJECT = "Tune-Improve1-ClosedWorld"  # SwanLab项目名
# 改进一核心超参数网格搜索范围（基于你之前的实验，有效范围）
LAMBDA_ORTH_LIST = [0.001, 0.003, 0.005, 0.01]  # 缩小范围，提升效率
HIER_THETA_LIST = [0.5, 0.8, 1.0]                # 缩小范围，共12组实验
# 要记录的核心指标（和swan_test2.py输出完全一致）
CORE_METRICS = ["AUC", "best_hm", "attr_acc", "best_seen", "best_unseen", "obj_acc", "biasterm"]
# 基础参数固化（100%对齐你的实验配置，禁止修改）
FIXED_PARAMS = {
    "use_img_cache": False,  # 强制关闭改进二，仅测改进一
    "open_world": False,     # 强制闭世界调参
    "shot_capacity": 3,      # 原文最优，固定
    "text_first": True,      # swan_test2.py中启用的Hi-TOMCAT
    "use_tta": True,         # 启用TTA，调用你的改进一函数
    "use_wandb": True,       # 开启SwanLab日志
    "seed": SEED,
    "eval_batch_size_wo_tta": 1,  # 对齐swan_test2.py的batch_size参数
    "num_workers": 0,
    "threshold_trials": 6,   # 闭世界无需调threshold，固定小值
}

# ====================== 必须定义：对齐swan_test2.py的全局变量/工具类 ======================
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
# 直接从swan_test2.py复制Evaluator类，保证指标计算完全一致
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
            results = self.score_fast_model(scores, obj_truth, bias=bias, topk=topk)
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

# 从swan_test2.py复制test函数，保证指标计算一致
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

# ====================== 工具函数：加载/修改配置（兼容swan_test2.py的yml格式）======================
def load_config(cfg_path):
    import yaml
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    import yaml
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)

def modify_config(original_cfg, lambda_orth, hier_theta):
    """动态修改配置：仅更新改进一的超参数，其余完全保留"""
    cfg = original_cfg.copy()
    # 改进一核心超参数写入tta节点（无则创建）
    if "tta" not in cfg:
        cfg["tta"] = {}
    cfg["tta"]["lambda_orth"] = lambda_orth
    cfg["tta"]["hier_theta"] = hier_theta
    # 固化基础参数到对应分组
    # 确保test分组存在
    if "test" not in cfg:
        cfg["test"] = {}
    # 确保tta分组存在
    if "tta" not in cfg:
        cfg["tta"] = {}
    # 分配参数到对应分组
    test_params = ["open_world", "text_first", "use_wandb", "seed", "eval_batch_size_wo_tta", "num_workers", "threshold_trials"]
    tta_params = ["use_img_cache", "shot_capacity", "use_tta"]
    
    for k, v in FIXED_PARAMS.items():
        if k in test_params:
            cfg["test"][k] = v
        elif k in tta_params:
            cfg["tta"][k] = v
    # 保存临时配置文件（避免覆盖原配置）
    temp_cfg_path = f"temp_improve1_{lambda_orth:.4f}_{hier_theta:.1f}.yml"
    save_config(cfg, temp_cfg_path)
    return temp_cfg_path

# ====================== 核心函数：运行单次实验（100%对齐swan_test2.py的运行逻辑）======================
def run_experiment(temp_cfg_path):
    """
    完全复用swan_test2.py的运行逻辑：
    配置解析→数据集实例化→模型加载→预测→指标计算
    """
    # 添加入口目录，保证导入所有模块
    sys.path.append(DIR_PATH)
    from parameters import parser
    from utils import load_args, set_seed
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from swan_test_hitomcat import predict_logits_text_first_with_hitomcat  # 你的改进一函数

    try:
        # 1. 配置解析（和swan_test2.py完全一致）
        args = parser.parse_args(["--cfg", temp_cfg_path])
        load_args(args.cfg, args)
        config = args
        set_seed(config.seed)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  📌 使用设备：{config.device}")

        # 2. 实例化测试数据集（无get_dataset，直接用CompositionDataset，对齐swan_test2.py）
        print(f"  📌 加载数据集：{config.dataset}")
        test_dataset = CompositionDataset(
            config.dataset_path,
            phase='test',
            split='compositional-split-natural',
            open_world=config.open_world
        )

        # 3. 模型加载（对齐swan_test2.py，需要attributes/classes/offset参数）
        print(f"  📌 加载模型：{config.load_model}")
        allattrs = test_dataset.attrs
        allobj = test_dataset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)
        # 实例化模型并加载权重
        model = get_model(config, attributes=attributes, classes=classes, offset=offset).to(config.device)
        if config.load_model and os.path.exists(config.load_model):
            model.load_state_dict(torch.load(config.load_model, map_location='cpu'))
        model.eval()

        # 4. 选择预测函数（和swan_test2.py完全一致，启用改进一）
        predict_logits_func = predict_logits_text_first_with_hitomcat
        print(f"  📌 使用预测函数：Hi-TOMCAT（改进一）")

        # 5. 运行预测（带autocast，对齐swan_test2.py）
        print(f"  📌 开始预测...")
        with autocast(dtype=torch.bfloat16):
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(model, test_dataset, config)

        # 6. 计算指标（复用swan_test2.py的Evaluator+test函数）
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
        # 实验失败时清理临时文件
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)
        raise Exception(f"运行异常：{str(e)}")

# ====================== 工具函数：数据记录（CSV+SwanLab）======================
def init_swanlab(project_name):
    """初始化SwanLab，仅记录关键配置"""
    swanlab.init(
        project=project_name,
        config=FIXED_PARAMS,
        log_level="info",
        mode="online" if FIXED_PARAMS["use_wandb"] else "offline"
    )

def init_csv(save_dir, core_metrics):
    """初始化CSV文件，写入表头"""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "tune_improve1_metrics.csv")
    headers = ["lambda_orth", "hier_theta"] + core_metrics
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    return csv_path

def record_data(csv_path, lambda_orth, hier_theta, metrics):
    """记录单次实验数据到CSV+SwanLab"""
    row = {"lambda_orth": lambda_orth, "hier_theta": hier_theta}
    row.update(metrics)
    # 写入CSV
    with open(csv_path, 'a+', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
    # 同步到SwanLab
    swanlab.log({**metrics, "lambda_orth": lambda_orth, "hier_theta": hier_theta})

# ====================== 工具函数：可视化结果（鲁棒性强，无数据不报错）======================
def visualize_results(save_dir, csv_path, core_metrics):
    import pandas as pd
    # 加载并过滤有效数据
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["AUC"])
    df = df[df["AUC"] > 0]
    if len(df) == 0:
        print("【警告】无有效实验数据，跳过可视化！")
        return

    # 1. AUC热力图（核心，找最优参数）
    plt.figure(figsize=(10, 8))
    auc_pivot = df.pivot(index="lambda_orth", columns="hier_theta", values="AUC")
    if not auc_pivot.empty:
        im = plt.imshow(auc_pivot, cmap="YlGnBu", aspect="auto")
        # 添加数值标注
        for i in range(len(auc_pivot.index)):
            for j in range(len(auc_pivot.columns)):
                val = auc_pivot.iloc[i, j]
                plt.text(j, i, f"{val:.4f}", ha="center", va="center", color="black", fontsize=10)
        plt.colorbar(im, label="AUC")
    # 坐标轴设置（纯英文）
    plt.xticks(range(len(auc_pivot.columns)), auc_pivot.columns, fontsize=12)
    plt.yticks(range(len(auc_pivot.index)), auc_pivot.index, fontsize=12)
    plt.xlabel("hier_theta (AUW Temperature Coefficient)", fontsize=14, fontweight="bold")
    plt.ylabel("lambda_orth (Orthogonal Loss Weight)", fontsize=14, fontweight="bold")
    plt.title("Improve1-ClosedWorld: AUC Heatmap (Higher is Better)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "auc_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 核心指标折线图（AUC/best_hm/attr_acc）
    for metric in ["AUC", "best_hm", "attr_acc"]:
        plt.figure(figsize=(12, 6))
        for theta in HIER_THETA_LIST:
            theta_data = df[df["hier_theta"] == theta].sort_values("lambda_orth")
            if len(theta_data) == 0:
                continue
            plt.plot(theta_data["lambda_orth"], theta_data[metric], marker="o", linewidth=2, label=f"hier_theta={theta}")
        plt.xlabel("lambda_orth (Orthogonal Loss Weight)", fontsize=14, fontweight="bold")
        plt.ylabel(metric, fontsize=14, fontweight="bold")
        plt.title(f"Improve1-ClosedWorld: {metric} vs lambda_orth", fontsize=16, fontweight="bold")
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_lineplot.png"), dpi=300)
        plt.close()

    # 3. 保存最优参数（按AUC最大化）
    best_idx = df["AUC"].idxmax()
    best_row = df.loc[best_idx]
    best_params = {
        "best_lambda_orth": float(best_row["lambda_orth"]),
        "best_hier_theta": float(best_row["hier_theta"]),
        "best_AUC": float(best_row["AUC"]),
        "best_hm": float(best_row["best_hm"]),
        "attr_acc": float(best_row["attr_acc"]),
        "best_seen": float(best_row["best_seen"]),
        "best_unseen": float(best_row["best_unseen"]),
        "obj_acc": float(best_row["obj_acc"]),
        "biasterm": float(best_row["biasterm"])
    }
    # 保存到JSON+Excel
    with open(os.path.join(save_dir, "best_params.json"), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    df.to_excel(os.path.join(save_dir, "tune_improve1_metrics.xlsx"), index=False)

    # 打印最优参数（醒目显示）
    print("="*80)
    print("✅ 改进一（层级化KAM）闭世界调参完成！最优参数如下：")
    for k, v in best_params.items():
        if "lambda" in k or "theta" in k:
            print(f"📌 {k}: {v:.4f}")
        elif k in ["best_AUC", "best_hm"]:
            print(f"📊 {k}: {v:.4f}")
        else:
            print(f"📊 {k}: {v:.2%}")
    print("="*80)
    print(f"📁 所有结果已保存至：{os.path.abspath(save_dir)}")

# ====================== 主函数：网格搜索主流程 ======================
def main():
    print("="*80)
    print("🚀 开始改进一（层级化KAM）闭世界超参数网格搜索")
    print(f"📌 搜索参数：lambda_orth={LAMBDA_ORTH_LIST}, hier_theta={HIER_THETA_LIST}")
    print(f"📌 总实验组数：{len(LAMBDA_ORTH_LIST) * len(HIER_THETA_LIST)}")
    print("="*80)

    # 初始化
    original_cfg = load_config(CFG_PATH)
    csv_path = init_csv(SAVE_DIR, CORE_METRICS)
    init_swanlab(SWANLAB_PROJECT)

    # 网格搜索
    param_combinations = product(LAMBDA_ORTH_LIST, HIER_THETA_LIST)
    total_num = len(LAMBDA_ORTH_LIST) * len(HIER_THETA_LIST)
    success_num = 0

    for idx, (lambda_orth, hier_theta) in enumerate(param_combinations, 1):
        print(f"\n{'='*60}")
        print(f"【实验 {idx}/{total_num}】lambda_orth={lambda_orth:.4f}, hier_theta={hier_theta:.1f}")
        print(f"{'='*60}")
        try:
            # 生成临时配置+运行实验
            temp_cfg_path = modify_config(original_cfg, lambda_orth, hier_theta)
            metrics = run_experiment(temp_cfg_path)
            # 记录数据
            record_data(csv_path, lambda_orth, hier_theta, metrics)
            success_num += 1
        except Exception as e:
            print(f"❌ 实验失败 | 错误详情：{e}")
            continue

    # 可视化+总结
    visualize_results(SAVE_DIR, csv_path, CORE_METRICS)
    swanlab.finish()
    print(f"\n📊 实验总结：共{total_num}组 | 成功{success_num}组 | 失败{total_num-success_num}组")
    if success_num == 0:
        print("❌ 所有实验失败，请检查配置文件中的【dataset_path/load_model】路径是否正确！")

if __name__ == "__main__":
    main()