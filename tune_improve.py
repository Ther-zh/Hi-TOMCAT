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
        [0.0001, 0.0005, 0.0008, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007],  # lambda_orth范围
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]                        # hier_theta范围
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
        return {**self.generate_predictions(scores, obj_truth, bias, topk), "scores": scores.clone()}

    def evaluate_predictions(self, preds, attr_gt, obj_gt, pair_gt, allpred, topk=1):
        from scipy.stats import hmean
        attr_gt, obj_gt, pair_gt = attr_gt.cpu(), obj_gt.cpu(), pair_gt.cpu()
        seen_ind = torch.tensor([i for i,(a,o) in enumerate(zip(attr_gt.numpy(), obj_gt.numpy())) if (a,o) in self.train_pairs])
        unseen_ind = torch.tensor([i for i,(a,o) in enumerate(zip(attr_gt.numpy(), obj_gt.numpy())) if (a,o) not in self.train_pairs])

        def process(s):
            a_match = (attr_gt.unsqueeze(1).repeat(1,topk) == s[0][:,:topk]).any(1).float()
            o_match = (obj_gt.unsqueeze(1).repeat(1,topk) == s[1][:,:topk]).any(1).float()
            match = (a_match * o_match).float()
            return a_match, o_match, match, match[seen_ind], match[unseen_ind], torch.ones(512,5), torch.ones(512,5), torch.ones(512,5)

        stats = {}
        for k in ["closed", "unbiased_closed"]:
            a,o,m,s,u,sc,ss,su = process(preds[k])
            stats[f"{k}_attr_match"] = a.mean().item()
            stats[f"{k}_obj_match"] = o.mean().item()
            stats[f"{k}_match"] = m.mean().item()
            stats[f"{k}_seen_match"] = s.mean().item() if len(s) else 0.0
            stats[f"{k}_unseen_match"] = u.mean().item() if len(u) else 0.0

        scores = preds["scores"]
        correct_scores = scores[torch.arange(len(scores)), pair_gt][unseen_ind]
        max_seen = scores[unseen_ind][:, self.seen_mask].topk(topk,1)[0][:,topk-1]
        diff = max_seen - correct_scores
        valid_diff = diff[stats["closed_unseen_match"]>0] - 1e-4
        biaslist = valid_diff[::max(len(valid_diff)//20,1)] if len(valid_diff) else [0.0]

        seen_acc, unseen_acc = [stats["closed_seen_match"]], [stats["closed_unseen_match"]]
        base_scores = torch.stack([allpred[(a,o)] for a,o in self.dset.pairs], 1)
        for b in biaslist:
            s,u = process(self.score_fast_model(base_scores.clone(), obj_gt, b, topk))[3:5]
            seen_acc.append(s.mean().item() if len(s) else 0.0)
            unseen_acc.append(u.mean().item() if len(u) else 0.0)

        seen_acc, unseen_acc = np.array(seen_acc), np.array(unseen_acc)
        hm = hmean([seen_acc, unseen_acc], axis=0) if len(seen_acc) else 0.0
        return {
            **stats,
            "AUC": np.trapz(seen_acc, unseen_acc),
            "best_hm": np.max(hm) if len(hm) else 0.0,
            "best_seen": np.max(seen_acc),
            "best_unseen": np.max(unseen_acc),
            "biasterm": biaslist[np.argmax(hm)] if len(hm) else 1e3
        }

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        scores[~self.seen_mask.repeat(scores.shape[0],1)] += bias
        closed = scores.masked_fill(~self.closed_mask.repeat(scores.shape[0],1), -1e10)
        _, pred = closed.topk(topk,1)
        pred = pred.view(-1)
        return (self.pairs[pred][:,0].view(-1,topk), self.pairs[pred][:,1].view(-1,topk))

def test(test_dset, evaluator, logits, attr_gt, obj_gt, pair_gt, config):
    preds = {p: logits[:,i] for i,p in enumerate(test_dset.pairs)}
    all_pred = torch.stack([preds[(a,o)] for a,o in test_dset.pairs], 1)
    res = evaluator.score_model(preds, obj_gt, 1e3, 1)
    attr_acc = (res['unbiased_closed'][0].squeeze(-1) == attr_gt).float().mean().item()
    obj_acc = (res['unbiased_closed'][1].squeeze(-1) == obj_gt).float().mean().item()
    stats = evaluator.evaluate_predictions(res, attr_gt, obj_gt, pair_gt, preds, 1)
    return {**stats, "attr_acc": attr_acc, "obj_acc": obj_acc}

# ====================== 核心工具函数（适配yml读取+调参逻辑）======================
def load_config(cfg_path):
    """加载yml配置，返回Namespace对象（和swan_test2.py一致）"""
    import yaml
    from parameters import parser
    args = parser.parse_args(["--cfg", cfg_path])
    from utils import load_args
    load_args(args.cfg, args)
    return args

def load_improve1_best_params(params_path):
    """加载阶段1最优改进一参数（阶段2用）"""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"请先运行阶段1（use_robust_cache=False）生成{params_path}")
    with open(params_path, 'r') as f:
        return json.load(f)

def modify_config(original_cfg, tune_params, param_values):
    """
    动态修改配置：仅覆盖当前要调的2个参数，其余保留yml值
    original_cfg: 从yml加载的原始配置
    tune_params: 当前阶段的调参配置（param_names + ranges）
    param_values: 本次实验的2个参数值
    """
    cfg = copy.deepcopy(original_cfg)
    # 仅覆盖要调的2个参数
    for param_name, param_val in zip(tune_params["param_names"], param_values):
        setattr(cfg, param_name, param_val)
    # 生成临时配置文件（基于原始yml修改，仅改2个参数）
    temp_cfg_path = f"temp_tune_{'_'.join([str(v) for v in param_values])}.yml"
    import yaml
    with open(temp_cfg_path, 'w') as f:
        yaml.dump(vars(cfg), f, sort_keys=False)
    return temp_cfg_path

# ====================== 实验运行+记录+可视化（适配动态调参）======================
def run_experiment(temp_cfg_path):
    """运行单次实验（复用swan_test2.py逻辑）"""
    sys.path.append(DIR_PATH)
    from dataset import CompositionDataset
    from model.model_factory import get_model
    from swan_test_hitomcat import predict_logits_text_first_with_hitomcat

    try:
        # 加载临时配置（仅改了2个调参参数）
        config = load_config(temp_cfg_path)
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载数据集+模型（全部从yml读参数）
        test_dset = CompositionDataset(
            config.dataset_path, phase='test', split='compositional-split-natural', open_world=config.open_world
        )
        allattrs = [a.replace("."," ").lower() for a in test_dset.attrs]
        allobj = [o.replace("."," ").lower() for o in test_dset.objs]
        model = get_model(config, attributes=allattrs, classes=allobj, offset=len(allattrs)).to(config.device)
        if config.load_model and os.path.exists(config.load_model):
            model.load_state_dict(torch.load(config.load_model, map_location='cpu'))
        model.eval()

        # 预测+计算指标
        with autocast(dtype=torch.bfloat16):
            logits, attr_gt, obj_gt, pair_gt = predict_logits_text_first_with_hitomcat(model, test_dset, config)
        evaluator = Evaluator(test_dset, model, config.device)
        stats = test(test_dset, evaluator, logits, attr_gt, obj_gt, pair_gt, config)

        # 清理临时文件
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)
        return {k: stats.get(k, 0.0) for k in CORE_METRICS}
    except Exception as e:
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)
        raise Exception(f"实验失败：{str(e)}")

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
def main():
    # 1. 加载yml配置，自动识别阶段
    original_cfg = load_config(CFG_PATH)
    use_robust_cache = original_cfg.use_robust_cache
    if not use_robust_cache:
        # STAGE1：仅改进一，调lambda_orth+hier_theta
        tune_params = TUNE_PARAMS_STAGE1
        save_dir = f"{SAVE_DIR_PREFIX}stage1_improve1_only"
        swanlab_project = "Tune-Stage1-Improve1-Only"
    else:
        # STAGE2：改进一+二，调correction_interval+sim_threshold（固定改进一）
        tune_params = TUNE_PARAMS_STAGE2
        save_dir = f"{SAVE_DIR_PREFIX}stage2_improve1fixed_improve2"
        swanlab_project = "Tune-Stage2-Improve1Fixed+Improve2"
        # 加载阶段1最优改进一参数，固定到配置
        improve1_best = load_improve1_best_params(tune_params["improve1_best_params_path"])
        setattr(original_cfg, "lambda_orth", improve1_best["best_lambda_orth"])
        setattr(original_cfg, "hier_theta", improve1_best["best_hier_theta"])
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
            temp_cfg = modify_config(original_cfg, tune_params, param_vals)
            metrics = run_experiment(temp_cfg)
            record_data(csv_path, tune_params["param_names"], param_vals, metrics)
            success += 1
            print(f"✅ 成功 | AUC: {metrics['AUC']:.4f} | Best HM: {metrics['best_hm']:.4f}")
        except Exception as e:
            print(f"❌ 失败 | 错误：{str(e)[:100]}...")
            continue

    # 4. 可视化+总结
    visualize(save_dir, csv_path, tune_params["param_names"])
    swanlab.finish()
    print(f"\n📊 总结：共{total}组 | 成功{success}组 | 失败{total-success}组")

if __name__ == "__main__":
    main()