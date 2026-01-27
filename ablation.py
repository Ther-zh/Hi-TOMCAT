import os
import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import swanlab
from datetime import datetime

# ====================== 全局配置 ======================
BASE_CFG = "config/ut-zappos.yml"
MODEL_WEIGHT = "./result/ut-zappos/20250305_2/test_best.pt"
SAVE_DIR = "./ablation_results"
DEVICE = "cuda:0"
SEED = 0
SWANLAB_PROJECT = "TOMCAT-Ablation-UT-Zappos"

# 消融实验组
ABLATION_GROUPS = {
    "Full_Model_TOMCAT": {
        "model_use_adapter": True,
        "tta_use_img_cache": True,
        "tta_shot_capacity": 3,
        "tta_theta": 1,
        "tta_use_align_loss": True,
        "test_open_world": False,
        "display_name": "Full Model (TOMCAT)"
    },
    "w_o_Text_KAM": {
        "model_use_adapter": False,
        "tta_use_img_cache": True,
        "tta_shot_capacity": 3,
        "tta_theta": 1,
        "tta_use_align_loss": True,
        "test_open_world": False,
        "display_name": "w/o Text-KAM"
    },
    "w_o_Visual_KAM": {
        "model_use_adapter": True,
        "tta_use_img_cache": False,
        "tta_shot_capacity": 3,
        "tta_theta": 1,
        "tta_use_align_loss": True,
        "test_open_world": False,
        "display_name": "w/o Visual-KAM"
    },
    "w_o_Priority_Queue": {
        "model_use_adapter": True,
        "tta_use_img_cache": False,
        "tta_shot_capacity": 0,
        "tta_theta": 1,
        "tta_use_align_loss": True,
        "test_open_world": False,
        "display_name": "w/o Priority Queue"
    },
    "w_o_AUW": {
        "model_use_adapter": True,
        "tta_use_img_cache": True,
        "tta_shot_capacity": 3,
        "tta_theta": 0,
        "tta_use_align_loss": True,
        "test_open_world": False,
        "display_name": "w/o AUW"
    },
    "w_o_Align_Loss": {
        "model_use_adapter": True,
        "tta_use_img_cache": True,
        "tta_shot_capacity": 3,
        "tta_theta": 1,
        "tta_use_align_loss": False,
        "test_open_world": False,
        "display_name": "w/o Align Loss"
    },
    "Baseline_w_o_All_Modules": {
        "model_use_adapter": False,
        "tta_use_img_cache": False,
        "tta_shot_capacity": 0,
        "tta_theta": 0,
        "tta_use_align_loss": False,
        "test_open_world": False,
        "display_name": "Baseline (w/o All Modules)"
    }
}

CORE_METRICS = ["AUC", "best_hm", "best_seen", "best_unseen", "attr_acc", "obj_acc"]
# 优化正则表达式：兼容不同空格数量，匹配更宽松
METRIC_PATTERNS = {
    "AUC": r"AUC\s+(\d+\.\d+)",
    "best_hm": r"best_hm\s+(\d+\.\d+)",
    "best_seen": r"best_seen\s+(\d+\.\d+)",
    "best_unseen": r"best_unseen\s+(\d+\.\d+)",
    "attr_acc": r"attr_acc\s+(\d+\.\d+)",
    "obj_acc": r"obj_acc\s+(\d+\.\d+)"
}

# ====================== 初始化配置 ======================
# 解决matplotlib中文字体警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ====================== 工具函数 ======================
def create_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "configs"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "figures"), exist_ok=True)

def generate_ablation_cfg(exp_key, module_config):
    """生成配置文件"""
    cfg_filename = f"{exp_key}.yml"
    cfg_path = os.path.join(SAVE_DIR, "configs", cfg_filename)
    
    with open(BASE_CFG, "r", encoding="utf-8") as f:
        cfg_lines = f.readlines()
    
    new_cfg_lines = []
    current_section = ""
    
    for line in cfg_lines:
        stripped_line = line.strip()
        if stripped_line.startswith("model:"):
            current_section = "model"
            new_cfg_lines.append(line)
        elif stripped_line.startswith("train:"):
            current_section = "train"
            new_cfg_lines.append(line)
        elif stripped_line.startswith("test:"):
            current_section = "test"
            new_cfg_lines.append(line)
        elif stripped_line.startswith("tta:"):
            current_section = "tta"
            new_cfg_lines.append(line)
        elif stripped_line.startswith("others:"):
            current_section = "others"
            new_cfg_lines.append(line)
        elif stripped_line == "" or stripped_line.startswith("#"):
            new_cfg_lines.append(line)
        else:
            if current_section == "model" and "use_adapter:" in line:
                new_line = f"  use_adapter: {module_config['model_use_adapter']}\n"
                new_cfg_lines.append(new_line)
            elif current_section == "test" and "open_world:" in line:
                new_line = f"  open_world: {module_config['test_open_world']}\n"
                new_cfg_lines.append(new_line)
            elif current_section == "tta":
                if "shot_capacity:" in line:
                    new_line = f"  shot_capacity: {module_config['tta_shot_capacity']}\n"
                    new_cfg_lines.append(new_line)
                elif "use_img_cache:" in line:
                    new_line = f"  use_img_cache: {module_config['tta_use_img_cache']}\n"
                    new_cfg_lines.append(new_line)
                elif "theta:" in line:
                    new_line = f"  theta: {module_config['tta_theta']}\n"
                    new_cfg_lines.append(new_line)
                elif "use_align_loss:" in line:
                    new_line = f"  use_align_loss: {module_config['tta_use_align_loss']}\n"
                    new_cfg_lines.append(new_line)
                else:
                    new_cfg_lines.append(line)
            else:
                new_cfg_lines.append(line)
    
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(new_cfg_lines)
    
    print(f"✅ 生成配置文件：{cfg_path}")
    return cfg_path

def extract_metrics_from_log(log_path):
    """从日志提取指标（兼容部分指标缺失）"""
    metrics = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        
        # 提取每个指标，允许部分缺失
        for metric_name, pattern in METRIC_PATTERNS.items():
            match = re.search(pattern, log_content)
            if match:
                value = float(match.group(1))
                # 统一转换为小数（百分比/100）
                metrics[metric_name] = value / 100.0 if metric_name in ["AUC", "best_seen", "best_unseen", "attr_acc", "obj_acc"] else value / 100.0
            else:
                print(f"⚠️  实验日志中未找到指标 {metric_name}（日志路径：{log_path}）")
                metrics[metric_name] = None
        
        # 判断是否有足够的有效指标（至少3个）
        valid_metrics = [v for v in metrics.values() if v is not None]
        if len(valid_metrics) >= 3:
            return metrics
        else:
            return None
    except Exception as e:
        print(f"❌ 解析日志失败：{e}（日志路径：{log_path}）")
        return None

def run_ablation_exp(exp_key, module_config):
    """运行单个实验（兼容失败组）"""
    print(f"\n{'='*50}")
    display_name = module_config["display_name"]
    print(f"开始消融实验：{display_name}")
    
    # 生成配置文件
    cfg_path = generate_ablation_cfg(exp_key, module_config)
    
    # 运行测试命令
    cmd = [
        "python", "swanlab_test.py",
        "--cfg", cfg_path,
        "--load_model", MODEL_WEIGHT
    ]
    
    # 保存日志
    log_filename = f"{exp_key}.log"
    log_path = os.path.join(SAVE_DIR, "logs", log_filename)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())
            log_file.write(line)
        process.wait()
    
    # 提取指标
    metrics = extract_metrics_from_log(log_path)
    if metrics is None:
        print(f"❌ 实验{display_name}失败：未提取到足够的有效指标")
        return None
    
    # 构造结果
    result = {
        "exp_name": display_name,
        "exp_key": exp_key,
        "cfg_path": cfg_path,
        "log_path": log_path,
        **metrics
    }
    
    # 打印提取的指标（仅显示有效指标）
    print(f"✅ 实验{display_name}成功，提取的有效指标：")
    for k, v in metrics.items():
        if v is not None:
            if k == "AUC":
                print(f"  - {k}: {v:.4f} (原始值：{v*100:.2f}%)")
            elif k == "best_hm":
                print(f"  - {k}: {v:.4f} (原始值：{v*100:.2f}%)")
            else:
                print(f"  - {k}: {v:.4f} (原始值：{v*100:.2f}%)")
    
    return result

def summarize_results(results):
    """汇总结果（过滤掉None）"""
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        print("❌ 无有效实验结果可汇总")
        return {}
    
    summary = {}
    for res in valid_results:
        exp_name = res["exp_name"]
        summary[exp_name] = {metric: res[metric] for metric in CORE_METRICS}
    
    # 保存JSON
    json_path = os.path.join(SAVE_DIR, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存CSV（处理None值为空）
    csv_path = os.path.join(SAVE_DIR, "ablation_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # 表头
        f.write("实验名称," + ",".join(CORE_METRICS) + "\n")
        # 数据行
        for exp_name, metrics in summary.items():
            values = []
            for m in CORE_METRICS:
                val = metrics[m]
                if val is None:
                    values.append("")
                elif m in ["AUC", "best_hm"]:
                    values.append(f"{val:.4f}")
                else:
                    values.append(f"{val:.4f}")
            f.write(f"{exp_name}," + ",".join(values) + "\n")
    
    print(f"\n✅ 结果汇总保存：")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    return summary

def plot_ablation_results(summary):
    """绘制对比图（仅使用有效数据）"""
    if not summary:
        print("❌ 无有效数据可绘图")
        return
    
    exp_names = list(summary.keys())
    # 过滤掉指标为None的数据
    def get_valid_values(metric_name):
        values = []
        for exp in exp_names:
            val = summary[exp][metric_name]
            values.append(val if val is not None else 0.0)
        return values
    
    # 获取有效指标值
    auc_values = get_valid_values("AUC")
    best_hm_values = [v * 100 for v in get_valid_values("best_hm")]  # 转为百分比
    best_seen_values = [v * 100 for v in get_valid_values("best_seen")]
    best_unseen_values = [v * 100 for v in get_valid_values("best_unseen")]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("TOMCAT Ablation Experiment Results (Closed World)", fontsize=20, fontweight='bold', y=0.95)
    
    # 颜色配置
    color_full = "#C73E1D"  # 完整模型红色
    color_ablation = "#2E86AB"  # 消融组蓝色
    colors = [color_full if "Full Model" in exp else color_ablation for exp in exp_names]
    
    # 子图1：AUC
    axes[0,0].bar(range(len(exp_names)), auc_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    axes[0,0].set_title("AUC", fontsize=16, fontweight='bold')
    axes[0,0].set_xticks(range(len(exp_names)))
    axes[0,0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0,0].grid(axis='y', alpha=0.3)
    for i, val in enumerate(auc_values):
        if val > 0:  # 仅显示非零值
            axes[0,0].text(i, val + 0.005, f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    
    # 子图2：Best HM (%)
    axes[0,1].bar(range(len(exp_names)), best_hm_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    axes[0,1].set_title("Best HM (%)", fontsize=16, fontweight='bold')
    axes[0,1].set_xticks(range(len(exp_names)))
    axes[0,1].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0,1].grid(axis='y', alpha=0.3)
    for i, val in enumerate(best_hm_values):
        if val > 0:
            axes[0,1].text(i, val + 0.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=10)
    
    # 子图3：Best Seen Accuracy (%)
    axes[1,0].bar(range(len(exp_names)), best_seen_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    axes[1,0].set_title("Best Seen Accuracy (%)", fontsize=16, fontweight='bold')
    axes[1,0].set_xticks(range(len(exp_names)))
    axes[1,0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1,0].grid(axis='y', alpha=0.3)
    for i, val in enumerate(best_seen_values):
        if val > 0:
            axes[1,0].text(i, val + 0.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=10)
    
    # 子图4：Best Unseen Accuracy (%)
    axes[1,1].bar(range(len(exp_names)), best_unseen_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    axes[1,1].set_title("Best Unseen Accuracy (%)", fontsize=16, fontweight='bold')
    axes[1,1].set_xticks(range(len(exp_names)))
    axes[1,1].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1,1].grid(axis='y', alpha=0.3)
    for i, val in enumerate(best_unseen_values):
        if val > 0:
            axes[1,1].text(i, val + 0.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=10)
    
    # 保存图片
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plot_path = os.path.join(SAVE_DIR, "figures", "ablation_comparison.png")
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 对比图保存：{plot_path}")

# ====================== 主函数 ======================
def main():
    create_dirs()
    
    # 初始化SwanLab（简化版，避免API错误）
    try:
        run = swanlab.init(project=SWANLAB_PROJECT, config={"ablation_groups": [v["display_name"] for v in ABLATION_GROUPS.values()]})
        swanlab_run_id = run.id if hasattr(run, 'id') else "unknown"
    except Exception as e:
        print(f"⚠️  SwanLab初始化警告：{e}")
        swanlab_run_id = "unknown"
    
    all_results = []
    
    # 运行所有消融组
    for exp_key, module_config in ABLATION_GROUPS.items():
        result = run_ablation_exp(exp_key, module_config)
        if result is not None:
            all_results.append(result)
            # 记录到SwanLab（仅记录有效指标）
            try:
                log_data = {"exp_name": result["exp_name"]}
                for metric in ["AUC", "best_hm", "best_seen", "best_unseen"]:
                    val = result[metric]
                    if val is not None:
                        log_data[metric] = val * 100 if metric != "AUC" else val
                swanlab.log(log_data)
            except Exception as e:
                print(f"⚠️  SwanLab日志记录警告：{e}")
    
    # 汇总+绘图
    summary = summarize_results(all_results)
    plot_ablation_results(summary)
    
    # 输出最终信息（移除错误的get_run_url）
    print(f"\n{'='*60}")
    print("TOMCAT 消融实验完成！")
    print(f"成功实验组数：{len(all_results)}/{len(ABLATION_GROUPS)}")
    print(f"结果目录：{SAVE_DIR}")
    print(f"SwanLab项目名称：{SWANLAB_PROJECT}")
    print(f"SwanLab运行ID：{swanlab_run_id}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()