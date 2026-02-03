import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================== ✅ 仅需用户修改这2项 ✅ ======================
SAVE_DIR = "tune_improve_final"    # 调参结果CSV所在的保存目录（和原脚本一致）
CORE_METRICS = ["AUC", "best_hm", "attr_acc", "best_seen", "best_unseen", "obj_acc", "biasterm"]  # 和原脚本一致

# ====================== 全局配置：解决Linux/Windows中文/负号显示问题 ======================
plt.switch_backend('Agg')  # 无GUI环境兼容（Linux服务器必备）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']  # 多系统字体兼容
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块问题
plt.rcParams['figure.dpi'] = 100  # 基础分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片高分辨率
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# ====================== 核心函数：加载并验证CSV数据 ======================
def load_and_validate_data(save_dir):
    """
    自动识别调参场景的CSV文件，加载并过滤有效数据
    :return: df(有效数据DataFrame), csv_suffix(场景后缀), csv_path(CSV文件路径)
    """
    # 遍历目录，匹配调参生成的CSV文件（两种场景：robustcache_on/off）
    csv_path = None
    csv_suffix = ""
    for file in os.listdir(save_dir):
        if file.startswith("tune_metrics_") and file.endswith(".csv"):
            csv_path = os.path.join(save_dir, file)
            if "robustcache_on" in file:
                csv_suffix = "robustcache_on"
            else:
                csv_suffix = "robustcache_off"
            break
    
    # 校验CSV文件是否存在
    if csv_path is None:
        raise FileNotFoundError(f"在目录 {save_dir} 中未找到调参CSV文件，请检查SAVE_DIR配置是否正确！")
    
    # 加载数据并过滤无效值（剔除空值、AUC<=0的实验）
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.dropna(subset=["AUC"])  # 删除AUC为空的行
    df = df[df["AUC"] > 0]          # 保留AUC>0的有效实验
    
    # 校验有效数据
    if len(df) == 0:
        raise ValueError("CSV文件中无有效实验数据（所有AUC为空或<=0），无法可视化！")
    
    print(f"✅ 成功加载数据：{csv_path}")
    print(f"✅ 调参场景：{csv_suffix} | 有效实验数：{len(df)}")
    return df, csv_suffix, csv_path

# ====================== 核心函数：自动识别调参参数 ======================
def get_tune_params(df):
    """
    从CSV表头自动识别调参参数（前2列为调参参数1/2）
    :return: param1(参数1名), param2(参数2名), vals1(参数1唯一值), vals2(参数2唯一值)
    """
    param1 = df.columns[0]
    param2 = df.columns[1]
    # 对参数值排序，保证可视化顺序和原调参范围一致
    vals1 = sorted(df[param1].unique())
    vals2 = sorted(df[param2].unique())
    print(f"✅ 自动识别调参参数：{param1} × {param2}")
    print(f"✅ {param1}范围：{vals1}")
    print(f"✅ {param2}范围：{vals2}")
    return param1, param2, vals1, vals2

# ====================== 可视化函数：热力图（AUC/best_hm/attr_acc） ======================
def plot_heatmaps(df, param1, param2, save_dir, csv_suffix):
    """绘制核心指标热力图，标注具体数值，保存高分辨率图片"""
    metrics = ["AUC", "best_hm", "attr_acc"]
    for metric in metrics:
        plt.figure(figsize=(10, 8))
        # 构建透视表（参数1为行，参数2为列，指标为值）
        pivot = df.pivot(index=param1, columns=param2, values=metric)
        # 绘制热力图，配色为蓝黄渐变（数值越高颜色越深）
        im = plt.imshow(pivot, cmap="YlGnBu", aspect="auto")
        # 标注每个单元格的具体数值（保留4位小数）
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = plt.text(j, i, f"{pivot.iloc[i, j]:.4f}",
                                ha="center", va="center", fontsize=10,
                                color="black" if pivot.iloc[i, j] < pivot.max().max()*0.7 else "white")
        # 配置图表元素
        plt.colorbar(im, label=metric, shrink=0.8)
        plt.xlabel(param2, fontsize=14, labelpad=10)
        plt.ylabel(param1, fontsize=14, labelpad=10)
        plt.title(f"{metric} Heatmap (Higher is Better)", fontsize=16, pad=20)
        plt.xticks(range(len(pivot.columns)), [f"{x:.4f}" for x in pivot.columns], fontsize=12)
        plt.yticks(range(len(pivot.index)), [f"{y:.4f}" for y in pivot.index], fontsize=12)
        # 保存图片，避免重名
        save_path = os.path.join(save_dir, f"{metric}_heatmap_{csv_suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"📊 热力图已保存：{save_path}")

# ====================== 可视化函数：折线图（AUC/best_hm/attr_acc） ======================
def plot_lineplots(df, param1, param2, vals2, save_dir, csv_suffix):
    """绘制核心指标折线图，按参数2分组，直观展示参数1对指标的影响"""
    metrics = ["AUC", "best_hm", "attr_acc"]
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        # 按参数2的每个值分组绘制折线
        for p2_val in vals2:
            p2_data = df[df[param2] == p2_val].sort_values(param1)
            if len(p2_data) == 0:
                continue
            plt.plot(p2_data[param1], p2_data[metric],
                     marker="o", markersize=6, linewidth=2,
                     label=f"{param2}={p2_val:.4f}")
        # 配置图表元素
        plt.xlabel(param1, fontsize=14, labelpad=10)
        plt.ylabel(metric, fontsize=14, labelpad=10)
        plt.title(f"Parameter Tune: {metric} vs {param1}", fontsize=16, pad=20)
        plt.legend(loc="best", fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle="-")
        plt.tick_params(axis="both", labelsize=12)
        # 保存图片
        save_path = os.path.join(save_dir, f"{metric}_lineplot_{csv_suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"📊 折线图已保存：{save_path}")

# ====================== 结果保存：最优参数+全量数据 ======================
def save_best_results(df, param1, param2, core_metrics, save_dir, csv_suffix):
    """
    按AUC最大化筛选最优参数，保存：
    1. 最优参数到JSON文件
    2. 全量有效数据到Excel文件
    并打印醒目最优结果
    """
    # 按AUC降序排序，取第一行为最优参数
    df_sorted = df.sort_values("AUC", ascending=False)
    best_row = df_sorted.iloc[0]
    # 构造最优参数字典
    best_params = {
        f"best_{param1}": round(float(best_row[param1]), 4),
        f"best_{param2}": round(float(best_row[param2]), 4),
        "best_AUC": round(float(best_row["AUC"]), 4),
        **{k: round(float(best_row[k]), 4) for k in core_metrics if k not in ["AUC"]}
    }
    # 保存最优参数到JSON
    best_json_path = os.path.join(save_dir, f"best_params_{csv_suffix}.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)
    # 保存全量有效数据到Excel（方便后续分析）
    excel_path = os.path.join(save_dir, f"tune_metrics_valid_{csv_suffix}.xlsx")
    df_sorted.to_excel(excel_path, index=False, engine="openpyxl")
    # 醒目打印最优结果
    print("\n" + "="*80)
    print(f"✅ 调参最优结果（按AUC最大化筛选）| 场景：{csv_suffix}")
    print("="*80)
    print(f"📌 最优{param1}：{best_params[f'best_{param1}']}")
    print(f"📌 最优{param2}：{best_params[f'best_{param2}']}")
    print("-"*80)
    for k in core_metrics:
        val = best_params[k] if k in best_params else best_row[k]
        if k in ["obj_acc", "attr_acc", "best_seen", "best_unseen"]:
            print(f"📊 {k:12s}：{val:.2%}")  # 百分比显示
        else:
            print(f"📊 {k:12s}：{val:.4f}")   # 小数显示
    print("="*80)
    print(f"📁 最优参数保存：{best_json_path}")
    print(f"📁 有效数据保存：{excel_path}")
    print("="*80)

# ====================== 主函数：串联所有可视化流程 ======================
def main():
    try:
        # 1. 加载并验证CSV数据
        df, csv_suffix, _ = load_and_validate_data(SAVE_DIR)
        # 2. 自动识别调参参数
        param1, param2, _, vals2 = get_tune_params(df)
        # 3. 绘制热力图
        plot_heatmaps(df, param1, param2, SAVE_DIR, csv_suffix)
        # 4. 绘制折线图
        plot_lineplots(df, param1, param2, vals2, SAVE_DIR, csv_suffix)
        # 5. 筛选并保存最优参数
        save_best_results(df, param1, param2, CORE_METRICS, SAVE_DIR, csv_suffix)
        print("\n🎉 所有可视化任务完成！结果已保存至：", os.path.abspath(SAVE_DIR))
    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
    except ValueError as e:
        print(f"\n❌ 错误：{e}")
    except Exception as e:
        print(f"\n❌ 未知错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()