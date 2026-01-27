import json
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family']='simhei'
# ====================== 全局美化配置（所有图表统一风格） ======================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['savefig.dpi'] = 300  # 所有图统一高清保存

# 专业配色方案
COLOR_ACC = '#2E86AB'      # 准确率类
COLOR_SEEN = '#A23B72'     # Seen指标
COLOR_UNSEEN = '#F18F01'   # Unseen指标
COLOR_CORE = '#C73E1D'     # 核心指标（AUC/HM）
COLOR_HM = '#7209B7'       # 调和均值专属

# ====================== 解析数据 ======================
result_json = '''
{
    "test": {
        "closed_attr_match": 0.509608805179596,
        "closed_obj_match": 0.7635552287101746,
        "closed_match": 0.4680851101875305,
        "closed_seen_match": 0.0,
        "closed_unseen_match": 0.7213114500045776,
        "biasterm": 2.9999001026153564,
        "best_unseen": 0.7213114500045776,
        "best_seen": 0.6930596232414246,
        "AUC": 0.4424373754727018,
        "hm_unseen": 0.5658381581306458,
        "hm_seen": 0.572825014591217,
        "best_hm": 0.5693101506263041,
        "attr_acc": 0.5909402966499329,
        "obj_acc": 0.7460535168647766
    }
}
'''
data = json.loads(result_json)['test']

# 按类别整理指标
# 1. 基础准确率指标
basic_metrics = {
    '属性准确率 (attr_acc)': data['attr_acc'],
    '对象准确率 (obj_acc)': data['obj_acc'],
    '闭合配对准确率 (closed_match)': data['closed_match']
}
# 2. Seen/Unseen最佳准确率
seen_unseen_metrics = {
    '最佳Seen准确率': data['best_seen'],
    '最佳Unseen准确率': data['best_unseen']
}
# 3. HM调和均值指标
hm_metrics = {
    'HM-Seen': data['hm_seen'],
    'HM-Unseen': data['hm_unseen'],
    '最佳调和均值 (Best HM)': data['best_hm']
}
# 4. 核心指标（AUC + Best HM）
core_metrics = {
    'AUC': data['AUC'],
    '最佳调和均值 (Best HM)': data['best_hm'] * 100  # 转为百分比
}

# ====================== 工具函数（复用绘图逻辑） ======================
def draw_bar_chart(metrics, title, ylabel, save_path, colors):
    """绘制柱状图并保存"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(metrics.keys())
    y = [v * 100 for v in metrics.values()]  # 转为百分比
    
    # 绘制柱状图
    bars = ax.bar(x, y, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # 美化配置
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_ylim(0, max(y) * 1.2)  # 预留120%的高度放数值标注
    ax.grid(axis='y', linestyle='--')
    
    # 添加数值标注
    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{val:.2f}%',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )
    
    # 旋转x轴标签（避免重叠）
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 图表已保存：{save_path}")

def draw_core_card_chart(metrics, title, save_path):
    """绘制核心指标卡片图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # 隐藏坐标轴
    
    # 绘制两个卡片（AUC + Best HM）
    card_width = 0.35
    card_height = 0.7
    # AUC卡片（左侧）
    ax.add_patch(plt.Rectangle((0.15, 0.15), card_width, card_height,
                               facecolor=COLOR_CORE, alpha=0.8, edgecolor='white', linewidth=3))
    ax.text(0.325, 0.75, 'AUC', fontsize=20, fontweight='bold', ha='center', color='white')
    ax.text(0.325, 0.45, f'{metrics["AUC"]:.4f}', fontsize=28, fontweight='bold', ha='center', color='white')
    
    # Best HM卡片（右侧）
    ax.add_patch(plt.Rectangle((0.525, 0.15), card_width, card_height,
                               facecolor=COLOR_HM, alpha=0.8, edgecolor='white', linewidth=3))
    ax.text(0.7, 0.75, 'Best HM', fontsize=20, fontweight='bold', ha='center', color='white')
    ax.text(0.7, 0.45, f'{metrics["最佳调和均值 (Best HM)"]:.2f}%', fontsize=28, fontweight='bold', ha='center', color='white')
    
    # 总标题
    ax.text(0.5, 0.95, title, fontsize=18, fontweight='bold', ha='center')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 图表已保存：{save_path}")

# ====================== 生成4张独立图表 ======================
# 图1：基础准确率指标
draw_bar_chart(
    metrics=basic_metrics,
    title='CZSL 基础准确率指标',
    ylabel='准确率 (%)',
    save_path='czsl_basic_accuracy.png',
    colors=[COLOR_ACC, COLOR_ACC, COLOR_ACC]
)

# 图2：Seen/Unseen最佳准确率
draw_bar_chart(
    metrics=seen_unseen_metrics,
    title='CZSL 最佳Seen/Unseen准确率',
    ylabel='准确率 (%)',
    save_path='czsl_seen_unseen_best.png',
    colors=[COLOR_SEEN, COLOR_UNSEEN]
)

# 图3：HM调和均值指标
draw_bar_chart(
    metrics=hm_metrics,
    title='CZSL 调和均值 (HM) 指标',
    ylabel='调和均值 (%)',
    save_path='czsl_harmonic_mean.png',
    colors=[COLOR_SEEN, COLOR_UNSEEN, COLOR_HM]
)

# 图4：核心指标卡片图
draw_core_card_chart(
    metrics=core_metrics,
    title='CZSL 核心评估指标',
    save_path='czsl_core_metrics.png'
)

# ====================== 文本结果汇总 ======================
print("\n" + "="*60)
print("CZSL 模型评估结果汇总")
print("="*60)
print("📊 基础准确率：")
for name, val in basic_metrics.items():
    print(f"  - {name}: {val*100:.2f}%")
print("\n📊 Seen/Unseen 准确率：")
for name, val in seen_unseen_metrics.items():
    print(f"  - {name}: {val*100:.2f}%")
print("\n📊 调和均值指标：")
for name, val in hm_metrics.items():
    print(f"  - {name}: {val*100:.2f}%")
print("\n🎯 核心指标：")
print(f"  - AUC: {core_metrics['AUC']:.4f}")
print(f"  - Best HM: {core_metrics['最佳调和均值 (Best HM)']:.2f}%")
print("="*60)