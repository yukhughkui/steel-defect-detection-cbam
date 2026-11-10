# -*- coding: utf-8 -*-
"""生成论文全套图表（matplotlib 学术风格，中文标注，300dpi）"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from common import CLASSES, CN_NAMES, list_images, RESULTS_DIR, DATA_ROOT

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name), encoding='utf-8') as f:
        return json.load(f)


def savefig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', p)


# ---------- 图1 数据集样本 ----------
def fig_samples():
    rng = np.random.RandomState(42)
    rows, cols = 2, 6
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 5.0))
    by_cls = {c: [] for c in range(6)}
    for p, y in list_images('train'):
        by_cls[y].append(p)
    for c in range(6):
        for r in range(rows):
            p = rng.choice(by_cls[c])
            img = Image.open(p).convert('L')
            ax = axes[r, c]
            ax.imshow(img, cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(CN_NAMES[CLASSES[c]], fontsize=11)
    savefig(fig, 'fig1_samples.png')


# ---------- 图2 方法流程 ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 2.8))
    ax.axis('off')
    blocks = [
        ('缺陷图像\n200×200\n灰度化/128×128', '#eef3fb'),
        ('卷积块1-2\n16/32通道\nMaxPool', '#e8f1e8'),
        ('卷积块3-4\n64/128通道\n+ CBAM 注意力', '#fdf3e3'),
        ('全局平均池化\n128维特征', '#efe8f5'),
        ('全连接分类\n6类缺陷', '#fdeaea'),
    ]
    x = 0.4
    for text, color in blocks:
        box = FancyBboxPatch((x, 0.32), 2.05, 1.05,
                             boxstyle='round,pad=0.08', fc=color, ec='#4a5568', lw=1.2)
        ax.add_patch(box)
        ax.text(x + 1.02, 0.85, text, ha='center', va='center', fontsize=10.5)
        if x + 2.05 < 11.5:
            arrow = FancyArrowPatch((x + 2.18, 0.85), (x + 2.52, 0.85),
                                    arrowstyle='-|>', mutation_scale=18, color='#4a5568')
            ax.add_patch(arrow)
        x += 2.45
    # CBAM 内部说明
    ax.text(5.35, 0.13, 'CBAM = 通道注意力(全局池化→MLP→Sigmoid) + 空间注意力(7×7卷积→Sigmoid)',
            fontsize=9.5, color='#718096', ha='center')
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 1.7)
    savefig(fig, 'fig2_pipeline.png')


# ---------- 图3 训练曲线 ----------
def fig_curves():
    base = load_json('exp2_base.json')
    cbam = load_json('exp2_cbam.json')
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ep = range(1, len(base['history']['train_acc']) + 1)
    ax.plot(ep, base['history']['train_acc'], '--', color='#a0aec0', lw=1.4, label='BaseCNN 训练')
    ax.plot(ep, base['history']['val_acc'], '-', color='#3182ce', lw=1.8, label='BaseCNN 验证')
    ax.plot(ep, cbam['history']['train_acc'], '--', color='#f6ad55', lw=1.4, label='CBAM-CNN 训练')
    ax.plot(ep, cbam['history']['val_acc'], '-', color='#dd6b20', lw=1.8, label='CBAM-CNN 验证')
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=11)
    ax.set_ylabel('准确率', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    savefig(fig, 'fig3_curves.png')


# ---------- 图4 混淆矩阵三联 ----------
def fig_confusion():
    svm = load_json('exp1_svm.json')
    base = load_json('exp2_base.json')
    cbam = load_json('exp2_cbam.json')
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    short = ['裂纹', '夹杂', '斑块', '麻点', '氧化皮', '划痕']
    for ax, res, title in zip(axes, [svm, base, cbam],
                              ['(a) HOG+SVM', '(b) BaseCNN', '(c) CBAM-CNN']):
        cm = np.array(res['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues')
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=9)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(short, fontsize=9, rotation=30)
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel('预测类别', fontsize=10)
        ax.set_ylabel('真实类别', fontsize=10)
        acc = res['accuracy'] if 'accuracy' in res else res.get('acc_mean')
        ax.set_title(f"{title}  Acc={acc*100:.2f}%", fontsize=11)
    savefig(fig, 'fig4_confusion.png')


# ---------- 图5 小样本对比 ----------
def fig_fewshot():
    data = load_json('exp3_fewshot.json')
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels, means, stds, allvals = [], [], [], []
    for r in data:
        labels.append(r['method'].replace('(', '\n('))
        means.append(r['acc_mean'] * 100)
        stds.append(r['acc_std'] * 100)
        allvals.append([a * 100 for a in r['acc_list']])
    x = np.arange(len(labels))
    bars = ax.bar(x, means, width=0.45, color=['#a0aec0', '#dd6b20'], capsize=6,
                  yerr=stds, error_kw=dict(ecolor='#444', lw=1.4))
    for i, (m, vals) in enumerate(zip(means, allvals)):
        ax.scatter([x[i]] * 3, vals, color='#2d3748', zorder=3, s=22)
        ax.text(x[i], m + 1.2, f'{m:.2f}%', ha='center', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('测试集准确率 (%)', fontsize=11)
    ax.set_xlabel('模型（每类训练样本 30 张，3 个随机种子）', fontsize=10)
    ax.set_ylim(85, 100)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, 'fig5_fewshot.png')


# ---------- 图6 Grad-CAM ----------
def fig_gradcam():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'gradcam_raw', '*.npz')))
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 10.0))
    for i, f in enumerate(files[:12]):
        d = np.load(f)
        gray, cam = d['gray'], d['cam']
        ax = axes[i // 4, i % 4]
        ax.imshow(gray, cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([])
        true = CLASSES[int(d['true'])]
        ax.set_title(CN_NAMES[true], fontsize=11)
    fig.suptitle('Grad-CAM 类激活热力图（红=高响应，蓝=低响应）', fontsize=13)
    savefig(fig, 'fig6_gradcam.png')


if __name__ == '__main__':
    fig_samples()
    fig_pipeline()
    fig_curves()
    fig_confusion()
    fig_fewshot()
    fig_gradcam()
    print('全部图表生成完毕')
