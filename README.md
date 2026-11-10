# 钢材表面缺陷识别：HOG+SVM / BaseCNN / CBAM-CNN 三级对比

基于 NEU-DET 数据集（热轧带钢 6 类表面缺陷，1800 幅 200×200 灰度图）的对比研究型实验：
传统手工特征（HOG+SVM）→ 轻量卷积网络（BaseCNN）→ 注意力增强网络（CBAM-CNN），
外加小样本实验与 Grad-CAM 可解释性分析。

## 结果一览（180 幅测试图）

| 方法 | 准确率 |
|---|---|
| HOG + SVM | 88.33% |
| BaseCNN | 99.44% |
| CBAM-CNN | 97.78% |

小样本（每类 30 幅、3 个随机种子）下 CBAM-CNN 平均准确率比 BaseCNN 高 1.67 个百分点，
表明通道-空间注意力在样本稀缺时更稳。Grad-CAM 热力图显示 CBAM-CNN 的判别区域
与缺陷实际位置吻合，具备可解释性。

## 环境

```bash
pip install -r requirements.txt
```

有 NVIDIA GPU 时装 CUDA 版 torch 可大幅加速 exp2/exp3；纯 CPU 也能跑通，只是慢。

## 数据集

NEU-DET 需自行下载（约 33MB），放到如下位置：

```
datasets/NEU-DET/
├── train/
│   ├── images/   # 1620 张，文件名如 crazing_1.jpg（类别_编号）
│   └── labels/
└── test/
    ├── images/   # 180 张
    └── labels/
```

下载来源（任选其一）：
- [NEU-DET-with-yolov8](https://github.com/Marfbin/NEU-DET-with-yolov8)（本实验所用划分的出处）
- 或搜索 "NEU-DET dataset download"，东北大学发布的公开数据集

## 运行

```bash
# 顺序执行全部实验（exp1 → exp2 → exp3 → gradcam → make_figures）
python run_all.py

# 或单独运行某一步（在仓库根目录执行）
python src/exp1_svm.py       # HOG+SVM 基线，CPU 几分钟
python src/exp2_cnn.py       # 训练 BaseCNN 与 CBAM-CNN，GPU 约 15 分钟
python src/exp3_fewshot.py   # 小样本对比，需先跑 exp2
python src/gradcam.py        # Grad-CAM 热力图，需 results/cbam.pt
python src/make_figures.py   # 生成全部图表，需 results/*.json
```

## 开箱即玩

仓库自带 `results/`（训练好的 `base.pt`、`cbam.pt` 与四个实验 JSON，约 7MB），
clone 后不训练也能直接跑后两步看效果：

```bash
python src/gradcam.py       # 出 Grad-CAM 热力图
python src/make_figures.py  # 重现论文全部图表
```

## 目录结构

```
├── run_all.py            # 一键顺序运行全部实验
├── requirements.txt
├── src/
│   ├── common.py         # 公共配置：类别、数据划分（train 1350 / val 270 / test 180，seed=42）
│   ├── exp1_svm.py       # 实验 1：HOG+SVM
│   ├── exp2_cnn.py       # 实验 2：BaseCNN 与 CBAM-CNN
│   ├── exp3_fewshot.py   # 实验 3：小样本（每类 30 幅 × 3 种子）
│   ├── gradcam.py        # Grad-CAM 可解释性分析
│   └── make_figures.py   # 图表生成
└── results/              # 训练产物（权重 + JSON + 图表，随仓库分发）
```

## 说明

- 数据划分固定 seed=42，划分结果可复现；GPU 训练存在约 ±1 幅图量级的非确定性波动，
  重跑结果与上表可能有极小出入，属正常现象。
- CBAM 引用自 Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018；
  Grad-CAM 引用自 Selvaraju et al., ICCV 2017。
