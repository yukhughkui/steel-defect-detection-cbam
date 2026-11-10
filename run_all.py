# -*- coding: utf-8 -*-
"""NEU-DET 钢材表面缺陷识别：HOG+SVM / BaseCNN / CBAM-CNN 三级对比实验

运行前请把 NEU-DET 数据集放到 datasets/NEU-DET/（见 README.md）。
带 results/ 里的权重与 JSON 时，可跳过训练直接跑 gradcam / make_figures。
"""

# 阶段一：传统方法基线（HOG + SVM，CPU 约几分钟）
# python src/exp1_svm.py

# 阶段二：深度模型对比（BaseCNN 与 CBAM-CNN 各训练一次，GPU 约十几分钟，CPU 会慢很多）
# python src/exp2_cnn.py

# 阶段三：小样本实验（每类 30 幅、3 个随机种子，需先跑完 exp2）
# python src/exp3_fewshot.py

# 可解释性分析：Grad-CAM 热力图（需先跑完 exp2 得到 results/cbam.pt）
# python src/gradcam.py

# 生成论文用图（需先跑完 exp1/exp2/exp3 得到 results/*.json）
# python src/make_figures.py

# 统一入口：按顺序全部执行
if __name__ == '__main__':
    import os
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for script in ['exp1_svm.py', 'exp2_cnn.py', 'exp3_fewshot.py',
                   'gradcam.py', 'make_figures.py']:
        print(f'\n========== {script} ==========')
        subprocess.run([sys.executable, os.path.join(here, 'src', script)],
                       check=True)
