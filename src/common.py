# -*- coding: utf-8 -*-
"""NEU-DET 钢材表面缺陷识别实验 - 公共模块
数据划分：train 1350 / val 270 / test 180（按类别分层，seed=42）
"""
import os
import json
import random
import numpy as np
import torch

CLASSES = ['crazing', 'inclusion', 'patches',
           'pitted_surface', 'rolled-in_scale', 'scratches']
CN_NAMES = {'crazing': '裂纹(crazing)', 'inclusion': '夹杂(inclusion)',
            'patches': '斑块(patches)', 'pitted_surface': '麻点(pitted_surface)',
            'rolled-in_scale': '氧化铁皮压入(rolled-in_scale)', 'scratches': '划痕(scratches)'}
# 仓库根目录（src/ 的上一级），与本文件所在位置无关，放到任何机器上都能跑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, 'datasets', 'NEU-DET')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(split):
    """返回 [(路径, 类别索引)] 列表"""
    d = os.path.join(DATA_ROOT, split, 'images')
    items = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(('.jpg', '.png', '.bmp')):
            cls = f.rsplit('_', 1)[0]
            items.append((os.path.join(d, f), CLASSES.index(cls)))
    return items


def make_split():
    """从 train(1620) 分层切出 train_core(1350) / val(270)；test(180) 保持不变"""
    by_cls = {c: [] for c in range(6)}
    for p, y in list_images('train'):
        by_cls[y].append(p)
    rng = random.Random(SEED)
    train_core, val = [], []
    for c in range(6):
        files = sorted(by_cls[c])
        rng.shuffle(files)
        val.extend(files[:45])
        train_core.extend(files[45:])
    return train_core, val


def save_json(name, obj):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print('[saved]', path)
    return path


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name), encoding='utf-8') as f:
        return json.load(f)
