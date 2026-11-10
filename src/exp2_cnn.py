# -*- coding: utf-8 -*-
"""实验二：CNN 基线 与 CBAM 注意力改进 CNN（PyTorch, GPU）
同一训练配置下对比两个模型，输出训练曲线、测试指标、混淆矩阵
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from common import (CLASSES, list_images, make_split, set_seed, save_json,
                    RESULTS_DIR)

set_seed()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH = 32
LR = 1e-3
IMG = 128   # 输入尺寸 128x128


# ---------------- 数据 ----------------
class DefectDataset(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert('L').resize((IMG, IMG))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0)   # 1xHxW
        if self.train:   # 轻量增广：翻转 + 小角度旋转
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2])
            k = np.random.randint(-1, 2)      # -90/0/90 度
            x = torch.rot90(x, k, dims=[1, 2])
        return x, y


# ---------------- CBAM 模块 ----------------
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False))
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.mlp(self.avg(x)) + self.mlp(self.mx(x)))


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        return x * self.act(self.conv(s))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ---------------- 网络 ----------------
class BaseCNN(nn.Module):
    """4 个卷积块 + 全局池化分类头"""
    def __init__(self, use_cbam=False, width=(16, 32, 64, 128)):
        super().__init__()
        self.use_cbam = use_cbam
        self.features = nn.ModuleList()
        in_ch = 1
        for w in width:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_ch, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)))
            if use_cbam:
                self.features.append(CBAM(w))
            in_ch = w
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width[-1], 6))

    def forward(self, x):
        for m in self.features:
            x = m(x)
        return self.head(x)

    def cam_layer(self):
        """Grad-CAM 目标层：最后一个卷积块的输出"""
        convs = [m for m in self.features if isinstance(m, nn.Sequential)]
        return convs[-1]


def train_model(model, name):
    train_paths, val_paths = make_split()
    tr_items = [(p, y) for p, y in list_images('train') if p in set(train_paths)]
    va_items = [(p, y) for p, y in list_images('train') if p in set(val_paths)]
    tr_loader = DataLoader(DefectDataset(tr_items, True), batch_size=BATCH,
                           shuffle=True, num_workers=0, drop_last=False)
    va_loader = DataLoader(DefectDataset(va_items, False), batch_size=BATCH)
    te_loader = DataLoader(DefectDataset(list_images('test'), False), batch_size=BATCH)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}
    best_val, best_state = 0.0, None
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()
        tr_acc, tr_loss = correct / total, loss_sum / total
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        va_acc = correct / total
        history['train_acc'].append(round(tr_acc, 4))
        history['val_acc'].append(round(va_acc, 4))
        history['train_loss'].append(round(tr_loss, 4))
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f'[{name}] ep{ep:02d} train_acc={tr_acc:.3f} val_acc={va_acc:.3f}')
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)   # 用验证集最优权重
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'{name}.pt'))

    # 测试
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for x, y in te_loader:
            out = model(x.to(DEVICE)).argmax(1).cpu()
            y_true.extend(y.tolist()); y_pred.extend(out.tolist())
    infer_ms = (time.perf_counter() - t0) / len(y_true) * 1000

    res = {
        'method': 'CBAM-CNN (ours)' if name == 'cbam' else 'BaseCNN',
        'epochs': EPOCHS, 'batch': BATCH, 'lr': LR, 'img_size': IMG,
        'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classes': CLASSES,
        'history': history,
        'train_time_s': round(train_time, 1),
        'infer_time_ms_per_img': round(infer_ms, 2),
    }
    print(f"[{name}] test acc = {res['accuracy']:.4f}")
    save_json(f'exp2_{name}.json', res)
    return res


if __name__ == '__main__':
    print('device:', DEVICE)
    train_model(BaseCNN(use_cbam=False), 'base')
    train_model(BaseCNN(use_cbam=True), 'cbam')
