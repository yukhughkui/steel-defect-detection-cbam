# -*- coding: utf-8 -*-
"""实验三：小样本条件下的对比（模拟工业标注稀缺场景）
训练集每类 30 张（共180张），3 个随机种子，报告 mean±std
"""
import numpy as np
from torch.utils.data import DataLoader
from common import list_images, CLASSES, save_json, RESULTS_DIR
from exp2_cnn import BaseCNN, DefectDataset, DEVICE, train_model
import torch, time
from sklearn.metrics import accuracy_score, confusion_matrix


def run_setting(model_fn, name, n_per_class, seeds=(42, 7, 2026)):
    import random
    accs, f1s = [], []
    cm_last = None
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random as rd
        rd.seed(seed)
        # 分层抽小训练集
        by_cls = {c: [] for c in range(6)}
        for p, y in list_images('train'):
            by_cls[y].append(p)
        rng = random.Random(seed)
        small = []
        for c in range(6):
            files = sorted(by_cls[c])
            rng.shuffle(files)
            small.extend(files[:n_per_class])
        # 复用 train_model 的训练逻辑但换数据集：直接内联
        tr_items = [(p, CLASSES.index(p.rsplit('\\', 1)[-1].rsplit('_', 1)[0])) for p in small]
        va_items = list_images('train')  # 验证用全部剩余太大 -> 用 test 之外的固定 270
        # 从全量里去掉训练样本作为验证
        va_items = [(p, y) for p, y in va_items if p not in set(small)]
        rng.shuffle(va_items)
        va_items = va_items[:270]
        tr_loader = DataLoader(DefectDataset(tr_items, True), batch_size=32, shuffle=True)
        va_loader = DataLoader(DefectDataset(va_items, False), batch_size=32)
        te_loader = DataLoader(DefectDataset(list_images('test'), False), batch_size=32)

        model = model_fn().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        crit = torch.nn.CrossEntropyLoss()
        best_val, best_state = 0.0, None
        for ep in range(50):
            model.train()
            for x, y in tr_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward(); opt.step()
            sched.step()
            model.eval(); correct = total = 0
            with torch.no_grad():
                for x, y in va_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            va = correct / total
            if va >= best_val:
                best_val = va
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in te_loader:
                out = model(x.to(DEVICE)).argmax(1).cpu()
                y_true.extend(y.tolist()); y_pred.extend(out.tolist())
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc); f1s.append(acc)
        cm_last = confusion_matrix(y_true, y_pred).tolist()
        print(f'[{name} seed={seed}] test acc = {acc:.4f}')
    res = {
        'method': name, 'n_per_class': n_per_class, 'seeds': list(seeds),
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'acc_list': [round(a, 4) for a in accs],
        'confusion_matrix_last': cm_last, 'classes': CLASSES,
    }
    return res


if __name__ == '__main__':
    out = []
    r = run_setting(lambda: BaseCNN(use_cbam=False), 'BaseCNN(小样本)', 30)
    out.append(r)
    r = run_setting(lambda: BaseCNN(use_cbam=True), 'CBAM-CNN(小样本)', 30)
    out.append(r)
    save_json('exp3_fewshot.json', out)
