# -*- coding: utf-8 -*-
"""实验一：传统机器学习基线 - HOG 特征 + SVM（RBF 核）
在 CPU 上运行，输出测试集指标与混淆矩阵
"""
import time
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from common import (CLASSES, list_images, make_split, set_seed, save_json)

set_seed()


def load_gray(path, size=128):
    img = Image.open(path).convert('L').resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def hog_feat(gray_img):
    f = hog(gray_img, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm='L2-Hys')
    return f


def extract(items):
    X, y = [], []
    for p, label in items:
        X.append(hog_feat(load_gray(p)))
        y.append(label)
    return np.array(X), np.array(y)


def main():
    train_paths, val_paths = make_split()
    test_items = list_images('test')
    train_items = [(p, CLASSES.index(p.rsplit('\\', 1)[-1].rsplit('_', 1)[0])) for p in train_paths]
    val_items = [(p, CLASSES.index(p.rsplit('\\', 1)[-1].rsplit('_', 1)[0])) for p in val_paths]
    train_all = train_items + val_items   # SVM 不需验证集，用全部 1620 训练

    t0 = time.perf_counter()
    X_train, y_train = extract(train_all)
    X_test, y_test = extract(test_items)
    feat_time = time.perf_counter() - t0
    print(f'特征提取完成: train {X_train.shape}, test {X_test.shape}, 耗时 {feat_time:.1f}s')

    t0 = time.perf_counter()
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42))
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = (time.perf_counter() - t0) / len(X_test) * 1000

    res = {
        'method': 'HOG + SVM (RBF, C=10)',
        'feature_dim': int(X_train.shape[1]),
        'n_train': len(X_train), 'n_test': len(X_test),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classes': CLASSES,
        'feature_time_s': round(feat_time, 1),
        'train_time_s': round(train_time, 1),
        'infer_time_ms_per_img': round(infer_time, 2),
    }
    for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
        print(k, '=', round(res[k], 4))
    save_json('exp1_svm.json', res)


if __name__ == '__main__':
    main()
