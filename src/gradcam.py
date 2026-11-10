# -*- coding: utf-8 -*-
"""Grad-CAM：对 CBAM 模型生成类激活热力图，验证其聚焦缺陷区域"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import CLASSES, list_images, RESULTS_DIR
from exp2_cnn import BaseCNN, DEVICE

MODEL_PATH = os.path.join(RESULTS_DIR, 'cbam.pt')
OUT_DIR = os.path.join(RESULTS_DIR, 'gradcam_raw')
IMG = 128


class GradCAM:
    def __init__(self, model):
        self.model = model.to(DEVICE).eval()
        self.feats = None
        self.grads = None
        target = model.cam_layer()
        target[-3].register_forward_hook(self._save_feat)   # 块内第二个 BN
        target[-3].register_full_backward_hook(self._save_grad)
        for m in model.modules():   # 关闭 inplace，避免 hook 冲突
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

    def _save_feat(self, m, i, o):
        self.feats = o.detach()

    def _save_grad(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, img_t, class_idx=None):
        x = img_t.unsqueeze(0).to(DEVICE)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)        # 通道权重
        cam = F.relu((w * self.feats).sum(1))[0].cpu()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(IMG, IMG),
                            mode='bilinear', align_corners=False)[0, 0]
        return cam.numpy(), class_idx


def load_img(path):
    img = Image.open(path).convert('L').resize((IMG, IMG))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0), img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = BaseCNN(use_cbam=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    cam_fn = GradCAM(model)

    items = list_images('test')
    rng = np.random.RandomState(42)
    picks = {c: [] for c in range(6)}
    for p, y in items:
        picks[y].append(p)
    report = []
    for c in range(6):
        chosen = rng.choice(picks[c], size=2, replace=False)
        for p in chosen:
            x, orig = load_img(p)
            cam, pred = cam_fn(x, class_idx=c)
            fname = os.path.join(OUT_DIR, f'{CLASSES[c]}_{os.path.basename(p)}.npz')
            np.savez_compressed(fname, gray=np.asarray(orig), cam=cam,
                                pred=pred, true=c)
            report.append({'file': os.path.basename(fname),
                           'true': CLASSES[c], 'pred': CLASSES[pred]})
            print(f"{os.path.basename(p):30s} true={CLASSES[c]:15s} pred={CLASSES[pred]:15s} ok={pred==c}")
    print('[done]', OUT_DIR)


if __name__ == '__main__':
    main()
