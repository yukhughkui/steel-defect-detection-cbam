# -*- coding: utf-8 -*-
"""把 results/cbam.pt（CBAM-CNN）导出为 ONNX 并验证"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np
import torch
from exp2_cnn import BaseCNN, CLASSES

model = BaseCNN(use_cbam=True)
model.load_state_dict(torch.load(os.path.join('results', 'cbam.pt'), map_location='cpu'))
model.eval()

dummy = torch.rand(1, 1, 128, 128)
torch.onnx.export(model, dummy, os.path.join('docs', 'cbam_cnn.onnx'),
                  opset_version=12, dynamo=False,
                  input_names=['input'], output_names=['logits'])
print('exported')

import onnxruntime as ort
sess = ort.InferenceSession(os.path.join('docs', 'cbam_cnn.onnx'), providers=['CPUExecutionProvider'])
IN = sess.get_inputs()[0].name
print('input:', IN, sess.get_inputs()[0].shape)

x = torch.rand(1, 1, 128, 128)
with torch.no_grad():
    t_logits = model(x)[0].numpy()
o_logits = sess.run(None, {IN: x.numpy()})[0][0]
print('logits 最大偏差:', np.abs(t_logits - o_logits).max())

# 真实测试图上的表现（top1 是否等于标签）
from PIL import Image
ok = 0
import random, glob, json
random.seed(1)
items = []
for ci, cls in enumerate(CLASSES):
    fs = glob.glob(os.path.join('datasets', 'NEU-DET', 'test', 'images', cls + '_*.jpg'))
    for p in random.sample(fs, 2):
        img = Image.open(p).convert('L').resize((128, 128))
        a = np.asarray(img, dtype=np.float32)[None, None] / 255.0
        pred = int(sess.run(None, {IN: a})[0][0].argmax())
        ok += pred == ci
        items.append((os.path.basename(p), CLASSES[ci], CLASSES[pred]))
print(f'12 张真实测试图 top1 正确 {ok}/12')
for it in items[:6]:
    print('  ', it)
