import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



import os
import torchvision
imgs_dir = []
# for c in ['p', 'f']:
# for c in ['p', 'f', 'c']:
for c in ['c']:
        for name in os.listdir('figs_vae_' + c):
            imgs_dir.append(os.path.join('figs_vae_' + c, name))
            # print(os.path.join('figs_vae_' + c, name))
# print(imgs_dir)
imgs = [torchvision.transforms.ToTensor()(Image.open(k)).permute(1, 2, 0) for k in imgs_dir]

import torch

num_cols = 6
num_rows = len(imgs) // num_cols
scale = 1.5

figsize = (num_cols * scale, num_rows * scale)
_, axes = plt.subplots(num_rows, num_cols, figsize = figsize)

for i, axx in enumerate(axes):
    # if i == 0: axx.set_title('pendulum')
    # if i == 1: axx.set_title('flow')
    for j, ax in enumerate(axx):
        if i % 2 != 0 or j: ax.set_title(f'epoch {(i % 2 * num_cols + j - 1) * 10}')
        else: ax.set_title('origin')
        ax.imshow(imgs[i * num_cols + j])
        # print((i - 1) * num_cols)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

plt.legend()
plt.show()
