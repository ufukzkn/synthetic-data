"""Preview new synthetic noise: direction arrows + curved dashed lines."""
import os, sys, random, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_noise import make_sample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 6
fig, axes = plt.subplots(N, 3, figsize=(18, 5 * N))

for i in range(N):
    inp, mask, clean = make_sample(512, 512, seed=None)

    axes[i, 0].imshow(inp)
    axes[i, 0].set_title(f'Input #{i+1}', fontsize=9)

    axes[i, 1].imshow(mask)
    axes[i, 1].set_title('Mask (R=arrow G=dash B=text)', fontsize=8)

    # Overlay: kırmızı=arrow, yeşil=dashed, mavi=text
    overlay = clean.copy()
    r_mask = mask[:, :, 0] > 128
    g_mask = mask[:, :, 1] > 128
    b_mask = mask[:, :, 2] > 128
    overlay[r_mask] = [255, 0, 0]
    overlay[g_mask] = [0, 200, 0]
    overlay[b_mask] = [0, 0, 255]
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title('Overlay (red=arrow green=dash blue=text)', fontsize=7)

    # Stats
    n_a = int(r_mask.sum())
    n_d = int(g_mask.sum())
    n_t = int(b_mask.sum())
    print(f'  #{i+1}: arrows={n_a}, dashed={n_d}, text={n_t}')

for ax in axes.flat:
    ax.axis('off')
plt.suptitle('Synthetic Noise v3 — Direction Arrows + Curved Dashed', fontsize=14)
plt.tight_layout()
plt.savefig('preview_v3.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: preview_v3.png')
