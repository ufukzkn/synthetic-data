"""Preview — Her grafik türünden 5'er örnek görselleştir.
4 tür (peaked, peaked_oval, rising, falling) × 5 örnek = 20 satır.
Her satır: Input — Clean — Arrow mask — Dashed mask — Text mask — Combined mask
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_noise import make_sample


CURVE_TYPES = ['peaked', 'peaked_oval', 'rising', 'falling']
SAMPLES_PER_TYPE = 5


def preview(out_path=None):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'preview_noise.png')

    n_rows = len(CURVE_TYPES) * SAMPLES_PER_TYPE  # 25
    fig, axes = plt.subplots(n_rows, 6, figsize=(30, 5 * n_rows))

    col_titles = ['Input (grid-free)', 'Clean', 'Arrow mask (R)',
                  'Dashed mask (G)', 'Text mask (B)', 'Combined mask']

    row = 0
    for ctype in CURVE_TYPES:
        for j in range(SAMPLES_PER_TYPE):
            seed = hash(f'{ctype}_{j}') % (2**31)
            inp, mask, clean = make_sample(512, 512, seed=seed,
                                           force_curve_type=ctype)

            axes[row, 0].imshow(inp)
            axes[row, 1].imshow(clean)
            axes[row, 2].imshow(mask[:, :, 0], cmap='Reds', vmin=0, vmax=255)
            axes[row, 3].imshow(mask[:, :, 1], cmap='Greens', vmin=0, vmax=255)
            axes[row, 4].imshow(mask[:, :, 2], cmap='Blues', vmin=0, vmax=255)
            axes[row, 5].imshow(mask)

            label = f'{ctype} #{j+1}'
            axes[row, 0].set_ylabel(label, fontsize=12, fontweight='bold')
            row += 1

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f'Preview saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    preview()
