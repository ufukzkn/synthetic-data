"""
Grid Removal — Klasik CV ile grid çizgilerini kaldır.
Eğrilere minimum hasar vererek yatay/dikey grid çizgilerini siler.
"""

import cv2
import numpy as np


def remove_grid(img_rgb, h_kernel_len=40, v_kernel_len=40,
                grid_dilate=1, inpaint_radius=2, protect_axes=True,
                debug=False):
    """
    Grafikten grid çizgilerini kaldır.

    Args:
        img_rgb: RGB görüntü (uint8, 0-255)
        h_kernel_len: Yatay grid tespiti için kernel uzunluğu
        v_kernel_len: Dikey grid tespiti için kernel uzunluğu
        grid_dilate: Grid mask'ını genişletme miktarı
        inpaint_radius: cv2.inpaint yarıçapı
        protect_axes: Eksen çizgilerini koru
        debug: True ise ara adım görüntüleri döndür

    Returns:
        clean: Grid kaldırılmış RGB görüntü
        grid_mask: Tespit edilen grid pikselleri (binary, 0/255)
        (debug_dict): debug=True ise ara adımlar
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Binary threshold (grid çizgileri koyu, arka plan beyaz)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ─── Yatay grid çizgileri ───
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # ─── Dikey grid çizgileri ───
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Birleştir
    grid_mask = cv2.bitwise_or(h_lines, v_lines)

    # ─── Eksen koruması ───
    if protect_axes:
        h, w = grid_mask.shape
        left_region = int(w * 0.08)
        bottom_region = int(h * 0.92)

        axes_mask = np.zeros_like(grid_mask)

        # Sol eksen — dikey çizgi
        left_strip = binary[:, :left_region]
        left_axes = cv2.morphologyEx(left_strip, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)),
                                     iterations=1)
        axes_mask[:, :left_region] = left_axes

        # Alt eksen — yatay çizgi
        bottom_strip = binary[bottom_region:, :]
        bottom_axes = cv2.morphologyEx(bottom_strip, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)),
                                       iterations=1)
        axes_mask[bottom_region:, :] = bottom_axes

        # Kalın çizgileri koru
        thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thick_lines = cv2.erode(axes_mask, thick_kernel, iterations=1)
        thick_lines = cv2.dilate(thick_lines, thick_kernel, iterations=2)

        grid_mask = cv2.bitwise_and(grid_mask, cv2.bitwise_not(thick_lines))

    # ─── Grid mask'ını hafif genişlet ───
    if grid_dilate > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=grid_dilate)

    # ─── Inpaint ile grid'i kaldır ───
    clean = cv2.inpaint(img_rgb, grid_mask, inpaint_radius, cv2.INPAINT_TELEA)

    if debug:
        debug_dict = {
            'binary': binary,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'grid_mask_raw': cv2.bitwise_or(h_lines, v_lines),
            'grid_mask_final': grid_mask,
        }
        return clean, grid_mask, debug_dict

    return clean, grid_mask


def remove_grid_simple(img_rgb, kernel_len=40):
    """Basit versiyon — sadece temiz görüntü döndürür."""
    clean, _ = remove_grid(img_rgb, h_kernel_len=kernel_len, v_kernel_len=kernel_len)
    return clean


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys, os

    if len(sys.argv) < 2:
        print("Kullanım: python grid_removal.py <görüntü_yolu>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Dosya bulunamadı: {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clean, grid_mask, debug = remove_grid(img_rgb, debug=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(img_rgb);          axes[0, 0].set_title('Orijinal')
    axes[0, 1].imshow(debug['h_lines'], cmap='gray'); axes[0, 1].set_title('Yatay Grid')
    axes[0, 2].imshow(debug['v_lines'], cmap='gray'); axes[0, 2].set_title('Dikey Grid')
    axes[1, 0].imshow(grid_mask, cmap='gray');         axes[1, 0].set_title('Grid Mask')
    axes[1, 1].imshow(clean);            axes[1, 1].set_title('Temiz', color='green')
    diff = cv2.absdiff(img_rgb, clean)
    axes[1, 2].imshow(diff * 3);         axes[1, 2].set_title('Fark (3x)')
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(img_path), 'grid_removal_result.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Kaydedildi: {out}")
