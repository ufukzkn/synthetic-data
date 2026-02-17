"""
Local Test Script — Gerçek F-18 grafik görsellerini test et.

Kullanım:
  python test_local.py --image chart.png --model noise_unet.pt
  python test_local.py --image-dir scans/ --model noise_unet.pt
  python test_local.py --image chart.png --model noise_unet.pt --no-grid-removal
  python test_local.py --image chart.png --model noise_unet.pt --threshold 0.3

Pipeline:
  1. Görüntüyü yükle (RGB)
  2. Grid çizgilerini kaldır (grid_removal.py)
  3. UNet ile noise mask tahmini
  4. Threshold → binary mask
  5. cv2.inpaint ile temizle
  6. Sonuçları göster / kaydet
"""

import os
import sys
import argparse
import numpy as np
import cv2

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNet
from grid_removal import remove_grid


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════

def load_model(model_path, device=None):
    """Eğitilmiş modeli yükle."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=3, out_channels=3).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', '?')
    print(f'Model loaded: {model_path}')
    print(f'  Epoch: {epoch}, Loss: {loss}')
    print(f'  Device: {device}')

    return model, device


def predict_mask(model, img_rgb, device, img_size=512):
    """
    RGB görüntüden noise mask tahmini yap.

    Args:
        model: UNet modeli
        img_rgb: RGB görüntü (uint8, herhangi boyut)
        device: torch device
        img_size: Model input boyutu

    Returns:
        pred_mask: [H, W, 3] float32 [0,1] — orijinal boyutta
    """
    H_orig, W_orig = img_rgb.shape[:2]

    # Resize to model input size
    resized = cv2.resize(img_rgb, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)

    # To tensor
    inp = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    inp = inp.to(device)

    # Predict (model returns logits → apply sigmoid for probabilities)
    with torch.no_grad():
        pred = torch.sigmoid(model(inp)).squeeze(0).cpu().numpy()  # [3, img_size, img_size]

    # CHW → HWC
    pred = pred.transpose(1, 2, 0)  # [img_size, img_size, 3]

    # Resize back to original
    pred = cv2.resize(pred, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    return pred


def apply_inpaint(img_rgb, mask_binary, inpaint_radius=3):
    """
    Binary mask ile inpaint uygula.

    Args:
        img_rgb: RGB görüntü (uint8)
        mask_binary: Binary mask (uint8, 0/255) — birleşik 3 kanal

    Returns:
        inpainted: RGB görüntü (uint8)
    """
    # RGB → BGR for cv2
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, mask_binary, inpaint_radius,
                            cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ════════════════════════════════════════════════════════════════

def process_image(img_path, model, device,
                  threshold=0.5, img_size=512,
                  do_grid_removal=True,
                  inpaint_radius=3):
    """
    Tek bir görüntü için tam pipeline.

    Returns:
        dict: {
            'original': RGB,
            'after_grid': RGB,
            'pred_mask_raw': [H,W,3] float,
            'pred_mask_bin': [H,W] uint8,
            'result': RGB (inpainted),
            'channels': {'arrows': mask, 'dashed': mask, 'text': mask},
        }
    """
    # Load image (supports Turkish/unicode filenames)
    data = np.fromfile(img_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f'Image not found or unreadable: {img_path}')

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f'\nProcessing: {img_path}')
    print(f'  Size: {img_rgb.shape[1]}x{img_rgb.shape[0]}')

    # Step 1: Grid removal (optional)
    if do_grid_removal:
        after_grid, grid_mask = remove_grid(img_rgb)
        print(f'  Grid removal: {grid_mask.sum() // 255} pixels removed')
    else:
        after_grid = img_rgb.copy()
        print(f'  Grid removal: skipped')

    # Step 2: UNet prediction
    pred_raw = predict_mask(model, after_grid, device, img_size)
    print(f'  Prediction range: [{pred_raw.min():.3f}, {pred_raw.max():.3f}]')

    # Step 3: Threshold
    arrows_mask = (pred_raw[:, :, 0] > threshold).astype(np.uint8) * 255
    dashed_mask = (pred_raw[:, :, 1] > threshold).astype(np.uint8) * 255
    text_mask = (pred_raw[:, :, 2] > threshold).astype(np.uint8) * 255

    # Combined mask for inpainting
    combined_mask = np.maximum(np.maximum(arrows_mask, dashed_mask), text_mask)

    n_arrow = arrows_mask.sum() // 255
    n_dashed = dashed_mask.sum() // 255
    n_text = text_mask.sum() // 255
    print(f'  Detected pixels — arrows: {n_arrow}, dashed: {n_dashed}, text: {n_text}')

    # Step 4: Inpaint
    result = apply_inpaint(after_grid, combined_mask, inpaint_radius)
    print(f'  Inpaint complete')

    return {
        'original': img_rgb,
        'after_grid': after_grid,
        'pred_mask_raw': pred_raw,
        'pred_mask_bin': combined_mask,
        'result': result,
        'channels': {
            'arrows': arrows_mask,
            'dashed': dashed_mask,
            'text': text_mask,
        },
    }


# ════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════

def show_results(output, title='', save_path=None):
    """Pipeline sonuçlarını görselleştir."""
    import matplotlib
    matplotlib.use('TkAgg')  # Local display
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Pipeline
    axes[0, 0].imshow(output['original'])
    axes[0, 0].set_title('1. Original')

    axes[0, 1].imshow(output['after_grid'])
    axes[0, 1].set_title('2. After Grid Removal')

    axes[0, 2].imshow(output['result'])
    axes[0, 2].set_title('6. Final Result (Inpainted)')

    # Row 2: Masks
    # RGB mask visualization
    mask_rgb = np.stack([
        output['channels']['arrows'],
        output['channels']['dashed'],
        output['channels']['text'],
    ], axis=-1)

    axes[1, 0].imshow(output['pred_mask_raw'])
    axes[1, 0].set_title('3. Predicted Mask (raw)')

    axes[1, 1].imshow(mask_rgb)
    axes[1, 1].set_title('4. Thresholded (R=arrow G=dash B=text)')

    axes[1, 2].imshow(output['pred_mask_bin'], cmap='gray')
    axes[1, 2].set_title('5. Combined Mask (for inpaint)')

    for ax in axes.flat:
        ax.axis('off')

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')

    plt.show()


def save_results(output, out_dir, basename):
    """Sonuçları ayrı dosyalar olarak kaydet."""
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, f'{basename}_original.png'),
                cv2.cvtColor(output['original'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{basename}_after_grid.png'),
                cv2.cvtColor(output['after_grid'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{basename}_result.png'),
                cv2.cvtColor(output['result'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f'{basename}_mask_arrows.png'),
                output['channels']['arrows'])
    cv2.imwrite(os.path.join(out_dir, f'{basename}_mask_dashed.png'),
                output['channels']['dashed'])
    cv2.imwrite(os.path.join(out_dir, f'{basename}_mask_text.png'),
                output['channels']['text'])
    cv2.imwrite(os.path.join(out_dir, f'{basename}_mask_combined.png'),
                output['pred_mask_bin'])

    print(f'  Files saved to: {out_dir}')


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Test UNet noise removal on real F-18 chart images')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory with images to process')
    parser.add_argument('--model', type=str, default='noise_unet.pt',
                        help='Trained model path')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Mask threshold (0-1)')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Model input size')
    parser.add_argument('--no-grid-removal', action='store_true',
                        help='Skip grid removal step')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Save results to this directory')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show matplotlib window')
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error('--image veya --image-dir belirtmelisiniz')

    # Load model
    model, device = load_model(args.model)

    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    if args.image_dir:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        for f in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                images.append(os.path.join(args.image_dir, f))
        print(f'\nFound {len(images)} images in {args.image_dir}')

    # Process each
    for img_path in images:
        try:
            output = process_image(
                img_path, model, device,
                threshold=args.threshold,
                img_size=args.img_size,
                do_grid_removal=not args.no_grid_removal,
            )

            basename = os.path.splitext(os.path.basename(img_path))[0]

            if args.save_dir:
                save_results(output, args.save_dir, basename)

            if not args.no_show:
                save_path = None
                if args.save_dir:
                    save_path = os.path.join(args.save_dir,
                                             f'{basename}_comparison.png')
                show_results(output, title=basename, save_path=save_path)

        except Exception as e:
            print(f'  ERROR: {e}')

    print('\nDone!')


if __name__ == '__main__':
    main()
