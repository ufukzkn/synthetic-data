"""
Local Test Script — Gerçek F-18 grafik görsellerini test et.

Kullanım:
  python test_local.py --image chart.png --model noise_unet.pt
  python test_local.py --image-dir scans/ --model noise_unet.pt
  python test_local.py --image chart.png --model noise_unet.pt --no-grid-removal
  python test_local.py --image chart.png --model noise_unet.pt --threshold 0.3
  python test_local.py --image chart.png --model noise_unet.pt --no-ocr
  python test_local.py --image chart.png --model noise_unet.pt --no-trace

Pipeline:
  1. Görüntüyü yükle (RGB)
  2. Grid çizgilerini kaldır (grid_removal.py)
  3. UNet ile noise mask tahmini (2 kanal: arrows + dashed)
  4. Arrow tip'lerden geriye leader line trace (postprocess)
  5. Tesseract OCR ile text detection (postprocess)
  6. Tüm mask'ları birleştir
  7. cv2.inpaint ile temizle
  8. Sonuçları göster / kaydet
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
from postprocess import build_inpaint_mask


# ════════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════════

def load_model(model_path, device=None):
    """Eğitilmiş modeli yükle."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=3, out_channels=2).to(device)
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
        pred_mask: [H, W, 2] float32 [0,1] — orijinal boyutta
                   Ch0=arrows, Ch1=dashed
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
        pred = torch.sigmoid(model(inp)).squeeze(0).cpu().numpy()  # [2, img_size, img_size]

    # CHW → HWC
    pred = pred.transpose(1, 2, 0)  # [img_size, img_size, 2]

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
                  inpaint_radius=3,
                  do_trace=True,
                  do_ocr=True,
                  max_trace_len=300,
                  ocr_padding=3,
                  ocr_min_conf=30):
    """
    Tek bir görüntü için tam pipeline.

    Returns:
        dict: {
            'original': RGB,
            'after_grid': RGB,
            'pred_mask_raw': [H,W,2] float,
            'pred_mask_bin': [H,W] uint8,
            'result': RGB (inpainted),
            'channels': {'arrows': mask, 'leaders': mask,
                         'dashed': mask, 'text': mask},
            'ocr_texts': list,
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

    # Step 2: UNet prediction (2 channels: arrows + dashed)
    pred_raw = predict_mask(model, after_grid, device, img_size)
    print(f'  Prediction range: [{pred_raw.min():.3f}, {pred_raw.max():.3f}]')

    # Step 3: Threshold
    arrows_mask = (pred_raw[:, :, 0] > threshold).astype(np.uint8) * 255
    dashed_mask = (pred_raw[:, :, 1] > threshold).astype(np.uint8) * 255

    n_arrow = arrows_mask.sum() // 255
    n_dashed = dashed_mask.sum() // 255
    print(f'  UNet detected — arrows: {n_arrow}, dashed: {n_dashed}')

    # Step 4: Postprocess (arrow trace + OCR)
    pp_result = build_inpaint_mask(
        after_grid, arrows_mask, dashed_mask,
        trace_leaders=do_trace,
        use_ocr=do_ocr,
        max_trace_len=max_trace_len,
        ocr_padding=ocr_padding,
        ocr_min_conf=ocr_min_conf,
    )

    combined_mask = pp_result['combined']
    leader_mask = pp_result['leaders']
    text_mask = pp_result['text']
    ocr_texts = pp_result['ocr_texts']

    n_leader = leader_mask.sum() // 255
    n_text = text_mask.sum() // 255
    print(f'  Postprocess — leaders: {n_leader}, text(OCR): {n_text}')
    if ocr_texts:
        print(f'  OCR texts: {[t["text"] for t in ocr_texts[:10]]}')

    # Step 5: Inpaint
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
            'leaders': leader_mask,
            'dashed': dashed_mask,
            'text': text_mask,
        },
        'ocr_texts': ocr_texts,
    }


# ════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════

def show_results(output, title='', save_path=None):
    """Pipeline sonuçlarını görselleştir."""
    import matplotlib
    matplotlib.use('TkAgg')  # Local display
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # Row 1: Pipeline
    axes[0, 0].imshow(output['original'])
    axes[0, 0].set_title('1. Original')

    axes[0, 1].imshow(output['after_grid'])
    axes[0, 1].set_title('2. After Grid Removal')

    axes[0, 2].imshow(output['pred_mask_bin'], cmap='gray')
    axes[0, 2].set_title('5. Combined Mask')

    axes[0, 3].imshow(output['result'])
    axes[0, 3].set_title('6. Final Result (Inpainted)')

    # Row 2: Individual masks
    # UNet mask (arrows=R, dashed=G, zeroed B)
    ch = output['channels']
    unet_rgb = np.stack([ch['arrows'], ch['dashed'],
                         np.zeros_like(ch['arrows'])], axis=-1)
    axes[1, 0].imshow(unet_rgb)
    axes[1, 0].set_title('3a. UNet (R=arrow G=dash)')

    axes[1, 1].imshow(ch['leaders'], cmap='gray')
    axes[1, 1].set_title('3b. Leader Lines (traced)')

    axes[1, 2].imshow(ch['text'], cmap='gray')
    axes[1, 2].set_title('3c. Text (OCR)')

    # All channels combined color
    all_rgb = np.stack([
        np.maximum(ch['arrows'], ch['leaders']),  # R = arrows + leaders
        ch['dashed'],                              # G = dashed
        ch['text'],                                # B = text
    ], axis=-1)
    axes[1, 3].imshow(all_rgb)
    axes[1, 3].set_title('4. All Channels (R=arrow G=dash B=text)')

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
    cv2.imwrite(os.path.join(out_dir, f'{basename}_mask_leaders.png'),
                output['channels']['leaders'])
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
    parser.add_argument('--no-trace', action='store_true',
                        help='Skip arrow leader line tracing')
    parser.add_argument('--no-ocr', action='store_true',
                        help='Skip Tesseract OCR text detection')
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
                do_trace=not args.no_trace,
                do_ocr=not args.no_ocr,
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
