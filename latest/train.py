"""
Training Script — UNet-64 Noise Segmentation

Colab / Kaggle / Local'de çalışır.
  python train.py --epochs 30 --batch 4 --samples 6000

Veya notebook'tan:
  from train import run_training
  history = run_training(epochs=30, batch_size=4)
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# AMP — use new-style API (torch.amp) for PyTorch >= 2.4 compat

# Aynı klasörden import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNet, BCEDiceLoss, count_parameters
from synthetic_noise import get_torch_dataset


# ════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════

def run_training(
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    n_samples: int = 6000,
    img_size: int = 512,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    focal_weight: float = 0.0,
    use_amp: bool = True,
    num_workers: int = 0,
    save_dir: str = '.',
    model_name: str = 'noise_unet.pt',
    log_every: int = 50,
    save_every: int = 5,
    visualize_every: int = 5,
):
    """
    Ana eğitim fonksiyonu.

    Args:
        epochs: Epoch sayısı
        batch_size: Batch boyutu (T4 16GB → 4 veya 8 @ 512px)
        lr: Learning rate
        n_samples: Epoch başına örnek sayısı
        img_size: Görüntü boyutu (kare)
        bce_weight: BCE loss ağırlığı
        dice_weight: Dice loss ağırlığı
        focal_weight: Focal loss ağırlığı (0 = kullanma)
        use_amp: Mixed precision (AMP) kullan
        num_workers: DataLoader worker sayısı (0 = safe default)
        save_dir: Model kayıt dizini
        model_name: Model dosya adı
        log_every: Her N batch'te log yazdır
        save_every: Her N epoch'ta checkpoint kaydet
        visualize_every: Her N epoch'ta görsel örnek göster

    Returns:
        dict: Training history (losses, times, etc.)
    """

    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
        print(f'  VRAM: {vram:.1f} GB')

    # ── Model ──
    model = UNet(in_channels=3, out_channels=2).to(device)
    n_params = count_parameters(model)
    print(f'Model: UNet-64 ({n_params:,} params ≈ {n_params/1e6:.1f}M)')

    # ── Loss ──
    criterion = BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
    ).to(device)
    loss_desc = f'BCE({bce_weight}) + Dice({dice_weight})'
    if focal_weight > 0:
        loss_desc += f' + Focal({focal_weight})'
    print(f'Loss: {loss_desc}')

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f'Optimizer: Adam (lr={lr}, wd=1e-5), CosineAnnealing')

    # ── Dataset ──
    print(f'Dataset: {n_samples} samples, {img_size}×{img_size}, on-the-fly')
    dataset = get_torch_dataset(n_samples=n_samples, W=img_size, H=img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # shuffle her epoch için index'leri karıştırır
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
    )
    n_batches = len(loader)
    print(f'Batches per epoch: {n_batches}')

    # ── AMP ──
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None
    if scaler:
        print('AMP: Enabled (mixed precision)')

    # ── Training history ──
    history = {
        'epoch_loss': [],
        'epoch_bce': [],
        'epoch_dice': [],
        'epoch_time': [],
        'lr': [],
        'best_loss': float('inf'),
        'best_epoch': 0,
    }

    print(f'\n{"="*60}')
    print(f'Training for {epochs} epochs...')
    print(f'{"="*60}\n')

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(loader, 1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    preds = model(inputs)
                    loss = criterion(preds, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % log_every == 0 or batch_idx == n_batches:
                avg = epoch_loss / batch_idx
                print(f'  [{epoch}/{epochs}] batch {batch_idx}/{n_batches}  '
                      f'loss: {loss.item():.4f}  avg: {avg:.4f}', end='\r')

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']

        history['epoch_loss'].append(avg_loss)
        history['epoch_time'].append(epoch_time)
        history['lr'].append(current_lr)

        # Best model tracking
        if avg_loss < best_loss:
            best_loss = avg_loss
            history['best_loss'] = best_loss
            history['best_epoch'] = epoch
            best_path = os.path.join(save_dir, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'history': history,
            }, best_path)

        print(f'  Epoch {epoch:3d}/{epochs}  loss: {avg_loss:.4f}  '
              f'lr: {current_lr:.2e}  time: {epoch_time:.1f}s'
              f'{"  ★ best" if avg_loss <= best_loss else ""}')

        # Periodic checkpoint
        if save_every > 0 and epoch % save_every == 0:
            ckpt_path = os.path.join(save_dir, f'checkpoint_ep{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)

        # Visualize
        if visualize_every > 0 and epoch % visualize_every == 0:
            _visualize_predictions(model, dataset, device, epoch, save_dir)

    print(f'\n{"="*60}')
    print(f'Training complete!')
    print(f'Best loss: {best_loss:.4f} (epoch {history["best_epoch"]})')
    print(f'Model saved: {os.path.join(save_dir, model_name)}')
    print(f'Total time: {sum(history["epoch_time"]):.0f}s '
          f'({sum(history["epoch_time"])/60:.1f}min)')
    print(f'{"="*60}')

    # Plot training curves
    _plot_training_curves(history, save_dir)

    return history


# ════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════

def _visualize_predictions(model, dataset, device, epoch, save_dir):
    """Birkaç örnek için tahmin görseli oluştur."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    with torch.no_grad():
        for row in range(3):
            inp, gt = dataset[row]
            inp_dev = inp.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(inp_dev)).squeeze(0).cpu()

            # Input image
            axes[row, 0].imshow(inp.permute(1, 2, 0).numpy())
            axes[row, 0].set_title('Input' if row == 0 else '')

            # Ground truth mask (2ch → 3ch görselleştirme)
            gt_vis = torch.zeros(3, gt.shape[1], gt.shape[2])
            gt_vis[0] = gt[0]  # arrows → R
            gt_vis[1] = gt[1]  # dashed → G
            axes[row, 1].imshow(gt_vis.permute(1, 2, 0).numpy())
            axes[row, 1].set_title('Ground Truth (R=arrow G=dash)' if row == 0 else '')

            # Predicted mask (raw, 2ch → 3ch)
            pred_vis = torch.zeros(3, pred.shape[1], pred.shape[2])
            pred_vis[0] = pred[0]
            pred_vis[1] = pred[1]
            axes[row, 2].imshow(pred_vis.permute(1, 2, 0).numpy())
            axes[row, 2].set_title('Prediction' if row == 0 else '')

            # Thresholded (binary)
            binary = (pred > 0.5).float()
            bin_vis = torch.zeros(3, binary.shape[1], binary.shape[2])
            bin_vis[0] = binary[0]
            bin_vis[1] = binary[1]
            axes[row, 3].imshow(bin_vis.permute(1, 2, 0).numpy())
            axes[row, 3].set_title('Threshold>0.5' if row == 0 else '')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, f'predictions_ep{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    model.train()


def _plot_training_curves(history, save_dir):
    """Loss ve LR eğrilerini çiz ve kaydet."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['epoch_loss']) + 1)

    ax1.plot(epochs, history['epoch_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    if history['best_epoch'] > 0:
        ax1.axvline(x=history['best_epoch'], color='r', linestyle='--',
                    alpha=0.5, label=f'Best (ep{history["best_epoch"]})')
        ax1.legend()

    ax2.plot(epochs, history['lr'], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved: {path}')


# ════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ════════════════════════════════════════════════════════════════

def evaluate_model(model_path, n_samples=100, img_size=512, threshold=0.5):
    """
    Eğitilmiş modeli değerlendir.

    Returns:
        dict: Per-channel IoU, Dice, Precision, Recall
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNet(in_channels=3, out_channels=2).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = get_torch_dataset(n_samples=n_samples, W=img_size, H=img_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    channel_names = ['Arrow', 'Dashed']
    metrics = {ch: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for ch in channel_names}

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = torch.sigmoid(model(inputs)).cpu()
            preds_bin = (preds > threshold).float()
            targets_bin = (targets > 0.5).float()

            for c, ch_name in enumerate(channel_names):
                p = preds_bin[:, c].flatten()
                t = targets_bin[:, c].flatten()
                metrics[ch_name]['tp'] += (p * t).sum().item()
                metrics[ch_name]['fp'] += (p * (1 - t)).sum().item()
                metrics[ch_name]['fn'] += ((1 - p) * t).sum().item()
                metrics[ch_name]['tn'] += ((1 - p) * (1 - t)).sum().item()

    results = {}
    for ch_name in channel_names:
        m = metrics[ch_name]
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        results[ch_name] = {
            'IoU': iou, 'Dice': dice,
            'Precision': precision, 'Recall': recall,
        }
        print(f'  {ch_name}:  IoU={iou:.4f}  Dice={dice:.4f}  '
              f'P={precision:.4f}  R={recall:.4f}')

    return results


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train UNet-64 noise segmentation')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--samples', type=int, default=6000)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--bce-w', type=float, default=0.5)
    parser.add_argument('--dice-w', type=float, default=0.5)
    parser.add_argument('--focal-w', type=float, default=0.0)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default='.')
    parser.add_argument('--model-name', type=str, default='noise_unet.pt')
    parser.add_argument('--eval', type=str, default=None,
                        help='Evaluate model at this path instead of training')
    args = parser.parse_args()

    if args.eval:
        print(f'Evaluating: {args.eval}')
        evaluate_model(args.eval)
    else:
        run_training(
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            n_samples=args.samples,
            img_size=args.img_size,
            bce_weight=args.bce_w,
            dice_weight=args.dice_w,
            focal_weight=args.focal_w,
            use_amp=not args.no_amp,
            num_workers=args.workers,
            save_dir=args.save_dir,
            model_name=args.model_name,
        )


if __name__ == '__main__':
    main()
