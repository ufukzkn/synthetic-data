# train_curve_segmentation.py
"""
Train a U-Net model for curve segmentation from aircraft charts.
Optimized for RTX 5060 or Google Colab.

Usage:
    python train_curve_segmentation.py --epochs 100 --batch_size 8
    
For Colab, copy this file and run.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# Model: Enhanced U-Net with Attention
# ============================================================

class ConvBlock(nn.Module):
    """Double convolution block with batch norm."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates for better curve segmentation.
    """
    def __init__(self, in_ch=3, out_ch=1, base_filters=32, dropout=0.1):
        super().__init__()
        
        f = base_filters
        
        # Encoder
        self.enc1 = ConvBlock(in_ch, f)
        self.enc2 = ConvBlock(f, f*2)
        self.enc3 = ConvBlock(f*2, f*4)
        self.enc4 = ConvBlock(f*4, f*8, dropout)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(f*8, f*16, dropout)
        
        # Decoder with attention
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.att4 = AttentionGate(f*8, f*8, f*4)
        self.dec4 = ConvBlock(f*16, f*8)
        
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.att3 = AttentionGate(f*4, f*4, f*2)
        self.dec3 = ConvBlock(f*8, f*4)
        
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.att2 = AttentionGate(f*2, f*2, f)
        self.dec2 = ConvBlock(f*4, f*2)
        
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.att1 = AttentionGate(f, f, f//2)
        self.dec1 = ConvBlock(f*2, f)
        
        # Output
        self.out_conv = nn.Conv2d(f, out_ch, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)


# ============================================================
# Loss Functions
# ============================================================

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice + Focal loss."""
    def __init__(self, bce_weight=0.3, dice_weight=0.4, focal_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce = self.bce(pred, target.float())
        dice = dice_loss(pred, target)
        focal = focal_loss(pred, target)
        
        return (self.bce_weight * bce + 
                self.dice_weight * dice + 
                self.focal_weight * focal)


# ============================================================
# Dataset
# ============================================================

class OnTheFlyDataset(Dataset):
    """
    Generate synthetic samples on-the-fly during training.
    More memory efficient than pre-generating.
    """
    def __init__(self, length=10000, img_size=512, augment=True):
        self.length = length
        self.img_size = img_size
        self.augment = augment
        
        # Import here to avoid circular imports
        from synthetic_aircraft_chart import make_aircraft_chart_sample
        self.make_sample = make_aircraft_chart_sample
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Generate sample
        img, mask = self.make_sample(
            W=self.img_size, H=self.img_size,
            add_artifacts=self.augment
        )
        
        # To tensor
        img_tensor = self.img_transform(img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Random augmentations
        if self.augment and np.random.random() < 0.5:
            # Horizontal flip
            img_tensor = torch.flip(img_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[2])
        
        return img_tensor, mask_tensor


class PreGeneratedDataset(Dataset):
    """Dataset from pre-generated images."""
    def __init__(self, data_dir, img_size=512):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        self.images = sorted((self.data_dir / 'images').glob('*.png'))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.data_dir / 'masks' / img_path.name
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor, mask_tensor


# ============================================================
# Training
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate dice score
            preds = torch.sigmoid(outputs) > 0.5
            dice = (2 * (preds * masks).sum()) / (preds.sum() + masks.sum() + 1e-8)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    return total_loss / len(dataloader), total_dice / len(dataloader)


def train(
    epochs: int = 100,
    batch_size: int = 8,
    img_size: int = 512,
    lr: float = 1e-4,
    save_dir: str = "checkpoints",
    data_dir: Optional[str] = None,
    samples_per_epoch: int = 1000
):
    """Main training function."""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Dataset
    if data_dir and Path(data_dir).exists():
        print(f"Loading pre-generated dataset from {data_dir}")
        train_ds = PreGeneratedDataset(data_dir, img_size)
    else:
        print("Generating samples on-the-fly")
        train_ds = OnTheFlyDataset(
            length=samples_per_epoch * epochs,
            img_size=img_size,
            augment=True
        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0,  # Windows compatibility
        pin_memory=True,
        drop_last=True
    )
    
    # Model
    model = AttentionUNet(in_ch=3, out_ch=1, base_filters=32, dropout=0.1)
    model = model.to(DEVICE)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = CombinedLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith("cuda"))
    
    # Training loop
    best_loss = float('inf')
    history = {'train_loss': [], 'val_dice': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )
        history['train_loss'].append(train_loss)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            print("✓ Saved best model")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/final_model.pt")
    
    # Save history
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump(history, f)
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f"{save_dir}/training_curve.png")
    plt.close()
    
    print(f"\n✅ Training complete! Model saved to {save_dir}/")
    
    return model


# ============================================================
# Inference
# ============================================================

def predict(model, image_path: str, output_path: str = "prediction.png"):
    """Run inference on a single image."""
    from PIL import Image
    
    model.eval()
    
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Resize back
    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
    pred_img = pred_img.resize(original_size, Image.BILINEAR)
    pred_img.save(output_path)
    
    print(f"Saved prediction to {output_path}")
    
    return np.array(pred_img)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train curve segmentation model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--samples_per_epoch", type=int, default=500)
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        samples_per_epoch=args.samples_per_epoch
    )
