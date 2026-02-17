"""
UNet-64 — 3-channel noise segmentation model.

Input:  RGB image  [B, 3, H, W]  float [0,1]
Output: Noise mask [B, 3, H, W]  float [0,1]
        R = arrows, G = dashed lines, B = text

Architecture: Encoder [64,128,256,512] → Bottleneck 1024 → Decoder with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
# Building Blocks
# ════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """ConvTranspose2d (upsample) → concat skip → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if sizes don't match (edge cases with odd dimensions)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2,
                          dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ════════════════════════════════════════════════════════════════
# UNet-64
# ════════════════════════════════════════════════════════════════

class UNet(nn.Module):
    """
    UNet with base features = 64.
    Levels: 64 → 128 → 256 → 512 → 1024 (bottleneck) → decode back.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 64)       # → 64
        self.down1 = Down(64, 128)                    # → 128
        self.down2 = Down(128, 256)                   # → 256
        self.down3 = Down(256, 512)                   # → 512
        self.down4 = Down(512, 1024)                  # → 1024 (bottleneck)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Final 1×1 conv (returns LOGITS — no sigmoid here)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # [B, 64, H, W]
        x2 = self.down1(x1)     # [B, 128, H/2, W/2]
        x3 = self.down2(x2)     # [B, 256, H/4, W/4]
        x4 = self.down3(x3)     # [B, 512, H/8, W/8]
        x5 = self.down4(x4)     # [B, 1024, H/16, W/16]

        # Decoder
        x = self.up1(x5, x4)    # [B, 512, H/8, W/8]
        x = self.up2(x, x3)     # [B, 256, H/4, W/4]
        x = self.up3(x, x2)     # [B, 128, H/2, W/2]
        x = self.up4(x, x1)     # [B, 64, H, W]

        return self.outc(x)  # raw logits — apply sigmoid at inference


# ════════════════════════════════════════════════════════════════
# Loss Functions
# ════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Soft Dice Loss, computed per-channel then averaged.
    Expects LOGITS (applies sigmoid internally)."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred = logits [B, C, H, W] → apply sigmoid for probabilities
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)     # [B,C,N]
        target_flat = target.view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)        # [B,C]
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)      # [B,C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for class-imbalanced pixel segmentation.
    Helps the model focus on hard-to-classify pixels.
    Expects LOGITS (uses binary_cross_entropy_with_logits for AMP safety).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred = logits — safe with AMP autocast
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        prob = torch.sigmoid(pred)
        pt = torch.where(target >= 0.5, prob, 1.0 - prob)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BCEDiceLoss(nn.Module):
    """
    Combined loss: BCE + Dice (+ optional Focal).
    Default weights: 0.5 BCE + 0.5 Dice
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, focal_weight=0.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.focal_w = focal_weight

        self.dice = DiceLoss()
        if focal_weight > 0:
            self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        # pred = logits — all sub-losses handle sigmoid internally
        loss = 0.0
        if self.bce_w > 0:
            loss += self.bce_w * F.binary_cross_entropy_with_logits(pred, target)
        if self.dice_w > 0:
            loss += self.dice_w * self.dice(pred, target)
        if self.focal_w > 0:
            loss += self.focal_w * self.focal(pred, target)
        return loss


# ════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════

def count_parameters(model):
    """Toplam eğitilebilir parametre sayısı."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """Model hakkında kısa bilgi yazdır."""
    total = count_parameters(model)
    print(f'UNet Parameters: {total:,}')
    print(f'  ≈ {total / 1e6:.1f}M parameters')

    # Dummy forward to check output shape
    with torch.no_grad():
        dummy = torch.randn(1, 3, 512, 512)
        logits = model(dummy)
        out = torch.sigmoid(logits)
        print(f'  Input:  {dummy.shape}')
        print(f'  Output: {logits.shape}')
        print(f'  Logit range:  [{logits.min():.3f}, {logits.max():.3f}]')
        print(f'  Prob range:   [{out.min():.3f}, {out.max():.3f}]')


# ════════════════════════════════════════════════════════════════
# TEST
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=3)
    model_summary(model)

    # Quick loss test (pred = logits, not probabilities)
    pred = torch.randn(2, 3, 128, 128)  # randn for logits (can be negative)
    target = (torch.rand(2, 3, 128, 128) > 0.8).float()

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    loss = criterion(pred, target)
    print(f'\n  BCE+Dice loss: {loss.item():.4f}')

    criterion_focal = BCEDiceLoss(bce_weight=0.4, dice_weight=0.4, focal_weight=0.2)
    loss_f = criterion_focal(pred, target)
    print(f'  BCE+Dice+Focal loss: {loss_f.item():.4f}')
