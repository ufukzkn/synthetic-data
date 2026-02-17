# synthetic_v5_custom.py
"""
Özel grafik üretici - Arkadaşa gösterim için.
Üretilen her grafik için 3 versiyon çıkarır:
1. Orijinal (gürültülü/taranmış görünümlü)
2. Mask (siyah-beyaz eğriler)
3. Colored (renkli eğriler)

Kullanım:
    python synthetic_v5_custom.py --count 20 --output ./output_samples
"""

import io
import os
import math
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from typing import Tuple, List


@dataclass 
class ChartConfig:
    """Configuration for chart generation."""
    x_min: float = 0.30
    x_max: float = 1.00
    y_min: float = 0.04
    y_max: float = 0.15
    n_curves: int = 8
    curve_type: str = 'peaked'
    curve_lw: float = 0.6
    add_grid: bool = True
    add_arrows: bool = True
    add_envelope_optimum: bool = True
    add_envelope_endurance: bool = False
    add_vmax_line: bool = False
    add_text_boxes: bool = True
    add_fuel_labels: bool = True
    add_drag_labels: bool = True


def generate_curve_shape(x: np.ndarray, curve_type: str, curve_index: int, total_curves: int) -> np.ndarray:
    """Generate different curve shapes."""
    alt = curve_index / max(total_curves - 1, 1)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    if curve_type == 'peaked':
        peak_pos = 0.30 + random.uniform(-0.05, 0.05)
        start_y = 0.12 + random.uniform(-0.02, 0.02)
        peak_y = 0.45 + alt * 0.40 + random.uniform(-0.03, 0.03)
        end_y = 0.20 + alt * 0.25 + random.uniform(-0.02, 0.02)
        
        y = np.zeros_like(x_norm)
        for i, t in enumerate(x_norm):
            if t <= peak_pos:
                progress = t / peak_pos
                y[i] = start_y + (peak_y - start_y) * (1 - (1 - progress) ** 2)
            else:
                progress = (t - peak_pos) / (1 - peak_pos)
                y[i] = peak_y - (peak_y - end_y) * (progress ** 0.7)
                
    elif curve_type == 'peaked_oval':
        peak_pos = 0.45 + random.uniform(-0.07, 0.07)
        start_y = 0.12 + random.uniform(-0.02, 0.02)
        peak_y = 0.45 + alt * 0.38 + random.uniform(-0.03, 0.03)
        end_y = 0.20 + alt * 0.25 + random.uniform(-0.02, 0.02)
        
        y = np.zeros_like(x_norm)
        for i, t in enumerate(x_norm):
            if t <= peak_pos:
                progress = t / peak_pos
                y[i] = start_y + (peak_y - start_y) * (math.sin(progress * math.pi / 2) ** 1.2)
            else:
                progress = (t - peak_pos) / (1 - peak_pos)
                y[i] = end_y + (peak_y - end_y) * (math.cos(progress * math.pi / 2) ** 1.2)
                
    elif curve_type == 'wavy':
        freq = random.choice([1.0, 1.5, 2.0])
        phase = random.uniform(0, 1)
        wave = 0.5 + 0.25 * np.sin(2 * np.pi * (x_norm * freq + phase))
        wave += 0.12 * np.sin(4 * np.pi * (x_norm * freq + phase))
        hump_center = random.uniform(0.55, 0.70)
        hump = 0.12 * np.exp(-((x_norm - hump_center) / 0.22) ** 2)
        y = wave + hump
        y = np.clip(y, 0.05, 0.95)
        
    elif curve_type == 'rising':
        start_y = 0.08 + alt * 0.05 + random.uniform(-0.02, 0.02)
        end_y = 0.55 + alt * 0.30 + random.uniform(-0.03, 0.03)
        curvature = random.uniform(0.7, 1.3)
        y = start_y + (end_y - start_y) * (x_norm ** curvature)
        
    elif curve_type == 'falling':
        start_y = 0.65 + alt * 0.25 + random.uniform(-0.03, 0.03)
        end_y = 0.12 + alt * 0.10 + random.uniform(-0.02, 0.02)
        curvature = random.uniform(0.5, 1.0)
        y = start_y - (start_y - end_y) * (x_norm ** curvature)
        
    else:  # mixed
        return generate_curve_shape(x, random.choice(['peaked', 'peaked_oval', 'rising', 'falling', 'wavy']),
                                   curve_index, total_curves)
    
    return y


def fig_to_array(fig, dpi=150, tight=True) -> np.ndarray:
    """Convert matplotlib figure to numpy array."""
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white', edgecolor='none')
    else:
        fig.savefig(buf, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def draw_chart_matplotlib(config: ChartConfig, W: int = 800, H: int = 600) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Draw chart using matplotlib.
    Returns: full_img, mask, curves_data
    """
    fig_w, fig_h = W / 100, H / 100
    
    # Generate curves data first
    x = np.linspace(config.x_min + 0.02, config.x_max - 0.02, 400)
    curves_data = []
    
    for i in range(config.n_curves):
        y_norm = generate_curve_shape(x, config.curve_type, i, config.n_curves)
        y = config.y_min + y_norm * (config.y_max - config.y_min)
        y = np.clip(y, config.y_min + 0.001, config.y_max - 0.001)
        curves_data.append((x.copy(), y))
    
    # ========== FULL IMAGE ==========
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
    ax1.set_xlim(config.x_min, config.x_max)
    ax1.set_ylim(config.y_min, config.y_max)
    
    # Grid
    if config.add_grid:
        x_range = config.x_max - config.x_min
        y_range = config.y_max - config.y_min
        
        x_major = 0.1 if x_range > 0.5 else 0.05
        y_major = 0.01 if y_range < 0.08 else 0.02
        
        ax1.set_xticks(np.arange(config.x_min, config.x_max + 0.001, x_major))
        ax1.set_xticks(np.arange(config.x_min, config.x_max + 0.001, x_major/2), minor=True)
        ax1.set_yticks(np.arange(config.y_min, config.y_max + 0.001, y_major))
        ax1.set_yticks(np.arange(config.y_min, config.y_max + 0.001, y_major/2), minor=True)
        
        ax1.grid(True, which='major', linewidth=0.8, alpha=0.5, color='black')
        ax1.grid(True, which='minor', linewidth=0.4, alpha=0.3, color='black')
    
    # Axes
    ax1.axhline(y=config.y_min, color='black', linewidth=2.0, zorder=10)
    ax1.axvline(x=config.x_min, color='black', linewidth=2.0, zorder=10)
    
    ax1.tick_params(axis='both', which='major', length=6, width=1.5, direction='in')
    ax1.tick_params(axis='both', which='minor', length=3, width=1.0, direction='in')
    
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    ax1.set_xlabel('MACH NUMBER', fontsize=10, fontweight='bold')
    ax1.set_ylabel('SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL', fontsize=8)
    
    # Draw curves
    for cx, cy in curves_data:
        ax1.plot(cx, cy, 'k-', linewidth=config.curve_lw)
    
    # OPTIMUM CRUISE envelope
    if config.add_envelope_optimum:
        if config.curve_type in ['peaked', 'peaked_oval']:
            envelope_pts = [(cx[np.argmax(cy)], cy.max()) for cx, cy in curves_data]
        else:
            envelope_pts = [(cx[int(len(cx)*0.5)], cy[int(len(cy)*0.5)]) for cx, cy in curves_data]
        
        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2)
        
        ax1.text(ex[0] - 0.03, ey[-1] + (config.y_max - config.y_min) * 0.02,
                'OPTIMUM\nCRUISE', fontsize=8, ha='right', va='bottom')
    
    # Arrows
    if config.add_arrows:
        fuel_flows = ['3000', '3500', '4000', '4500', '5000', '5500', 
                     '6000', '6500', '7000', '7500', '8000', '8500']
        
        for idx, (cx, cy) in enumerate(reversed(curves_data)):
            if idx >= len(fuel_flows):
                break
            
            if random.random() < 0.5:
                arrow_idx = -1
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]
                dx = random.uniform(0.04, 0.08)
                dy = random.uniform(-0.005, 0.005)
                x_tail = x_head + dx
                y_tail = y_head + dy
            else:
                mid_start = len(cx) // 4
                mid_end = 3 * len(cx) // 4
                arrow_idx = random.randint(mid_start, mid_end)
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]
                angle = random.uniform(20, 70)
                dist = random.uniform(0.05, 0.10)
                if random.random() < 0.5:
                    dx = dist * math.cos(math.radians(angle))
                    dy = dist * math.sin(math.radians(angle))
                else:
                    dx = dist * math.cos(math.radians(-angle))
                    dy = dist * math.sin(math.radians(-angle))
                x_tail = x_head + dx
                y_tail = y_head + dy
            
            ax1.plot([x_tail, x_head], [y_tail, y_head], color="black", linewidth=0.6)
            
            if random.random() < 0.4:
                arrow_style = random.choice(["-|>", "->"])
                fill_style = "none"
            else:
                arrow_style = random.choice(["-|>", "-|>", "->"])
                fill_style = "black"
            
            ax1.annotate(
                "",
                xy=(x_head, y_head),
                xytext=(x_tail, y_tail),
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    lw=random.uniform(0.7, 1.1),
                    color="black",
                    fc=fill_style,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=random.uniform(12, 18),
                ),
            )
            
            label_x = x_tail + random.uniform(0.06, 0.10)
            ax1.text(label_x, y_tail + random.uniform(-0.002, 0.002),
                    fuel_flows[idx], fontsize=8, va='center', ha='left')
    
    # Text boxes
    if config.add_text_boxes:
        ax1.text(
            config.x_max - 0.05, config.y_max - 0.005,
            'TOTAL FUEL FLOW—\nPOUNDS PER HOUR',
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black')
        )
    
    full_img = fig_to_array(fig1, dpi=150, tight=True)
    full_img = cv2.resize(full_img, (W, H))
    
    # ========== MASK (curves only) ==========
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.set_xlim(config.x_min, config.x_max)
    ax2.set_ylim(config.y_min, config.y_max)
    ax2.set_position([0, 0, 1, 1])
    ax2.axis('off')
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    
    for cx, cy in curves_data:
        ax2.plot(cx, cy, 'w-', linewidth=config.curve_lw + 0.6)
    
    mask_img = fig_to_array(fig2, dpi=150, tight=False)
    mask_img = cv2.resize(mask_img, (W, H))
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)
    
    return full_img, mask, curves_data


def colorize_curves(curves_data: List, config: ChartConfig, W: int, H: int, black_bg: bool = True) -> np.ndarray:
    """Render colored curves."""
    fig_w, fig_h = W / 100, H / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    
    bg_color = 'black' if black_bg else 'white'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    n_curves = len(curves_data)
    for i, (cx, cy) in enumerate(curves_data):
        hue = int(180 * i / max(n_curves, 1))
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
        ax.plot(cx, cy, color=np.array(rgb_color) / 255.0, linewidth=config.curve_lw + 0.5, zorder=2)
    
    colored = fig_to_array(fig, dpi=150, tight=False)
    colored = cv2.resize(colored, (W, H))
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def random_config() -> ChartConfig:
    """Generate random chart configuration with weighted curve types."""
    x_ranges = [
        (0.30, 0.95), (0.30, 1.00), (0.40, 1.10), (0.50, 1.20),
        (0.50, 1.30), (0.50, 1.40), (0.60, 1.40)
    ]
    y_ranges = [
        (0.04, 0.15), (0.05, 0.15), (0.06, 0.17), (0.07, 0.18),
        (0.08, 0.19), (0.08, 0.20), (0.05, 0.14)
    ]
    
    x_min, x_max = random.choice(x_ranges)
    y_min, y_max = random.choice(y_ranges)
    
    # Ağırlıklı curve type seçimi (dataset ile aynı)
    # peaked_oval: 28%, peaked: 26%, rising: 16%, falling: 14%, wavy: 10%, mixed: 6%
    curve_types = ['peaked_oval'] * 28 + ['peaked'] * 26 + ['rising'] * 16 + ['falling'] * 14 + ['wavy'] * 10 + ['mixed'] * 6
    curve_type = random.choice(curve_types)
    
    # Curve type'a göre farklı eğri sayıları
    if curve_type == 'wavy':
        n_curves = random.randint(3, 6)
    elif curve_type in ['falling', 'mixed']:
        n_curves = random.randint(4, 7)
    elif curve_type == 'rising':
        n_curves = random.randint(5, 8)
    else:  # peaked, peaked_oval
        n_curves = random.randint(6, 12)
    
    return ChartConfig(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        n_curves=n_curves,
        curve_type=curve_type,
        curve_lw=random.uniform(0.3, 0.6),
        add_grid=random.random() < 0.95,
        add_arrows=random.random() < 0.85,
        add_envelope_optimum=random.random() < 0.70,
        add_envelope_endurance=random.random() < 0.35,
        add_vmax_line=random.random() < 0.25,
        add_text_boxes=random.random() < 0.75,
        add_fuel_labels=random.random() < 0.80,
        add_drag_labels=random.random() < 0.55,
    )


def add_scan_artifacts(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Add scan/photocopy artifacts."""
    pil_img = Image.fromarray(img)
    
    # Rotation
    angle = random.uniform(-1.2, 1.2) * strength
    pil_img = pil_img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    
    # Brightness/contrast
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.90, 1.10))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.88, 1.12))
    
    # Noise
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.012 * strength, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    
    # JPEG artifacts
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(
        buf, format='JPEG', quality=random.randint(50, 80)
    )
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def generate_samples(count: int, output_dir: str, width: int = 800, height: int = 600):
    """
    Generate multiple samples with all three versions.
    
    Args:
        count: Number of samples to generate
        output_dir: Output directory
        width: Image width
        height: Image height
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'colored'), exist_ok=True)
    
    print(f"🎨 Generating {count} samples to {output_dir}/")
    print(f"   Image size: {width}x{height}")
    print()
    
    type_counts = {}
    
    for i in range(count):
        config = random_config()
        
        # Count curve types
        type_counts[config.curve_type] = type_counts.get(config.curve_type, 0) + 1
        
        # Generate chart
        full_img, mask, curves_data = draw_chart_matplotlib(config, W=width, H=height)
        
        # Add scan artifacts
        full_img_noisy = add_scan_artifacts(full_img)
        
        # Generate colored version
        colored = colorize_curves(curves_data, config, W=width, H=height, black_bg=True)
        
        # Save files
        filename = f"sample_{i+1:03d}"
        
        cv2.imwrite(os.path.join(output_dir, 'original', f'{filename}.png'), 
                   cv2.cvtColor(full_img_noisy, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, 'mask', f'{filename}.png'), mask)
        cv2.imwrite(os.path.join(output_dir, 'colored', f'{filename}.png'), colored)
        
        print(f"  ✓ {filename} ({config.curve_type}, {config.n_curves} curves)")
        
        # Memory cleanup
        plt.close('all')
    
    # Summary
    print()
    print("=" * 50)
    print("📊 ÖZET")
    print("=" * 50)
    print(f"Toplam: {count} grafik")
    print()
    print("Curve Type Dağılımı:")
    for ctype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = cnt / count * 100
        print(f"  {ctype:12s}: {cnt:3d} ({pct:5.1f}%)")
    
    print()
    print(f"📁 Output klasörleri:")
    print(f"  {output_dir}/original/  - Orijinal (gürültülü) grafikler")
    print(f"  {output_dir}/mask/      - Siyah-beyaz maskeler")
    print(f"  {output_dir}/colored/   - Renkli eğriler (siyah arka plan)")
    print()
    print("✅ Tamamlandı!")


def main():
    parser = argparse.ArgumentParser(description='F-18 Chart Synthetic Data Generator')
    parser.add_argument('--count', '-n', type=int, default=20,
                       help='Number of samples to generate (default: 20)')
    parser.add_argument('--output', '-o', type=str, default='./output_samples',
                       help='Output directory (default: ./output_samples)')
    parser.add_argument('--width', '-W', type=int, default=800,
                       help='Image width (default: 800)')
    parser.add_argument('--height', '-H', type=int, default=600,
                       help='Image height (default: 600)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed: {args.seed}")
    
    generate_samples(
        count=args.count,
        output_dir=args.output,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()
