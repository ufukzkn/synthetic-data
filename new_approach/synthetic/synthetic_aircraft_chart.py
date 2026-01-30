# synthetic_aircraft_chart.py
"""
Generate synthetic aircraft performance charts that closely match real charts.
Key features:
- Specific range curve shapes (rise, peak, fall)
- Proper grid with major/minor lines
- Arrow annotations pointing to curves
- Text labels (altitude values, annotations)
- Dashed envelope curves (OPTIMUM CRUISE, MAXIMUM ENDURANCE)
- Realistic scan artifacts
"""

import io
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageChops, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Tuple, List, Optional
import cv2


def generate_specific_range_curve(
    x: np.ndarray,
    base_level: float,
    peak_x: float = 0.35,
    peak_height: float = 0.15,
    decay_rate: float = 0.8
) -> np.ndarray:
    """
    Generate a curve shape typical of aircraft specific range charts.
    
    Shape: starts low, rises to peak around Mach 0.35-0.45, then gradually falls.
    """
    # Normalize x to 0-1 range if not already
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    # Rising portion (exponential approach to peak)
    rise = 1 - np.exp(-8 * x_norm / peak_x)
    
    # Falling portion (gradual decay after peak)
    fall_start = np.maximum(0, x_norm - peak_x)
    fall = np.exp(-decay_rate * fall_start * 3)
    
    # Combine
    shape = rise * fall
    
    # Scale and shift
    y = base_level + peak_height * shape
    
    # Add small noise for realism
    noise = 0.003 * np.sin(20 * np.pi * x_norm) + 0.002 * np.random.randn(len(x))
    y += noise
    
    return np.clip(y, 0.01, 0.10)


def generate_curve_family(
    n_curves: int = 15,
    n_points: int = 300
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a family of specific range curves at different altitudes.
    Higher altitudes = higher curves (better specific range).
    """
    # X range: Mach 0.15 to 0.95
    x = np.linspace(0.15, 0.95, n_points)
    
    curves = []
    
    # Base levels spaced to avoid overlap
    base_levels = np.linspace(0.015, 0.045, n_curves)
    
    # Peak heights decrease for lower curves (higher altitude = better range)
    peak_heights = np.linspace(0.050, 0.025, n_curves)
    
    # Peak x shifts slightly
    peak_positions = np.linspace(0.38, 0.42, n_curves) + np.random.uniform(-0.02, 0.02, n_curves)
    
    for i in range(n_curves):
        y = generate_specific_range_curve(
            x,
            base_level=base_levels[i],
            peak_x=peak_positions[i],
            peak_height=peak_heights[i],
            decay_rate=random.uniform(0.6, 1.0)
        )
        curves.append((x.copy(), y))
    
    return curves


def fig_to_array(fig, dpi: int = 150) -> np.ndarray:
    """Convert matplotlib figure to numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def draw_chart_with_matplotlib(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    W: int = 900,
    H: int = 700,
    draw_grid: bool = True,
    draw_arrows: bool = True,
    draw_text: bool = True,
    draw_envelope: bool = True,
    curve_style: str = 'solid'  # 'solid', 'dashed', or 'mixed'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw chart and return (full_image, curve_mask).
    """
    fig_w, fig_h = W / 100, H / 100
    
    # X and Y axis ranges
    x_min, x_max = 0.10, 1.00
    y_min, y_max = 0.01, 0.10
    
    # ========== FULL IMAGE (with all elements) ==========
    fig1 = plt.figure(figsize=(fig_w, fig_h))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Grid
    if draw_grid:
        ax1.set_xticks(np.arange(x_min, x_max + 0.001, 0.10))
        ax1.set_xticks(np.arange(x_min, x_max + 0.001, 0.05), minor=True)
        ax1.set_yticks(np.arange(y_min, y_max + 0.001, 0.01))
        ax1.set_yticks(np.arange(y_min, y_max + 0.001, 0.005), minor=True)
        
        ax1.grid(True, which='major', linewidth=0.8, alpha=0.5, color='black')
        ax1.grid(True, which='minor', linewidth=0.4, alpha=0.3, color='black')
    
    # Axis labels
    ax1.set_xlabel('MACH NUMBER', fontsize=10, fontweight='bold')
    ax1.set_ylabel('SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL', fontsize=8)
    
    # Draw curves
    curve_lw = random.uniform(1.0, 1.8)
    
    for i, (cx, cy) in enumerate(curves):
        if curve_style == 'dashed':
            linestyle = '--'
        elif curve_style == 'mixed' and random.random() < 0.3:
            linestyle = '--'
        else:
            linestyle = '-'
        
        ax1.plot(cx, cy, color='black', linewidth=curve_lw, linestyle=linestyle)
    
    # Envelope curves (OPTIMUM CRUISE, MAXIMUM ENDURANCE)
    if draw_envelope and curves:
        # Optimum cruise - connects peaks
        peaks_x = []
        peaks_y = []
        for cx, cy in curves:
            peak_idx = np.argmax(cy)
            peaks_x.append(cx[peak_idx])
            peaks_y.append(cy[peak_idx])
        
        if len(peaks_x) > 2:
            # Sort by x
            sorted_idx = np.argsort(peaks_x)
            peaks_x = np.array(peaks_x)[sorted_idx]
            peaks_y = np.array(peaks_y)[sorted_idx]
            
            ax1.plot(peaks_x, peaks_y, 'k-', linewidth=1.5)
            ax1.text(peaks_x[0] - 0.02, peaks_y[0] + 0.003, 
                    'OPTIMUM CRUISE', fontsize=8, ha='right')
    
    # Arrows pointing to curves
    if draw_arrows:
        n_arrows = random.randint(15, 30)
        
        # Altitude labels on right side
        altitudes = ['4500', '5000', '5500', '6000', '6500', '7000', '7500', 
                    '8000', '8500', '9000', '9500', '10,000', '11,000', 
                    '12,000', '13,000', '14,000', '15,000', '17,000']
        
        for i, (cx, cy) in enumerate(curves[:len(altitudes)]):
            # Arrow from label to curve end
            end_x = cx[-1]
            end_y = cy[-1]
            
            if i < len(altitudes):
                # Draw arrow
                ax1.annotate(
                    '',
                    xy=(end_x - 0.02, end_y),
                    xytext=(end_x + 0.03, end_y + random.uniform(-0.003, 0.003)),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='black')
                )
                
                # Draw label
                ax1.text(
                    end_x + 0.04, end_y,
                    altitudes[i],
                    fontsize=8, va='center'
                )
    
    # Text boxes
    if draw_text:
        # Title box
        ax1.text(
            0.75, 0.095,
            'TOTAL FUEL FLOW —\nPOUNDS PER HOUR',
            fontsize=9,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black'),
            va='top'
        )
        
        # Legend box
        ax1.text(
            0.25, 0.025,
            '◀ CRUISE    DASH ▶\n    AOA         AOA\n(USED FOR INTERFERENCE\nDRAG DETERMINATION)',
            fontsize=7,
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black'),
            va='bottom', ha='left'
        )
        
        # Random numbers near curves
        for _ in range(random.randint(5, 12)):
            tx = random.uniform(0.20, 0.70)
            ty = random.uniform(0.02, 0.06)
            val = random.choice(['0.00', '25.00', '50.00', '75.00', '100.00', 
                                '125.00', '150.00', '200.00', '250.00', '300.00'])
            ax1.text(tx, ty, val, fontsize=7, alpha=0.9)
    
    full_img = fig_to_array(fig1, dpi=150)
    
    # Resize to exact dimensions
    full_img = cv2.resize(full_img, (W, H))
    
    # ========== MASK (curves only) ==========
    fig2 = plt.figure(figsize=(fig_w, fig_h))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.axis('off')
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    
    for cx, cy in curves:
        ax2.plot(cx, cy, color='white', linewidth=curve_lw + 0.5)
    
    mask_img = fig_to_array(fig2, dpi=150)
    mask_img = cv2.resize(mask_img, (W, H))
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 30, 1, cv2.THRESH_BINARY)
    
    return full_img, mask_bin


def add_scan_artifacts(
    img: np.ndarray,
    rotation_range: Tuple[float, float] = (-1.5, 1.5),
    noise_sigma: float = 0.02,
    jpeg_quality: int = 50,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.75, 1.25)
) -> np.ndarray:
    """Apply realistic scan/copy artifacts."""
    
    pil_img = Image.fromarray(img)
    W, H = pil_img.size
    
    # Rotation
    angle = random.uniform(*rotation_range)
    pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    
    # Trim and resize back
    pil_img = pil_img.resize((W, H), Image.BILINEAR)
    
    # Brightness/Contrast
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(*brightness_range))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(*contrast_range))
    
    # Noise
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr += np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)
    
    # Slight blur
    if random.random() < 0.5:
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        arr = np.array(pil_img).astype(np.float32) / 255.0
    
    # JPEG compression
    if random.random() < 0.6:
        buf = io.BytesIO()
        Image.fromarray((arr * 255).astype(np.uint8)).save(
            buf, format='JPEG', quality=random.randint(35, jpeg_quality)
        )
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        arr = np.array(pil_img).astype(np.float32) / 255.0
    
    return (arr * 255).astype(np.uint8)


def make_aircraft_chart_sample(
    W: int = 900,
    H: int = 700,
    n_curves: Optional[int] = None,
    add_artifacts: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic aircraft chart sample.
    
    Returns:
        full_img: RGB numpy array (H, W, 3)
        mask: Binary numpy array (H, W) with 0/1
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if n_curves is None:
        n_curves = random.randint(10, 18)
    
    # Generate curves
    curves = generate_curve_family(n_curves=n_curves)
    
    # Draw chart
    full_img, mask = draw_chart_with_matplotlib(
        curves, W, H,
        draw_grid=True,
        draw_arrows=random.random() < 0.8,
        draw_text=random.random() < 0.85,
        draw_envelope=random.random() < 0.6,
        curve_style=random.choice(['solid', 'solid', 'mixed'])
    )
    
    # Add artifacts
    if add_artifacts:
        full_img = add_scan_artifacts(full_img)
    
    return full_img, mask


def generate_dataset(
    output_dir: str,
    n_samples: int = 100,
    W: int = 512,
    H: int = 512
):
    """Generate a dataset of synthetic charts."""
    from pathlib import Path
    import json
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'images').mkdir(exist_ok=True)
    (Path(output_dir) / 'masks').mkdir(exist_ok=True)
    
    for i in range(n_samples):
        img, mask = make_aircraft_chart_sample(W, H, seed=i)
        
        # Save
        cv2.imwrite(f"{output_dir}/images/{i:05d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/masks/{i:05d}.png", mask * 255)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_samples}")
    
    # Save metadata
    meta = {'n_samples': n_samples, 'W': W, 'H': H}
    with open(f"{output_dir}/meta.json", 'w') as f:
        json.dump(meta, f)
    
    print(f"✅ Dataset saved to {output_dir}/")


if __name__ == "__main__":
    # Quick test
    img, mask = make_aircraft_chart_sample(seed=42)
    
    cv2.imwrite("test_synth_chart.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("test_synth_mask.png", mask * 255)
    
    print("✅ Saved test_synth_chart.png and test_synth_mask.png")
    
    # Show
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title("Synthetic Chart")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Curve Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_synth_comparison.png", dpi=150)
    plt.show()
