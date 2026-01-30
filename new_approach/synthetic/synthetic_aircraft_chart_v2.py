# synthetic_aircraft_chart_v2.py
"""
Realistic synthetic aircraft performance chart generator.
Closely matches the actual chart characteristics:
- Smooth curves that converge at low Mach, peak around 0.35-0.40, diverge at high Mach
- Large arrows pointing to curve endpoints
- Dashed envelope lines (OPTIMUM CRUISE, MAXIMUM ENDURANCE)
- Text boxes, legends, altitude labels
- Grid with major/minor lines
"""

import io
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches
from typing import Tuple, List, Optional
import cv2


def generate_realistic_curve(
    x: np.ndarray,
    altitude_index: int,
    total_curves: int,
    convergence_point: float = 0.18,
    peak_mach: float = 0.38,
) -> np.ndarray:
    """
    Generate a realistic specific range curve.
    
    Key characteristics:
    - All curves CONVERGE at low Mach (around 0.15-0.20)
    - Peak around Mach 0.35-0.45
    - Higher altitude = higher peak, earlier peak
    - Lower altitude = lower peak, flatter decline
    - Curves DIVERGE at high Mach
    """
    # Normalize altitude index (0 = lowest/worst, 1 = highest/best)
    alt_factor = altitude_index / (total_curves - 1) if total_curves > 1 else 0.5
    
    # Common starting point (all curves converge here)
    start_y = 0.012 + random.uniform(-0.001, 0.001)
    
    # Peak characteristics based on altitude
    # Higher altitude = higher peak, slightly earlier
    peak_y = 0.035 + alt_factor * 0.025 + random.uniform(-0.002, 0.002)
    peak_x = peak_mach - alt_factor * 0.03 + random.uniform(-0.01, 0.01)
    
    # End point (divergence at high Mach)
    # Higher altitude curves stay higher at high Mach
    end_y = 0.018 + alt_factor * 0.015 + random.uniform(-0.002, 0.002)
    
    # Build the curve shape
    y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        if xi <= convergence_point:
            # Rising from start - all curves similar here
            t = (xi - x[0]) / (convergence_point - x[0] + 1e-8)
            # Smooth rise
            rise_target = start_y + (peak_y - start_y) * 0.3
            y[i] = start_y + (rise_target - start_y) * (t ** 0.8)
            
        elif xi <= peak_x:
            # Rising to peak
            t = (xi - convergence_point) / (peak_x - convergence_point + 1e-8)
            # Smooth curve to peak
            y[i] = start_y + (peak_y - start_y) * 0.3 + \
                   (peak_y - (start_y + (peak_y - start_y) * 0.3)) * (1 - (1 - t) ** 2)
        else:
            # Falling from peak - this is where curves diverge
            t = (xi - peak_x) / (x[-1] - peak_x + 1e-8)
            # Gradual decline with slight curvature
            y[i] = peak_y + (end_y - peak_y) * (t ** 0.7)
    
    # Very subtle smoothing (no visible waviness)
    from scipy.ndimage import gaussian_filter1d
    y = gaussian_filter1d(y, sigma=2)
    
    return y


def draw_arrow_annotation(ax, x_tip, y_tip, label, direction='right'):
    """
    Draw a large arrow pointing to a curve endpoint with label.
    Similar to the actual chart's arrow style.
    """
    if direction == 'right':
        # Arrow coming from right, pointing left to the curve
        dx = 0.06
        dy = random.uniform(-0.003, 0.003)
        text_x = x_tip + dx + 0.02
        ha = 'left'
    else:
        # Arrow from left
        dx = -0.06
        dy = random.uniform(-0.003, 0.003)
        text_x = x_tip + dx - 0.02
        ha = 'right'
    
    # Draw arrow
    ax.annotate(
        '',
        xy=(x_tip, y_tip),  # Arrow tip
        xytext=(x_tip + dx, y_tip + dy),  # Arrow tail
        arrowprops=dict(
            arrowstyle='->',
            lw=1.2,
            color='black',
            mutation_scale=15,
        )
    )
    
    # Draw label
    ax.text(text_x, y_tip + dy, label, fontsize=9, ha=ha, va='center')


def draw_envelope_line(ax, curves_x, curves_y, line_type='optimum'):
    """
    Draw dashed envelope line (OPTIMUM CRUISE or MAXIMUM ENDURANCE).
    """
    if line_type == 'optimum':
        # Connect the peaks of all curves
        peaks_x = []
        peaks_y = []
        for cx, cy in zip(curves_x, curves_y):
            peak_idx = np.argmax(cy)
            peaks_x.append(cx[peak_idx])
            peaks_y.append(cy[peak_idx])
        
        # Sort by y
        sorted_idx = np.argsort(peaks_y)
        peaks_x = np.array(peaks_x)[sorted_idx]
        peaks_y = np.array(peaks_y)[sorted_idx]
        
        # Draw smooth line through peaks
        ax.plot(peaks_x, peaks_y, 'k-', linewidth=1.5, zorder=5)
        
        # Label
        ax.text(peaks_x[0] - 0.03, peaks_y[-1] + 0.003, 
                'OPTIMUM CRUISE', fontsize=9, ha='right', va='bottom')
    
    elif line_type == 'endurance':
        # Maximum endurance - different characteristic
        # Usually a steeper line at lower Mach
        pass  # Will implement if needed


def create_chart_image(
    n_curves: int = 15,
    W: int = 900,
    H: int = 700,
    add_grid: bool = True,
    add_arrows: bool = True,
    add_labels: bool = True,
    add_envelope: bool = True,
    add_legend: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create a realistic aircraft performance chart.
    
    Returns:
        full_image: RGB numpy array with all elements
        mask: Binary mask with only curves (0/1)
        curves_data: List of (x, y) arrays for each curve
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Axis ranges (matching the actual chart)
    x_min, x_max = 0.10, 1.00
    y_min, y_max = 0.01, 0.10
    
    # Generate curve data
    x = np.linspace(0.15, 0.92, 400)  # Smooth curves with many points
    curves_data = []
    
    for i in range(n_curves):
        y = generate_realistic_curve(x, i, n_curves)
        curves_data.append((x.copy(), y))
    
    # ========== FULL IMAGE ==========
    fig1, ax1 = plt.subplots(figsize=(W/100, H/100))
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    # Grid
    if add_grid:
        ax1.set_xticks(np.arange(0.1, 1.01, 0.1))
        ax1.set_xticks(np.arange(0.1, 1.01, 0.05), minor=True)
        ax1.set_yticks(np.arange(0.01, 0.101, 0.01))
        ax1.set_yticks(np.arange(0.01, 0.101, 0.005), minor=True)
        
        ax1.grid(True, which='major', linewidth=0.8, alpha=0.6, color='black')
        ax1.grid(True, which='minor', linewidth=0.4, alpha=0.4, color='black')
    
    # Axis labels
    ax1.set_xlabel('MACH NUMBER', fontsize=11, fontweight='bold')
    ax1.set_ylabel('SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL', 
                   fontsize=9, fontweight='bold')
    
    # Draw curves
    curve_lw = random.uniform(1.0, 1.4)
    for cx, cy in curves_data:
        ax1.plot(cx, cy, 'k-', linewidth=curve_lw)
    
    # Arrows pointing to curve endpoints
    if add_arrows:
        # Altitude labels (from actual chart)
        altitudes_top = ['4500', '5000', '5500', '6000', '6500', '7000', 
                        '7500', '8000', '8500']
        altitudes_bottom = ['9000', '9500', '10,000', '11,000', '12,000',
                           '13,000', '14,000', '15,000', '17,000',
                           '18,000', '19,000', '20,000', '23,000', '25,000']
        
        # Top group arrows (curves ending higher)
        for i, (cx, cy) in enumerate(reversed(curves_data[-len(altitudes_top):])):
            if i < len(altitudes_top):
                end_x, end_y = cx[-1], cy[-1]
                # Stagger arrows slightly
                draw_arrow_annotation(ax1, end_x, end_y, altitudes_top[i], 'right')
        
        # Add text box for top group
        if add_labels:
            ax1.text(0.78, 0.092, 'TOTAL FUEL FLOW —\nPOUNDS PER HOUR',
                    fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', 
                             edgecolor='black', linewidth=1))
    
    # Envelope line (OPTIMUM CRUISE)
    if add_envelope:
        draw_envelope_line(ax1, [c[0] for c in curves_data], 
                          [c[1] for c in curves_data], 'optimum')
    
    # Legend box
    if add_legend:
        legend_text = '◄ CRUISE    DASH ►\n     AOA         AOA\n(USED FOR INTERFERENCE\nDRAG DETERMINATION)'
        ax1.text(0.32, 0.022, legend_text, fontsize=7, ha='left', va='bottom',
                bbox=dict(boxstyle='square,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1))
    
    # Random number labels (like 0.00, 25.00, 50.00, etc.)
    if add_labels:
        number_labels = ['0.00', '25.00', '50.00', '75.00', '100.00', 
                        '125.00', '150.00', '200.00', '250.00', '300.00']
        
        # Place some on the left side
        for i, lbl in enumerate(number_labels[:4]):
            ax1.text(0.18 + random.uniform(-0.01, 0.01), 
                    0.058 + i * 0.004 + random.uniform(-0.001, 0.001),
                    lbl, fontsize=8, ha='right')
        
        # Place some in the middle-bottom
        for i, lbl in enumerate(number_labels[4:]):
            ax1.text(0.55 + i * 0.04 + random.uniform(-0.01, 0.01),
                    0.022 + random.uniform(-0.002, 0.002),
                    lbl, fontsize=8, ha='center')
        
        # Vmax label
        ax1.text(0.82, 0.025, r'$V_{max}$ (MIL)', fontsize=9)
        
        # MAXIMUM ENDURANCE label
        ax1.text(0.18, 0.046, 'MAXIMUM ENDURANCE', fontsize=9, ha='left')
    
    # Convert to numpy array
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=120, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig1)
    buf1.seek(0)
    full_img = np.array(Image.open(buf1).convert('RGB'))
    full_img = cv2.resize(full_img, (W, H))
    
    # ========== MASK (curves only) ==========
    fig2, ax2 = plt.subplots(figsize=(W/100, H/100))
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.axis('off')
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    
    # Draw only curves in white
    for cx, cy in curves_data:
        ax2.plot(cx, cy, 'w-', linewidth=curve_lw + 1)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=120, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig2)
    buf2.seek(0)
    mask_img = np.array(Image.open(buf2).convert('L'))
    mask_img = cv2.resize(mask_img, (W, H))
    mask = (mask_img > 30).astype(np.uint8)
    
    return full_img, mask, curves_data


def add_scan_artifacts(
    img: np.ndarray,
    rotation: float = None,
    noise_level: float = None,
    jpeg_quality: int = None,
    brightness: float = None,
    contrast: float = None
) -> np.ndarray:
    """Apply realistic scan/photocopy artifacts."""
    
    pil_img = Image.fromarray(img)
    W, H = pil_img.size
    
    # Random values if not specified
    if rotation is None:
        rotation = random.uniform(-1.0, 1.0)
    if noise_level is None:
        noise_level = random.uniform(0.005, 0.02)
    if jpeg_quality is None:
        jpeg_quality = random.randint(40, 75)
    if brightness is None:
        brightness = random.uniform(0.9, 1.1)
    if contrast is None:
        contrast = random.uniform(0.85, 1.15)
    
    # Rotation
    pil_img = pil_img.rotate(rotation, resample=Image.BICUBIC, 
                             expand=False, fillcolor=(255, 255, 255))
    
    # Brightness/Contrast
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    
    # Noise
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    
    # JPEG compression artifacts
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(
        buf, format='JPEG', quality=jpeg_quality
    )
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    
    return np.array(pil_img)


def make_training_sample(
    W: int = 512,
    H: int = 512,
    n_curves: int = None,
    add_artifacts: bool = True,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single training sample.
    
    Returns:
        image: RGB numpy array (H, W, 3)
        mask: Binary numpy array (H, W) with 0/1
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if n_curves is None:
        n_curves = random.randint(10, 20)
    
    # Random variations in what elements to include
    add_grid = random.random() < 0.95
    add_arrows = random.random() < 0.85
    add_labels = random.random() < 0.80
    add_envelope = random.random() < 0.60
    add_legend = random.random() < 0.70
    
    # Generate chart
    full_img, mask, _ = create_chart_image(
        n_curves=n_curves,
        W=W, H=H,
        add_grid=add_grid,
        add_arrows=add_arrows,
        add_labels=add_labels,
        add_envelope=add_envelope,
        add_legend=add_legend,
        seed=seed
    )
    
    # Add artifacts
    if add_artifacts:
        full_img = add_scan_artifacts(full_img)
    
    return full_img, mask


def visualize_sample(seed: int = 42):
    """Generate and display a sample."""
    img, mask = make_training_sample(W=800, H=600, seed=seed, add_artifacts=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(img)
    axes[0].set_title('Full Chart (Input)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Curve Mask (Target)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return img, mask


if __name__ == "__main__":
    # Import scipy for gaussian filter
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run(['pip', 'install', 'scipy'], check=True)
    
    print("Generating sample charts...")
    
    # Generate without artifacts to see clearly
    img1, mask1 = make_training_sample(W=900, H=700, seed=42, add_artifacts=False)
    cv2.imwrite('synth_v2_clean.png', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite('synth_v2_mask.png', mask1 * 255)
    
    # Generate with artifacts
    img2, mask2 = make_training_sample(W=900, H=700, seed=42, add_artifacts=True)
    cv2.imwrite('synth_v2_artifacts.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    
    print("✅ Saved:")
    print("   - synth_v2_clean.png (without artifacts)")
    print("   - synth_v2_artifacts.png (with scan artifacts)")
    print("   - synth_v2_mask.png (target mask)")
    
    # Comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Synthetic Chart (Clean)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask1, cmap='gray')
    axes[0, 1].set_title('Target Mask', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img2)
    axes[1, 0].set_title('Synthetic Chart (With Artifacts)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Show colored curves like the target
    colored = np.ones((*mask1.shape, 3), dtype=np.uint8) * 255
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask1)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))
    for i in range(1, num_labels):
        color = (colors[i][:3] * 255).astype(np.uint8)
        colored[labels == i] = color
    
    axes[1, 1].imshow(colored)
    axes[1, 1].set_title('Extracted Curves (Colored)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('synth_v2_comparison.png', dpi=150)
    plt.show()
    
    print("\n✅ Saved synth_v2_comparison.png")
