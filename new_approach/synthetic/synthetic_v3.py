# synthetic_v3.py
"""
Realistic aircraft performance chart generator - V3
Matches the actual chart curve shapes exactly.
"""

import numpy as np
import cv2
import random
from typing import Tuple, List, Optional
import io
from PIL import Image, ImageEnhance, ImageFilter


def generate_curve_v3(
    x: np.ndarray,
    curve_index: int,
    total_curves: int
) -> np.ndarray:
    """
    Generate curve matching actual aircraft chart shape.
    
    Key observations from real chart:
    - All curves START from nearly same point (convergence at low Mach)
    - Sharp rise to peak around Mach 0.35-0.40
    - Gradual decline after peak
    - Higher indexed curves = higher altitude = better range = higher curve
    - At high Mach, curves DIVERGE (spread apart)
    """
    # Altitude factor (0 = lowest curve, 1 = highest curve)
    alt = curve_index / max(total_curves - 1, 1)
    
    # All curves start from approximately the same point
    start_y = 0.011 + random.uniform(-0.001, 0.001)
    
    # Peak characteristics - higher altitude = higher peak
    peak_y = 0.038 + alt * 0.022  # Range from ~0.038 to ~0.060
    peak_x = 0.37 + random.uniform(-0.02, 0.02)  # Peak around Mach 0.35-0.40
    
    # End point - THIS IS WHERE DIVERGENCE HAPPENS
    # Higher altitude curves stay higher at high Mach
    end_y = 0.020 + alt * 0.013  # Range from ~0.020 to ~0.033
    
    y = np.zeros_like(x)
    
    # Key transition points
    x_start = x[0]  # ~0.15
    x_rise = 0.20   # Where steep rise begins
    x_peak = peak_x
    x_end = x[-1]   # ~0.90
    
    for i, xi in enumerate(x):
        if xi <= x_rise:
            # Initial slow rise (all curves together)
            t = (xi - x_start) / (x_rise - x_start + 1e-8)
            y[i] = start_y + (peak_y * 0.15) * (t ** 1.5)
            
        elif xi <= x_peak:
            # Steep rise to peak
            t = (xi - x_rise) / (x_peak - x_rise + 1e-8)
            y_at_rise = start_y + (peak_y * 0.15)
            # Smooth acceleration then deceleration
            y[i] = y_at_rise + (peak_y - y_at_rise) * (1 - (1 - t) ** 2.2)
            
        else:
            # Gradual decline from peak
            t = (xi - x_peak) / (x_end - x_peak + 1e-8)
            # Smooth decline
            y[i] = peak_y - (peak_y - end_y) * (t ** 0.8)
    
    return y


def draw_chart_cv(
    n_curves: int = 15,
    W: int = 900,
    H: int = 700,
    add_grid: bool = True,
    add_arrows: bool = True,
    add_text: bool = True,
    add_envelope: bool = True,
    curve_thickness: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw chart using OpenCV for speed.
    
    Returns:
        full_img: RGB image with all elements
        mask: Binary mask (only curves)
        colored: Curves with different colors (target output style)
    """
    # Create images
    full_img = np.ones((H, W, 3), dtype=np.uint8) * 255
    mask = np.zeros((H, W), dtype=np.uint8)
    colored = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # Plot area
    margin = {'left': 80, 'right': 100, 'top': 40, 'bottom': 60}
    plot_w = W - margin['left'] - margin['right']
    plot_h = H - margin['top'] - margin['bottom']
    
    # Axis ranges (matching real chart)
    x_range = (0.10, 1.00)
    y_range = (0.01, 0.10)
    
    def to_px(x_val, y_val):
        """Convert data coordinates to pixel coordinates."""
        px = int(margin['left'] + (x_val - x_range[0]) / (x_range[1] - x_range[0]) * plot_w)
        py = int(H - margin['bottom'] - (y_val - y_range[0]) / (y_range[1] - y_range[0]) * plot_h)
        return (px, py)
    
    # Draw grid
    if add_grid:
        # Major grid (every 0.1 on x, every 0.01 on y)
        for xv in np.arange(0.1, 1.01, 0.1):
            px, _ = to_px(xv, y_range[0])
            cv2.line(full_img, (px, margin['top']), (px, H - margin['bottom']), (180, 180, 180), 1)
        
        for yv in np.arange(0.01, 0.101, 0.01):
            _, py = to_px(x_range[0], yv)
            cv2.line(full_img, (margin['left'], py), (W - margin['right'], py), (180, 180, 180), 1)
        
        # Minor grid
        for xv in np.arange(0.1, 1.01, 0.05):
            px, _ = to_px(xv, y_range[0])
            cv2.line(full_img, (px, margin['top']), (px, H - margin['bottom']), (210, 210, 210), 1)
        
        for yv in np.arange(0.01, 0.101, 0.005):
            _, py = to_px(x_range[0], yv)
            cv2.line(full_img, (margin['left'], py), (W - margin['right'], py), (210, 210, 210), 1)
    
    # Draw border
    cv2.rectangle(full_img, 
                  (margin['left'], margin['top']), 
                  (W - margin['right'], H - margin['bottom']), 
                  (0, 0, 0), 2)
    
    # Generate curves
    x = np.linspace(0.15, 0.90, 500)  # Smooth curves
    curves_data = []
    
    for i in range(n_curves):
        y = generate_curve_v3(x, i, n_curves)
        curves_data.append((x, y))
    
    # Rainbow colors for colored output
    colors = []
    for i in range(n_curves):
        # OpenCV HSV: H=0-179, S=0-255, V=0-255
        hue = int(150 - (i / max(n_curves - 1, 1)) * 150)  # Blue to Red (0-150)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color_bgr)))
    
    # Draw curves
    for idx, (cx, cy) in enumerate(curves_data):
        points = [to_px(cx[j], cy[j]) for j in range(len(cx))]
        
        # Draw on full image (black)
        for j in range(len(points) - 1):
            cv2.line(full_img, points[j], points[j+1], (0, 0, 0), curve_thickness, cv2.LINE_AA)
        
        # Draw on mask (white)
        for j in range(len(points) - 1):
            cv2.line(mask, points[j], points[j+1], 255, curve_thickness, cv2.LINE_AA)
        
        # Draw on colored (rainbow)
        for j in range(len(points) - 1):
            cv2.line(colored, points[j], points[j+1], colors[idx], curve_thickness, cv2.LINE_AA)
    
    # Envelope line (OPTIMUM CRUISE - connects peaks)
    if add_envelope:
        peaks = []
        for cx, cy in curves_data:
            peak_idx = np.argmax(cy)
            peaks.append(to_px(cx[peak_idx], cy[peak_idx]))
        
        # Sort by y (top to bottom in image = higher to lower value)
        peaks.sort(key=lambda p: p[1])
        
        # Draw line through peaks
        for j in range(len(peaks) - 1):
            cv2.line(full_img, peaks[j], peaks[j+1], (0, 0, 0), 2, cv2.LINE_AA)
    
    # Arrows pointing to curve endpoints
    if add_arrows:
        altitudes = ['4500', '5000', '5500', '6000', '6500', '7000', '7500', 
                    '8000', '8500', '9000', '9500', '10,000', '11,000', 
                    '12,000', '13,000', '14,000', '15,000', '17,000']
        
        for idx, (cx, cy) in enumerate(reversed(curves_data)):
            if idx >= len(altitudes):
                break
            
            end_pt = to_px(cx[-1], cy[-1])
            
            # Arrow line
            arrow_start = (end_pt[0] + 40, end_pt[1] + random.randint(-5, 5))
            cv2.arrowedLine(full_img, arrow_start, end_pt, (0, 0, 0), 1, cv2.LINE_AA, tipLength=0.3)
            
            # Label
            cv2.putText(full_img, altitudes[idx], (arrow_start[0] + 5, arrow_start[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Text labels
    if add_text:
        # Axis labels
        cv2.putText(full_img, 'MACH NUMBER', (W//2 - 50, H - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Y axis label (rotated text is hard in cv2, simplified)
        cv2.putText(full_img, 'SPECIFIC RANGE', (10, H//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # OPTIMUM CRUISE label
        cv2.putText(full_img, 'OPTIMUM CRUISE', (120, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # TOTAL FUEL FLOW box
        box_x, box_y = W - margin['right'] - 120, margin['top'] + 20
        cv2.rectangle(full_img, (box_x, box_y), (box_x + 115, box_y + 45), (0, 0, 0), 1)
        cv2.putText(full_img, 'TOTAL FUEL FLOW -', (box_x + 5, box_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(full_img, 'POUNDS PER HOUR', (box_x + 5, box_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Number labels near curves
        for lbl in ['0.00', '25.00', '50.00', '75.00']:
            lx = random.randint(130, 200)
            ly = random.randint(200, 400)
            cv2.putText(full_img, lbl, (lx, ly),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Legend box
        legend_x, legend_y = 200, H - margin['bottom'] - 100
        cv2.rectangle(full_img, (legend_x, legend_y), (legend_x + 140, legend_y + 55), (0, 0, 0), 1)
        cv2.rectangle(full_img, (legend_x, legend_y), (legend_x + 140, legend_y + 55), (255, 255, 255), -1)
        cv2.rectangle(full_img, (legend_x, legend_y), (legend_x + 140, legend_y + 55), (0, 0, 0), 1)
        cv2.putText(full_img, 'CRUISE    DASH', (legend_x + 10, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(full_img, '  AOA       AOA', (legend_x + 10, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(full_img, '(USED FOR INTERFERENCE', (legend_x + 5, legend_y + 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
        cv2.putText(full_img, 'DRAG DETERMINATION)', (legend_x + 5, legend_y + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
    
    # Tick labels
    for xv in np.arange(0.1, 1.01, 0.1):
        px, py = to_px(xv, y_range[0])
        cv2.putText(full_img, f'{xv:.1f}', (px - 10, py + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    for yv in np.arange(0.01, 0.101, 0.01):
        px, py = to_px(x_range[0], yv)
        cv2.putText(full_img, f'{yv:.2f}', (px - 35, py + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    return full_img, mask, colored


def add_artifacts(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Add scan/photocopy artifacts."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Slight rotation
    angle = random.uniform(-1.0, 1.0) * strength
    pil_img = pil_img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    
    # Brightness/contrast
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.92, 1.08))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.9, 1.1))
    
    # Noise
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.01 * strength, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    
    # JPEG artifacts
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format='JPEG', quality=random.randint(50, 80))
    buf.seek(0)
    result = np.array(Image.open(buf).convert('RGB'))
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def make_sample(
    W: int = 512,
    H: int = 512,
    n_curves: int = None,
    add_noise: bool = True,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a training sample.
    
    Returns:
        image: BGR numpy array
        mask: Binary mask (0/255)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if n_curves is None:
        n_curves = random.randint(10, 18)
    
    # Random variations
    add_grid = random.random() < 0.9
    add_arrows = random.random() < 0.8
    add_text = random.random() < 0.75
    add_envelope = random.random() < 0.6
    
    full_img, mask, _ = draw_chart_cv(
        n_curves=n_curves,
        W=W, H=H,
        add_grid=add_grid,
        add_arrows=add_arrows,
        add_text=add_text,
        add_envelope=add_envelope
    )
    
    if add_noise:
        full_img = add_artifacts(full_img)
    
    return full_img, mask


if __name__ == "__main__":
    print("Generating V3 samples...")
    
    # Clean version
    full, mask, colored = draw_chart_cv(n_curves=15, W=900, H=700)
    cv2.imwrite('v3_full.png', full)
    cv2.imwrite('v3_mask.png', mask)
    cv2.imwrite('v3_colored.png', colored)
    
    # With artifacts
    full_noisy = add_artifacts(full)
    cv2.imwrite('v3_noisy.png', full_noisy)
    
    print("Saved: v3_full.png, v3_mask.png, v3_colored.png, v3_noisy.png")
    
    # Multiple samples
    for i in range(3):
        img, msk = make_sample(W=512, H=512, seed=i)
        cv2.imwrite(f'v3_sample_{i}.png', img)
        cv2.imwrite(f'v3_sample_{i}_mask.png', msk)
    
    print("Saved 3 random samples")
