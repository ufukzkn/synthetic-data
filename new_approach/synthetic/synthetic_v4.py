# synthetic_v4.py
"""
Comprehensive synthetic aircraft chart generator - V4
Based on analysis of 10+ real chart examples.

Key variations to cover:
1. Different curve shapes: rising, peaked, falling, flat
2. Different curve counts: 3-18
3. Large arrows at curve endpoints
4. Envelope lines (OPTIMUM CRUISE, MAXIMUM ENDURANCE)
5. Variable Mach ranges (0.3-0.9, 0.4-1.1, 0.5-1.4, etc.)
6. Text boxes and labels
7. Grid with varying density
"""

import numpy as np
import cv2
import random
from typing import Tuple, List, Optional, Dict
import io
from PIL import Image, ImageEnhance, ImageFilter
from dataclasses import dataclass


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    # Axis ranges
    x_min: float = 0.30
    x_max: float = 1.00
    y_min: float = 0.04
    y_max: float = 0.15
    
    # Curve settings
    n_curves: int = 8
    curve_type: str = 'peaked'  # 'peaked', 'rising', 'falling', 'mixed'
    
    # Elements to include
    add_grid: bool = True
    add_arrows: bool = True
    add_envelope_optimum: bool = True
    add_envelope_endurance: bool = False
    add_vmax_line: bool = False
    add_text_boxes: bool = True
    add_fuel_labels: bool = True
    add_drag_labels: bool = True
    
    # Style
    curve_thickness: int = 1  # Thinner curves like real data
    arrow_length: int = 50


def generate_curve_shape(
    x: np.ndarray,
    curve_type: str,
    curve_index: int,
    total_curves: int,
    params: Dict = None
) -> np.ndarray:
    """
    Generate different curve shapes based on type.
    
    Types:
    - 'peaked': Rise to peak, then fall (like Figure 5-147, 5-154)
    - 'rising': Monotonically rising (like Figure 5-158, 5-160)
    - 'falling': Monotonically falling
    - 'flat': Nearly horizontal with slight slope
    """
    if params is None:
        params = {}
    
    alt = curve_index / max(total_curves - 1, 1)  # 0 to 1
    
    # Normalize x to 0-1 for shape calculation
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    if curve_type == 'peaked':
        # Peak position varies slightly
        peak_pos = params.get('peak_pos', 0.3) + random.uniform(-0.05, 0.05)
        
        # All curves converge at start
        start_y = 0.15 + random.uniform(-0.02, 0.02)
        
        # Peak height varies with altitude
        peak_y = 0.50 + alt * 0.35 + random.uniform(-0.03, 0.03)
        
        # End point - curves diverge
        end_y = 0.25 + alt * 0.20 + random.uniform(-0.02, 0.02)
        
        y = np.zeros_like(x_norm)
        for i, t in enumerate(x_norm):
            if t <= peak_pos:
                # Rising to peak
                progress = t / peak_pos
                y[i] = start_y + (peak_y - start_y) * (1 - (1 - progress) ** 2)
            else:
                # Falling from peak
                progress = (t - peak_pos) / (1 - peak_pos)
                y[i] = peak_y - (peak_y - end_y) * (progress ** 0.7)
        
    elif curve_type == 'rising':
        # Curves that rise from left to right
        start_y = 0.10 + alt * 0.05 + random.uniform(-0.02, 0.02)
        end_y = 0.60 + alt * 0.25 + random.uniform(-0.03, 0.03)
        
        # Slight curve (not perfectly linear)
        curvature = random.uniform(0.8, 1.2)
        y = start_y + (end_y - start_y) * (x_norm ** curvature)
        
    elif curve_type == 'falling':
        # Curves that fall from left to right
        start_y = 0.70 + alt * 0.20 + random.uniform(-0.03, 0.03)
        end_y = 0.15 + alt * 0.10 + random.uniform(-0.02, 0.02)
        
        curvature = random.uniform(0.6, 1.0)
        y = start_y - (start_y - end_y) * (x_norm ** curvature)
        
    elif curve_type == 'flat':
        # Nearly horizontal curves
        base_y = 0.30 + alt * 0.40 + random.uniform(-0.02, 0.02)
        slope = random.uniform(-0.1, 0.1)
        y = base_y + slope * x_norm
        
    else:  # mixed - random choice per curve
        return generate_curve_shape(x, random.choice(['peaked', 'rising', 'falling']), 
                                   curve_index, total_curves, params)
    
    return y


def draw_large_arrow(img, start_pt, end_pt, color=(0, 0, 0), thickness=1):
    """
    Draw a large arrow like in real charts.
    Arrow points FROM start TO end.
    """
    # Draw the line
    cv2.line(img, start_pt, end_pt, color, thickness, cv2.LINE_AA)
    
    # Calculate arrow head
    angle = np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
    arrow_len = 12
    arrow_angle = np.pi / 6  # 30 degrees
    
    # Arrow head points
    pt1 = (int(end_pt[0] - arrow_len * np.cos(angle - arrow_angle)),
           int(end_pt[1] - arrow_len * np.sin(angle - arrow_angle)))
    pt2 = (int(end_pt[0] - arrow_len * np.cos(angle + arrow_angle)),
           int(end_pt[1] - arrow_len * np.sin(angle + arrow_angle)))
    
    # Draw arrow head
    cv2.line(img, end_pt, pt1, color, thickness, cv2.LINE_AA)
    cv2.line(img, end_pt, pt2, color, thickness, cv2.LINE_AA)


def draw_chart(config: ChartConfig, W: int = 800, H: int = 600) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw a synthetic chart based on configuration.
    
    Returns:
        full_img: BGR image with all elements
        mask: Binary mask (curves only)
        colored: Curves with different colors
    """
    # Create images
    full_img = np.ones((H, W, 3), dtype=np.uint8) * 255
    mask = np.zeros((H, W), dtype=np.uint8)
    colored = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # Plot area margins
    margin = {'left': 70, 'right': 90, 'top': 50, 'bottom': 50}
    plot_w = W - margin['left'] - margin['right']
    plot_h = H - margin['top'] - margin['bottom']
    
    def to_px(x_val, y_val):
        """Convert data coords to pixel coords."""
        px = int(margin['left'] + (x_val - config.x_min) / (config.x_max - config.x_min) * plot_w)
        py = int(H - margin['bottom'] - (y_val - config.y_min) / (config.y_max - config.y_min) * plot_h)
        return (px, py)
    
    # Draw grid
    if config.add_grid:
        # Calculate appropriate grid spacing
        x_range = config.x_max - config.x_min
        y_range = config.y_max - config.y_min
        
        x_major = 0.1 if x_range > 0.5 else 0.05
        y_major = 0.01 if y_range < 0.1 else 0.02
        
        # Major grid
        for xv in np.arange(config.x_min, config.x_max + 0.001, x_major):
            px, _ = to_px(xv, config.y_min)
            cv2.line(full_img, (px, margin['top']), (px, H - margin['bottom']), (180, 180, 180), 1)
        
        for yv in np.arange(config.y_min, config.y_max + 0.001, y_major):
            _, py = to_px(config.x_min, yv)
            cv2.line(full_img, (margin['left'], py), (W - margin['right'], py), (180, 180, 180), 1)
        
        # Minor grid
        for xv in np.arange(config.x_min, config.x_max + 0.001, x_major / 2):
            px, _ = to_px(xv, config.y_min)
            cv2.line(full_img, (px, margin['top']), (px, H - margin['bottom']), (210, 210, 210), 1)
        
        for yv in np.arange(config.y_min, config.y_max + 0.001, y_major / 2):
            _, py = to_px(config.x_min, yv)
            cv2.line(full_img, (margin['left'], py), (W - margin['right'], py), (210, 210, 210), 1)
    
    # Draw border
    cv2.rectangle(full_img, (margin['left'], margin['top']), 
                  (W - margin['right'], H - margin['bottom']), (0, 0, 0), 2)
    
    # Generate curves
    x = np.linspace(config.x_min + 0.02, config.x_max - 0.02, 400)
    curves_data = []
    
    for i in range(config.n_curves):
        y_norm = generate_curve_shape(x, config.curve_type, i, config.n_curves)
        # Scale y to actual axis range
        y = config.y_min + y_norm * (config.y_max - config.y_min)
        y = np.clip(y, config.y_min + 0.002, config.y_max - 0.002)
        curves_data.append((x.copy(), y))
    
    # Rainbow colors for colored output
    colors = []
    for i in range(config.n_curves):
        hue = int(140 - (i / max(config.n_curves - 1, 1)) * 140)  # Cyan to Red
        color_hsv = np.uint8([[[hue, 255, 220]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color_bgr)))
    
    # Draw curves
    for idx, (cx, cy) in enumerate(curves_data):
        points = [to_px(cx[j], cy[j]) for j in range(len(cx))]
        
        # Full image (black)
        for j in range(len(points) - 1):
            cv2.line(full_img, points[j], points[j+1], (0, 0, 0), config.curve_thickness, cv2.LINE_AA)
        
        # Mask (white)
        for j in range(len(points) - 1):
            cv2.line(mask, points[j], points[j+1], 255, config.curve_thickness, cv2.LINE_AA)
        
        # Colored
        for j in range(len(points) - 1):
            cv2.line(colored, points[j], points[j+1], colors[idx], config.curve_thickness, cv2.LINE_AA)
    
    # Draw OPTIMUM CRUISE envelope (connects peaks or end points)
    if config.add_envelope_optimum:
        if config.curve_type == 'peaked':
            # Connect peaks
            envelope_pts = []
            for cx, cy in curves_data:
                peak_idx = np.argmax(cy)
                envelope_pts.append(to_px(cx[peak_idx], cy[peak_idx]))
        else:
            # Connect a point along each curve
            envelope_pts = []
            connect_pos = random.uniform(0.4, 0.7)
            for cx, cy in curves_data:
                idx = int(len(cx) * connect_pos)
                envelope_pts.append(to_px(cx[idx], cy[idx]))
        
        # Sort by y position
        envelope_pts.sort(key=lambda p: p[1])
        
        # Draw envelope line
        for j in range(len(envelope_pts) - 1):
            cv2.line(full_img, envelope_pts[j], envelope_pts[j+1], (0, 0, 0), 2, cv2.LINE_AA)
        
        # Label
        cv2.putText(full_img, 'OPTIMUM', (envelope_pts[0][0] - 70, envelope_pts[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(full_img, 'CRUISE', (envelope_pts[0][0] - 60, envelope_pts[0][1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw MAXIMUM ENDURANCE envelope
    if config.add_envelope_endurance:
        envelope_pts = []
        for cx, cy in curves_data:
            # Find a different characteristic point
            idx = int(len(cx) * 0.25)
            envelope_pts.append(to_px(cx[idx], cy[idx]))
        
        envelope_pts.sort(key=lambda p: p[1])
        
        for j in range(len(envelope_pts) - 1):
            cv2.line(full_img, envelope_pts[j], envelope_pts[j+1], (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(full_img, 'MAXIMUM', (envelope_pts[-1][0] - 70, envelope_pts[-1][1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(full_img, 'ENDURANCE', (envelope_pts[-1][0] - 80, envelope_pts[-1][1] + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw Vmax line (dashed)
    if config.add_vmax_line:
        vmax_pts = []
        for cx, cy in curves_data:
            # Near end of each curve
            idx = int(len(cx) * 0.85)
            vmax_pts.append(to_px(cx[idx], cy[idx]))
        
        vmax_pts.sort(key=lambda p: p[1])
        
        # Dashed line
        for j in range(len(vmax_pts) - 1):
            pt1, pt2 = vmax_pts[j], vmax_pts[j+1]
            # Draw dashed
            dash_len = 8
            dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            n_dashes = int(dist / dash_len / 2)
            for k in range(n_dashes):
                t1 = k * 2 * dash_len / dist
                t2 = (k * 2 + 1) * dash_len / dist
                if t2 > 1:
                    t2 = 1
                p1 = (int(pt1[0] + t1 * (pt2[0] - pt1[0])), int(pt1[1] + t1 * (pt2[1] - pt1[1])))
                p2 = (int(pt1[0] + t2 * (pt2[0] - pt1[0])), int(pt1[1] + t2 * (pt2[1] - pt1[1])))
                cv2.line(full_img, p1, p2, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Label
        cv2.putText(full_img, 'Vmax(MIL)', (vmax_pts[-1][0] - 30, vmax_pts[-1][1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # Draw arrows at curve endpoints
    if config.add_arrows:
        fuel_flows = ['3000', '3500', '4000', '4500', '5000', '5500', '6000', 
                     '6500', '7000', '7500', '8000', '8500']
        
        for idx, (cx, cy) in enumerate(reversed(curves_data)):
            if idx >= len(fuel_flows):
                break
            
            # End point of curve
            end_x, end_y = cx[-1], cy[-1]
            end_px = to_px(end_x, end_y)
            
            # Arrow starts from right, points to curve
            arrow_len = config.arrow_length + random.randint(-10, 10)
            arrow_start = (end_px[0] + arrow_len, end_px[1] + random.randint(-8, 8))
            
            # Draw arrow (pointing TO the curve)
            draw_large_arrow(full_img, arrow_start, end_px, (0, 0, 0), 1)
            
            # Fuel flow label
            cv2.putText(full_img, fuel_flows[idx], (arrow_start[0] + 5, arrow_start[1] + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # Text boxes
    if config.add_text_boxes:
        # TOTAL FUEL FLOW box (top right usually)
        box_x = W - margin['right'] - 10
        box_y = margin['top'] + 30
        
        cv2.putText(full_img, 'TOTAL FUEL FLOW-', (box_x - 95, box_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(full_img, 'POUNDS PER HOUR', (box_x - 95, box_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Legend box (CRUISE/DASH AOA)
        if config.add_text_boxes:
            legend_x = margin['left'] + 30
            legend_y = margin['top'] + 30
            box_w, box_h = 130, 45
            
            cv2.rectangle(full_img, (legend_x, legend_y), (legend_x + box_w, legend_y + box_h), (0, 0, 0), 1)
            cv2.rectangle(full_img, (legend_x, legend_y), (legend_x + box_w, legend_y + box_h), (255, 255, 255), -1)
            cv2.rectangle(full_img, (legend_x, legend_y), (legend_x + box_w, legend_y + box_h), (0, 0, 0), 1)
            
            cv2.putText(full_img, 'CRUISE    DASH', (legend_x + 15, legend_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(full_img, '  AOA       AOA', (legend_x + 15, legend_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            cv2.putText(full_img, '(USED FOR INTERFERENCE', (legend_x + 5, legend_y + 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 0, 0), 1)
            
            # Small arrows in legend
            cv2.arrowedLine(full_img, (legend_x + 10, legend_y + 12), (legend_x + 30, legend_y + 12), (0, 0, 0), 1)
            cv2.arrowedLine(full_img, (legend_x + 120, legend_y + 12), (legend_x + 100, legend_y + 12), (0, 0, 0), 1)
    
    # Drag index labels (0.00, 25.00, 50.00, etc.)
    if config.add_drag_labels:
        drag_labels = ['0.00', '25.00', '50.00', '75.00', '100.00', '125.00', '150.00']
        
        # Place near bottom of plot
        base_x = margin['left'] + plot_w * 0.6
        base_y = H - margin['bottom'] - 40
        
        for i, lbl in enumerate(drag_labels[:random.randint(3, 7)]):
            cv2.putText(full_img, lbl, 
                       (int(base_x + random.uniform(-20, 20)), int(base_y + i * 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Axis labels
    cv2.putText(full_img, 'MACH NUMBER', (W // 2 - 40, H - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Tick labels
    x_major = 0.1 if (config.x_max - config.x_min) > 0.5 else 0.05
    for xv in np.arange(config.x_min, config.x_max + 0.001, x_major):
        px, py = to_px(xv, config.y_min)
        cv2.putText(full_img, f'{xv:.2f}', (px - 12, py + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    
    y_major = 0.01 if (config.y_max - config.y_min) < 0.1 else 0.02
    for yv in np.arange(config.y_min, config.y_max + 0.001, y_major):
        px, py = to_px(config.x_min, yv)
        cv2.putText(full_img, f'{yv:.2f}', (px - 35, py + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    
    return full_img, mask, colored


def random_config() -> ChartConfig:
    """Generate a random chart configuration."""
    # Random axis ranges (based on real charts)
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
    
    return ChartConfig(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_curves=random.randint(4, 15),
        curve_type=random.choice(['peaked', 'peaked', 'rising', 'falling', 'mixed']),
        add_grid=random.random() < 0.95,
        add_arrows=random.random() < 0.85,
        add_envelope_optimum=random.random() < 0.70,
        add_envelope_endurance=random.random() < 0.40,
        add_vmax_line=random.random() < 0.30,
        add_text_boxes=random.random() < 0.75,
        add_fuel_labels=random.random() < 0.80,
        add_drag_labels=random.random() < 0.60,
        curve_thickness=random.choice([1, 2, 2]),
        arrow_length=random.randint(40, 70)
    )


def add_scan_artifacts(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Add scan/photocopy artifacts."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Rotation
    angle = random.uniform(-1.5, 1.5) * strength
    pil_img = pil_img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    
    # Brightness/contrast
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.88, 1.12))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.85, 1.15))
    
    # Noise
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.015 * strength, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    
    # JPEG artifacts
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format='JPEG', 
                                                        quality=random.randint(45, 75))
    buf.seek(0)
    result = np.array(Image.open(buf).convert('RGB'))
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def make_sample(W: int = 512, H: int = 512, seed: int = None, 
                add_artifacts: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single training sample.
    
    Returns:
        image: BGR numpy array
        mask: Binary mask (0/255)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    config = random_config()
    full_img, mask, _ = draw_chart(config, W, H)
    
    if add_artifacts:
        full_img = add_scan_artifacts(full_img)
    
    return full_img, mask


if __name__ == "__main__":
    print("Generating V4 samples with various configurations...")
    
    # Test different curve types
    types = ['peaked', 'rising', 'falling', 'mixed']
    
    for i, ctype in enumerate(types):
        config = ChartConfig(
            x_min=0.30, x_max=1.10,
            y_min=0.06, y_max=0.16,
            n_curves=8,
            curve_type=ctype,
            add_envelope_optimum=True,
            add_envelope_endurance=(i % 2 == 0),
            add_vmax_line=(i % 3 == 0)
        )
        
        full, mask, colored = draw_chart(config, W=800, H=600)
        cv2.imwrite(f'v4_{ctype}.png', full)
        cv2.imwrite(f'v4_{ctype}_mask.png', mask)
        cv2.imwrite(f'v4_{ctype}_colored.png', colored)
        print(f"  Saved v4_{ctype}*.png")
    
    # Random samples
    print("\nGenerating random samples...")
    for i in range(5):
        full, mask = make_sample(W=512, H=512, seed=i, add_artifacts=True)
        cv2.imwrite(f'v4_random_{i}.png', full)
        cv2.imwrite(f'v4_random_{i}_mask.png', mask)
    
    print("Done! Generated 4 typed + 5 random samples")
