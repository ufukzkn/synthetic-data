# synthetic_v5.py
"""
Synthetic aircraft chart generator - V5
Uses matplotlib for realistic arrows (like original code).
Thinner curves, proper arrow styles.
"""

import io
import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import cv2


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
    curve_lw: float = 0.6  # Thinner like real data
    
    # Elements
    add_grid: bool = True
    add_arrows: bool = True
    add_envelope_optimum: bool = True
    add_envelope_endurance: bool = False
    add_vmax_line: bool = False
    add_text_boxes: bool = True
    add_fuel_labels: bool = True
    add_drag_labels: bool = True


@dataclass(frozen=True)
class AxisDetectionResult:
    origin_px: tuple[float, float]
    xref_px: tuple[float, float]
    yref_px: tuple[float, float]
    x_axis_line: tuple[int, int, int, int]
    y_axis_line: tuple[int, int, int, int]


def _len2(x1: int, y1: int, x2: int, y2: int) -> float:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    return dx * dx + dy * dy


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _pick_x_axis(segs: list[tuple[int, int, int, int]], w: int, h: int, curve_bbox: Optional[tuple[int,int,int,int]] = None) -> Optional[tuple[int, int, int, int]]:
    """Pick X axis: horizontal line just below curves."""
    candidates = []
    min_length = w * 0.25
    
    # If we have curve bbox, X axis should be at or below curve bottom
    curve_bottom = curve_bbox[3] if curve_bbox else h * 0.5
    
    for x1, y1, x2, y2 in segs:
        length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        if length < min_length:
            continue
            
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 15 or angle > 165:
            y_mid = (y1 + y2) / 2
            # X axis should be at or below curve bottom
            if y_mid >= curve_bottom - 20:  # Small tolerance
                candidates.append((y_mid, length, (x1, y1, x2, y2)))
    
    if not candidates:
        return None
    
    # Pick the one closest to curve bottom (but still below)
    candidates.sort(key=lambda x: (x[0], -x[1]))
    best = candidates[0][2]
    x1, y1, x2, y2 = best
    
    if x1 > x2:
        return (x2, y2, x1, y1)
    return best


def _pick_y_axis(segs: list[tuple[int, int, int, int]], w: int, h: int, curve_bbox: Optional[tuple[int,int,int,int]] = None) -> Optional[tuple[int, int, int, int]]:
    """Pick Y axis: vertical line just left of curves."""
    candidates = []
    min_length = h * 0.25
    
    # If we have curve bbox, Y axis should be at or left of curve left edge
    curve_left = curve_bbox[0] if curve_bbox else w * 0.5
    
    for x1, y1, x2, y2 in segs:
        length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        if length < min_length:
            continue
            
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if 75 < angle < 105:
            x_mid = (x1 + x2) / 2
            # Y axis should be at or left of curve left edge
            if x_mid <= curve_left + 20:  # Small tolerance
                candidates.append((x_mid, length, (x1, y1, x2, y2)))
    
    if not candidates:
        return None
    
    # Pick the one closest to curve left (but still left of it)
    candidates.sort(key=lambda x: (-x[0], -x[1]))  # Rightmost (closest to curves) first
    best = candidates[0][2]
    x1, y1, x2, y2 = best
    
    if y1 < y2:
        return (x2, y2, x1, y1)
    return best


def _extend_axes_to_intersection(x_axis, y_axis):
    """Extend axes to their intersection point."""
    if x_axis is None or y_axis is None:
        return x_axis, y_axis
    
    x1, y1, x2, y2 = x_axis
    x3, y3, x4, y4 = y_axis
    
    # Calculate slopes
    x_dx = x2 - x1
    x_dy = y2 - y1
    y_dx = x4 - x3
    y_dy = y4 - y3
    
    # Avoid division by zero
    if x_dx == 0:
        x_slope = float('inf')
    else:
        x_slope = x_dy / x_dx
        
    if y_dx == 0:
        y_slope = float('inf')
    else:
        y_slope = y_dy / y_dx
    
    # Check if lines are parallel
    if x_slope == y_slope:
        return x_axis, y_axis
    
    # Calculate intersection point
    if x_slope == float('inf'):
        intersection_x = x1
        y_intercept = y3 - y_slope * x3
        intersection_y = y_slope * intersection_x + y_intercept
    elif y_slope == float('inf'):
        intersection_x = x3
        x_intercept = y1 - x_slope * x1
        intersection_y = x_slope * intersection_x + x_intercept
    else:
        x_intercept = y1 - x_slope * x1
        y_intercept = y3 - y_slope * x3
        intersection_x = (y_intercept - x_intercept) / (x_slope - y_slope)
        intersection_y = x_slope * intersection_x + x_intercept
    
    ix, iy = int(round(intersection_x)), int(round(intersection_y))
    
    # Extend x_axis: keep the farther endpoint
    if (x1 - ix)**2 + (y1 - iy)**2 > (x2 - ix)**2 + (y2 - iy)**2:
        new_x_axis = (ix, iy, x1, y1)
    else:
        new_x_axis = (ix, iy, x2, y2)
    
    # Extend y_axis: keep the farther endpoint
    if (x3 - ix)**2 + (y3 - iy)**2 > (x4 - ix)**2 + (y4 - iy)**2:
        new_y_axis = (ix, iy, x3, y3)
    else:
        new_y_axis = (ix, iy, x4, y4)
    
    return new_x_axis, new_y_axis


def compute_axes_from_config(config: 'ChartConfig', W: int, H: int) -> AxisDetectionResult:
    """
    Compute axis pixel coordinates directly from config.
    For synthetic data where we know the exact axis ranges.
    """
    # The chart is drawn with set_position([0, 0, 1, 1]) so axes map to full image
    # X axis: from x_min to x_max at y = y_min (bottom of data area)
    # Y axis: from y_min to y_max at x = x_min (left of data area)
    
    # Map data coordinates to pixel coordinates
    # In image coords: y increases downward, x increases rightward
    def data_to_px(dx, dy):
        px = int((dx - config.x_min) / (config.x_max - config.x_min) * W)
        py = int((1.0 - (dy - config.y_min) / (config.y_max - config.y_min)) * H)
        return _clamp(px, 0, W-1), _clamp(py, 0, H-1)
    
    # Origin (bottom-left of data area)
    ox, oy = data_to_px(config.x_min, config.y_min)
    
    # X axis endpoints (bottom edge)
    x1_px, y1_px = data_to_px(config.x_min, config.y_min)
    x2_px, y2_px = data_to_px(config.x_max, config.y_min)
    
    # Y axis endpoints (left edge)
    x3_px, y3_px = data_to_px(config.x_min, config.y_min)
    x4_px, y4_px = data_to_px(config.x_min, config.y_max)
    
    return AxisDetectionResult(
        origin_px=(float(ox), float(oy)),
        xref_px=(float(x2_px), float(y2_px)),  # Right end of X axis
        yref_px=(float(x4_px), float(y4_px)),  # Top end of Y axis
        x_axis_line=(x1_px, y1_px, x2_px, y2_px),
        y_axis_line=(x3_px, y3_px, x4_px, y4_px),
    )


def detect_axes_from_rgb(image_rgb: np.ndarray, curve_mask: Optional[np.ndarray] = None) -> Optional[AxisDetectionResult]:
    """Detect X/Y axes from an RGB image, using curve mask for guidance."""
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    except Exception:
        return None

    h, w = gray.shape
    
    # Get curve bounding box from mask if provided
    curve_bbox = None
    if curve_mask is not None:
        if len(curve_mask.shape) == 3:
            mask_gray = cv2.cvtColor(curve_mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = curve_mask
        ys, xs = np.where(mask_gray > 127)
        if len(xs) > 0:
            curve_bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Minimum line length proportional to image size
    min_line_len = max(50, min(w, h) // 4)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=15,
    )

    if lines is None:
        return None

    segs = [tuple(map(int, l[0])) for l in lines]
    x_axis = _pick_x_axis(segs, w, h, curve_bbox)
    y_axis = _pick_y_axis(segs, w, h, curve_bbox)
    
    if x_axis is None or y_axis is None:
        return None
    
    # Extend axes to intersection
    x_axis, y_axis = _extend_axes_to_intersection(x_axis, y_axis)
    
    # Origin is at intersection (first point of both extended axes)
    ox, oy = x_axis[0], x_axis[1]
    
    # Reference points are at the other ends
    xref_x, xref_y = x_axis[2], x_axis[3]
    yref_x, yref_y = y_axis[2], y_axis[3]
    
    # Clamp to image bounds
    ox = _clamp(ox, 0, w - 1)
    oy = _clamp(oy, 0, h - 1)
    xref_x = _clamp(xref_x, 0, w - 1)
    xref_y = _clamp(xref_y, 0, h - 1)
    yref_x = _clamp(yref_x, 0, w - 1)
    yref_y = _clamp(yref_y, 0, h - 1)

    return AxisDetectionResult(
        origin_px=(float(ox), float(oy)),
        xref_px=(float(xref_x), float(xref_y)),
        yref_px=(float(yref_x), float(yref_y)),
        x_axis_line=(ox, oy, xref_x, xref_y),
        y_axis_line=(ox, oy, yref_x, yref_y),
    )


def generate_curve_shape(
    x: np.ndarray,
    curve_type: str,
    curve_index: int,
    total_curves: int
) -> np.ndarray:
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
        return generate_curve_shape(x, random.choice(['peaked', 'rising', 'falling']),
                                   curve_index, total_curves)
    
    return y


def fig_to_array(fig, dpi=150, tight=True) -> np.ndarray:
    """Convert matplotlib figure to numpy array."""
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white', edgecolor='none')
    else:
        # Fixed size for mask consistency
        fig.savefig(buf, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def draw_chart_matplotlib(
    config: ChartConfig,
    W: int = 800,
    H: int = 600
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Draw chart using matplotlib for realistic arrows.
    
    Returns:
        full_img: RGB numpy array
        mask: Binary mask (curves only)
        curves_data: List of (x, y) for each curve
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
    
    # Draw prominent X and Y axis lines (thicker than grid, for detection)
    # X axis at bottom
    ax1.axhline(y=config.y_min, color='black', linewidth=2.0, zorder=10)
    # Y axis at left
    ax1.axvline(x=config.x_min, color='black', linewidth=2.0, zorder=10)
    
    # Add tick marks on axes
    ax1.tick_params(axis='both', which='major', length=6, width=1.5, direction='in')
    ax1.tick_params(axis='both', which='minor', length=3, width=1.0, direction='in')
    
    # Spine styling (frame)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    ax1.set_xlabel('MACH NUMBER', fontsize=10, fontweight='bold')
    ax1.set_ylabel('SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL', fontsize=8)
    
    # Draw curves (thin!)
    for cx, cy in curves_data:
        ax1.plot(cx, cy, 'k-', linewidth=config.curve_lw)
    
    # OPTIMUM CRUISE envelope
    if config.add_envelope_optimum:
        if config.curve_type == 'peaked':
            envelope_pts = [(cx[np.argmax(cy)], cy.max()) for cx, cy in curves_data]
        else:
            envelope_pts = [(cx[int(len(cx)*0.5)], cy[int(len(cy)*0.5)]) for cx, cy in curves_data]
        
        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2)
        
        ax1.text(ex[0] - 0.03, ey[-1] + (config.y_max - config.y_min) * 0.02,
                'OPTIMUM\nCRUISE', fontsize=8, ha='right', va='bottom')
    
    # MAXIMUM ENDURANCE envelope
    if config.add_envelope_endurance:
        envelope_pts = [(cx[int(len(cx)*0.2)], cy[int(len(cy)*0.2)]) for cx, cy in curves_data]
        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2)
        
        ax1.text(ex[-1] - 0.02, ey[0] - (config.y_max - config.y_min) * 0.02,
                'MAXIMUM\nENDURANCE', fontsize=8, ha='right', va='top')
    
    # Arrows with matplotlib annotate (like original code!)
    if config.add_arrows:
        fuel_flows = ['3000', '3500', '4000', '4500', '5000', '5500', 
                     '6000', '6500', '7000', '7500', '8000', '8500']
        
        for idx, (cx, cy) in enumerate(reversed(curves_data)):
            if idx >= len(fuel_flows):
                break
            
            # Randomly choose arrow position: endpoint OR middle of curve
            if random.random() < 0.5:
                # Arrow to curve endpoint (original behavior)
                arrow_idx = -1
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]
                # Arrow tail extends to the right
                dx = random.uniform(0.04, 0.08)
                dy = random.uniform(-0.005, 0.005)
                x_tail = x_head + dx
                y_tail = y_head + dy
            else:
                # Arrow to middle region of curve
                mid_start = len(cx) // 4
                mid_end = 3 * len(cx) // 4
                arrow_idx = random.randint(mid_start, mid_end)
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]
                # Arrow can come from various directions in middle
                angle = random.uniform(20, 70)  # degrees
                dist = random.uniform(0.05, 0.10)
                # Randomly from above-right or below-right
                if random.random() < 0.5:
                    dx = dist * math.cos(math.radians(angle))
                    dy = dist * math.sin(math.radians(angle))
                else:
                    dx = dist * math.cos(math.radians(-angle))
                    dy = dist * math.sin(math.radians(-angle))
                x_tail = x_head + dx
                y_tail = y_head + dy
            
            # Leader line + arrow head (closer to real charts)
            ax1.plot([x_tail, x_head], [y_tail, y_head], color="black", linewidth=0.6)
            
            # Randomize arrowhead: sometimes hollow (open), sometimes filled
            if random.random() < 0.4:
                # Hollow/open arrowhead style
                arrow_style = random.choice(["-|>", "->"])
                fill_style = "none"  # hollow
            else:
                # Filled arrowhead
                arrow_style = random.choice(["-|>", "-|>", "->"])
                fill_style = "black"
            
            ax1.annotate(
                "",
                xy=(x_head, y_head),  # Arrow tip
                xytext=(x_tail, y_tail),  # Arrow tail
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    lw=random.uniform(0.7, 1.1),
                    color="black",
                    fc=fill_style,  # fill color
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=random.uniform(12, 18),
                ),
            )
            
            # Dashed leader segment with fixed angle (35-75 degrees)
            # Placed more centrally on the chart
            if random.random() < 0.85:
                dash_len = random.uniform(0.06, 0.14)
                dash_angle = math.radians(random.choice([35, 45, 55, 65, 75]))
                dash_dx = dash_len * math.cos(dash_angle)
                dash_dy = dash_len * math.sin(dash_angle)
                # Place it more centrally - random position within chart area
                base_x = config.x_min + (config.x_max - config.x_min) * random.uniform(0.25, 0.75)
                base_y = config.y_min + (config.y_max - config.y_min) * random.uniform(0.25, 0.75)
                dash_start = (base_x, base_y)
                dash_end = (dash_start[0] + dash_dx, dash_start[1] + dash_dy)
                ax1.plot([dash_start[0], dash_end[0]], [dash_start[1], dash_end[1]],
                    color="black", linewidth=0.6, linestyle=(0, (14, 8)))
            
            # Fuel flow label (after dashed line)
            label_x = x_tail + random.uniform(0.06, 0.10)
            ax1.text(label_x, y_tail + random.uniform(-0.002, 0.002),
                    fuel_flows[idx], fontsize=8, va='center', ha='left')
    
    # Additional standalone dashed lines in central areas (not tied to arrows)
    if random.random() < 0.7:
        n_extra_dashes = random.randint(2, 6)
        for _ in range(n_extra_dashes):
            # Random central position
            cx = config.x_min + (config.x_max - config.x_min) * random.uniform(0.2, 0.8)
            cy = config.y_min + (config.y_max - config.y_min) * random.uniform(0.2, 0.8)
            dash_len = random.uniform(0.04, 0.10)
            dash_angle = math.radians(random.choice([35, 45, 55, 65, 75]))
            dash_dx = dash_len * math.cos(dash_angle)
            dash_dy = dash_len * math.sin(dash_angle)
            ax1.plot([cx, cx + dash_dx], [cy, cy + dash_dy],
                    color="black", linewidth=0.5, linestyle=(0, (12, 7)))
    
    # Text boxes
    if config.add_text_boxes:
        # TOTAL FUEL FLOW box
        ax1.text(
            config.x_max - 0.05, config.y_max - 0.005,
            'TOTAL FUEL FLOW—\nPOUNDS PER HOUR',
            fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black')
        )
        
        # Legend box (CRUISE/DASH AOA)
        legend_x = config.x_min + (config.x_max - config.x_min) * 0.15
        legend_y = config.y_max - (config.y_max - config.y_min) * 0.1
        
        ax1.text(
            legend_x, legend_y,
            '◄─ CRUISE    DASH ─►\n      AOA          AOA\n(USED FOR INTERFERENCE\n DRAG DETERMINATION)',
            fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black')
        )
    
    # Drag index labels
    if config.add_drag_labels:
        labels = ['0.00', '25.00', '50.00', '75.00', '100.00', '125.00', '150.00']
        base_x = config.x_min + (config.x_max - config.x_min) * 0.65
        base_y = config.y_min + (config.y_max - config.y_min) * 0.15
        
        for i, lbl in enumerate(labels[:random.randint(4, 7)]):
            ax1.text(base_x + random.uniform(-0.02, 0.02),
                    base_y + i * (config.y_max - config.y_min) * 0.05,
                    lbl, fontsize=7, alpha=0.9)
    
    # Vmax line (dashed)
    if config.add_vmax_line:
        vmax_pts = [(cx[int(len(cx)*0.85)], cy[int(len(cy)*0.85)]) for cx, cy in curves_data]
        vmax_pts.sort(key=lambda p: p[1])
        vx, vy = zip(*vmax_pts)
        ax1.plot(vx, vy, 'k--', linewidth=0.8)
        ax1.text(vx[-1], vy[-1] + 0.003, r'$V_{max}$(MIL)', fontsize=7)
    
    full_img = fig_to_array(fig1, dpi=150, tight=True)
    full_img = cv2.resize(full_img, (W, H))
    
    # ========== MASK (curves only) ==========
    # Use same figure size and limits for consistent mapping
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.set_xlim(config.x_min, config.x_max)
    ax2.set_ylim(config.y_min, config.y_max)
    ax2.set_position([0, 0, 1, 1])  # Fill entire figure
    ax2.axis('off')
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    
    # Draw curves slightly thicker on mask for better detection
    for cx, cy in curves_data:
        ax2.plot(cx, cy, 'w-', linewidth=config.curve_lw + 0.6)
    
    mask_img = fig_to_array(fig2, dpi=150, tight=False)
    mask_img = cv2.resize(mask_img, (W, H))
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask_gray, 20, 255, cv2.THRESH_BINARY)
    
    return full_img, mask, curves_data


def random_config() -> ChartConfig:
    """Generate random chart configuration."""
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
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        n_curves=random.randint(4, 12),
        curve_type=random.choice(['peaked', 'peaked', 'rising', 'falling', 'mixed']),
        curve_lw=random.uniform(0.3, 0.6),  # Random thickness (0.3=very thin, 0.6=normal)
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


def make_sample(W: int = 512, H: int = 512, seed: int = None,
                add_artifacts: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a training sample."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    config = random_config()
    full_img, mask, _ = draw_chart_matplotlib(config, W, H)
    
    if add_artifacts:
        full_img = add_scan_artifacts(full_img)
    
    # Convert RGB to BGR for OpenCV
    full_img_bgr = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
    
    return full_img_bgr, mask


def colorize_curves_from_data(
    curves_data: List[Tuple[np.ndarray, np.ndarray]],
    config: ChartConfig,
    W: int,
    H: int,
    show_axes: bool = True,
    axis_result: Optional[AxisDetectionResult] = None,
    full_img_rgb: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Render colored curves directly from curve data so each curve
    gets a distinct color even if components touch.
    Also draws X and Y axes for reference.
    """
    fig_w, fig_h = W / 100, H / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Draw X and Y axes
    if show_axes:
        # X axis (bottom)
        ax.axhline(y=config.y_min, color='black', linewidth=1.5, zorder=1)
        # Y axis (left)
        ax.axvline(x=config.x_min, color='black', linewidth=1.5, zorder=1)
        
        # Axis labels
        ax.text(config.x_max, config.y_min - (config.y_max - config.y_min) * 0.05,
               f'X: {config.x_min:.2f} - {config.x_max:.2f}',
               fontsize=9, ha='right', va='top', color='black')
        ax.text(config.x_min - (config.x_max - config.x_min) * 0.02, config.y_max,
               f'Y: {config.y_min:.2f} - {config.y_max:.2f}',
               fontsize=9, ha='right', va='top', color='black', rotation=90)

    n_curves = len(curves_data)
    for i, (cx, cy) in enumerate(curves_data):
        hue = int(180 * i / max(n_curves, 1))
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
        ax.plot(cx, cy, color=np.array(rgb_color) / 255.0, linewidth=config.curve_lw + 0.3, zorder=2)

    colored = fig_to_array(fig, dpi=150, tight=False)
    colored = cv2.resize(colored, (W, H))
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    # Auto-detect axes from the full image if requested
    if show_axes and axis_result is None and full_img_rgb is not None:
        axis_result = detect_axes_from_rgb(full_img_rgb)

    # Overlay detected axes on the colored output (pixel space)
    if show_axes and axis_result is not None:
        x1, y1, x2, y2 = axis_result.x_axis_line
        x3, y3, x4, y4 = axis_result.y_axis_line
        cv2.line(colored_bgr, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.line(colored_bgr, (x3, y3), (x4, y4), (0, 0, 0), 2)

        ox, oy = map(int, axis_result.origin_px)
        cv2.circle(colored_bgr, (ox, oy), 3, (0, 0, 0), -1)

        # Label axis endpoints
        cv2.putText(colored_bgr, "X", (x2 + 4, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(colored_bgr, "Y", (x4 + 4, y4 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return colored_bgr


if __name__ == "__main__":
    import os
    
    print("Generating V5 samples (matplotlib arrows, thin curves)...")
    
    # Test different types
    for ctype in ['peaked', 'rising', 'falling']:
        config = ChartConfig(
            x_min=0.35, x_max=1.15,
            y_min=0.06, y_max=0.16,
            n_curves=10,
            curve_type=ctype,
            curve_lw=random.uniform(0.3, 0.6),  # Random thickness
            add_envelope_optimum=True,
            add_envelope_endurance=(ctype == 'peaked'),
            add_vmax_line=(ctype == 'falling')
        )
        
        full, mask, curves = draw_chart_matplotlib(config, W=800, H=600)
        colored = colorize_curves_from_data(curves, config, W=800, H=600, show_axes=False)
        
        cv2.imwrite(f'v5_{ctype}.png', cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'v5_{ctype}_mask.png', mask)
        cv2.imwrite(f'v5_{ctype}_colored.png', colored)
        print(f"  ✓ v5_{ctype}.png")
    
    # Random samples
    print("\nRandom samples with artifacts:")
    for i in range(5):
        if i * 10 is not None:
            random.seed(i * 10)
            np.random.seed(i * 10)
        config = random_config()
        full, mask, curves = draw_chart_matplotlib(config, W=512, H=512)
        full = add_scan_artifacts(full)
        full_bgr = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
        colored = colorize_curves_from_data(curves, config, W=512, H=512, show_axes=False)
        cv2.imwrite(f'v5_random_{i}.png', full_bgr)
        cv2.imwrite(f'v5_random_{i}_mask.png', mask)
        cv2.imwrite(f'v5_random_{i}_colored.png', colored)
        print(f"  ✓ v5_random_{i}.png")
    
    print("\n✅ Done! Check v5_*.png files")
