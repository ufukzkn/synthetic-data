# quick_curve_test.py
"""Quick test of curve generation without matplotlib display."""

import numpy as np
import cv2
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_realistic_curve(x, altitude_index, total_curves):
    """Generate a realistic specific range curve."""
    alt_factor = altitude_index / (total_curves - 1) if total_curves > 1 else 0.5
    
    start_y = 0.012
    peak_y = 0.035 + alt_factor * 0.025
    peak_x = 0.38 - alt_factor * 0.03
    end_y = 0.018 + alt_factor * 0.015
    convergence_point = 0.18
    
    y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        if xi <= convergence_point:
            t = (xi - x[0]) / (convergence_point - x[0] + 1e-8)
            rise_target = start_y + (peak_y - start_y) * 0.3
            y[i] = start_y + (rise_target - start_y) * (t ** 0.8)
        elif xi <= peak_x:
            t = (xi - convergence_point) / (peak_x - convergence_point + 1e-8)
            y[i] = start_y + (peak_y - start_y) * 0.3 + \
                   (peak_y - (start_y + (peak_y - start_y) * 0.3)) * (1 - (1 - t) ** 2)
        else:
            t = (xi - peak_x) / (x[-1] - peak_x + 1e-8)
            y[i] = peak_y + (end_y - peak_y) * (t ** 0.7)
    
    return y


def draw_curves_opencv(n_curves=15, W=900, H=700):
    """Draw curves using OpenCV (no matplotlib)."""
    
    # Create white image
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Margins
    margin_left = 80
    margin_right = 50
    margin_top = 30
    margin_bottom = 50
    
    plot_w = W - margin_left - margin_right
    plot_h = H - margin_top - margin_bottom
    
    # Axis ranges
    x_min, x_max = 0.10, 1.00
    y_min, y_max = 0.01, 0.10
    
    def to_pixel(x_val, y_val):
        px = int(margin_left + (x_val - x_min) / (x_max - x_min) * plot_w)
        py = int(H - margin_bottom - (y_val - y_min) / (y_max - y_min) * plot_h)
        return px, py
    
    # Draw grid
    for x_val in np.arange(0.1, 1.01, 0.1):
        px, _ = to_pixel(x_val, y_min)
        cv2.line(img, (px, margin_top), (px, H - margin_bottom), (200, 200, 200), 1)
    
    for y_val in np.arange(0.01, 0.101, 0.01):
        _, py = to_pixel(x_min, y_val)
        cv2.line(img, (margin_left, py), (W - margin_right, py), (200, 200, 200), 1)
    
    # Draw border
    cv2.rectangle(img, (margin_left, margin_top), 
                  (W - margin_right, H - margin_bottom), (0, 0, 0), 2)
    
    # Generate and draw curves
    x = np.linspace(0.15, 0.92, 400)
    
    colors_rainbow = [
        (148, 0, 211),    # Violet
        (75, 0, 130),     # Indigo
        (0, 0, 255),      # Blue
        (0, 127, 255),    # Cyan-Blue
        (0, 255, 255),    # Cyan
        (0, 255, 127),    # Spring Green
        (0, 255, 0),      # Green
        (127, 255, 0),    # Chartreuse
        (255, 255, 0),    # Yellow
        (255, 200, 0),    # Gold
        (255, 127, 0),    # Orange
        (255, 0, 0),      # Red
        (255, 0, 127),    # Rose
        (200, 0, 200),    # Magenta
        (127, 0, 255),    # Purple
    ]
    
    for i in range(n_curves):
        y = generate_realistic_curve(x, i, n_curves)
        
        # Convert to pixel coordinates
        points = []
        for j in range(len(x)):
            px, py = to_pixel(x[j], y[j])
            points.append((px, py))
        
        # Draw on full image (black)
        for j in range(len(points) - 1):
            cv2.line(img, points[j], points[j+1], (0, 0, 0), 2, cv2.LINE_AA)
        
        # Draw on mask (white)
        for j in range(len(points) - 1):
            cv2.line(mask, points[j], points[j+1], 255, 2, cv2.LINE_AA)
    
    # Create colored output
    colored = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    for i in range(n_curves):
        y = generate_realistic_curve(x, i, n_curves)
        points = []
        for j in range(len(x)):
            px, py = to_pixel(x[j], y[j])
            points.append((px, py))
        
        color = colors_rainbow[i % len(colors_rainbow)]
        for j in range(len(points) - 1):
            cv2.line(colored, points[j], points[j+1], color, 2, cv2.LINE_AA)
    
    # Add axis labels (simplified)
    cv2.putText(img, 'MACH NUMBER', (W//2 - 60, H - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img, mask, colored


if __name__ == "__main__":
    print("Generating curves with OpenCV...")
    
    img, mask, colored = draw_curves_opencv(n_curves=15)
    
    cv2.imwrite('test_curves_input.png', img)
    cv2.imwrite('test_curves_mask.png', mask)
    cv2.imwrite('test_curves_colored.png', colored)
    
    print("Saved:")
    print("  - test_curves_input.png (simulated input)")
    print("  - test_curves_mask.png (target mask)")
    print("  - test_curves_colored.png (colored output)")
