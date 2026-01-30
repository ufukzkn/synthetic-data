# test_cv_extraction.py
"""
Quick test of CV-based curve extraction.
Place your test image as 'test_chart.png' in the same directory.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from curve_extraction_cv import (
    ExtractionParams,
    full_extraction_pipeline,
    visualize_curves,
    remove_grid_lines,
    preprocess_for_skeletonization,
    remove_arrows_and_text,
    skeletonize
)


def test_step_by_step(image_path: str, output_dir: str = "cv_debug"):
    """
    Test each step individually and visualize.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✅ Loaded image: {gray.shape}")
    
    # Binary (inverted - curves are white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_dir}/01_binary.png", binary)
    print(f"   Binary threshold applied")
    
    # Step 1: Remove grid
    print("\n🔧 Step 1: Grid Removal")
    for kernel_size in [20, 30, 40, 50, 60]:
        no_grid = remove_grid_lines(gray, h_kernel_size=kernel_size, v_kernel_size=kernel_size)
        cv2.imwrite(f"{output_dir}/02_no_grid_k{kernel_size}.png", no_grid)
        white_pixels = np.sum(no_grid > 0)
        print(f"   Kernel={kernel_size}: {white_pixels} white pixels remaining")
    
    # Use best (40 as default)
    no_grid = remove_grid_lines(gray, h_kernel_size=40, v_kernel_size=40)
    
    # Step 2: Connect dashed lines
    print("\n🔧 Step 2: Connect Dashed Lines")
    for conn_k in [3, 5, 7, 9]:
        connected = preprocess_for_skeletonization(
            no_grid, 
            connection_kernel_size=conn_k,
            min_area_threshold=100,
            connection_iterations=2
        )
        cv2.imwrite(f"{output_dir}/03_connected_k{conn_k}.png", connected)
        num_components = cv2.connectedComponents(connected)[0] - 1
        print(f"   Kernel={conn_k}: {num_components} components")
    
    # Step 3: Remove arrows/text
    print("\n🔧 Step 3: Remove Arrows and Text")
    connected = preprocess_for_skeletonization(no_grid, connection_kernel_size=5, min_area_threshold=300)
    
    for min_length in [50, 100, 150, 200]:
        curves_only = remove_arrows_and_text(connected, min_curve_length=min_length)
        cv2.imwrite(f"{output_dir}/04_curves_only_l{min_length}.png", curves_only)
        num_components = cv2.connectedComponents(curves_only)[0] - 1
        print(f"   Min length={min_length}: {num_components} components")
    
    # Step 4: Skeletonize
    print("\n🔧 Step 4: Skeletonization")
    curves_only = remove_arrows_and_text(connected, min_curve_length=100)
    skeleton = skeletonize(curves_only)
    cv2.imwrite(f"{output_dir}/05_skeleton.png", skeleton)
    
    # Step 5: Full pipeline with different params
    print("\n🔧 Step 5: Full Pipeline")
    
    best_curves = []
    best_params = None
    
    for grid_k in [30, 40, 50]:
        for conn_k in [3, 5, 7]:
            for min_area in [200, 400, 600]:
                params = ExtractionParams(
                    grid_h_kernel=grid_k,
                    grid_v_kernel=grid_k,
                    connection_kernel_size=conn_k,
                    min_area_threshold=min_area
                )
                
                curves, _ = full_extraction_pipeline(img, params, debug=False)
                
                if 5 <= len(curves) <= 25:  # Reasonable number of curves
                    if len(curves) > len(best_curves):
                        best_curves = curves
                        best_params = params
                        print(f"   ✓ grid={grid_k}, conn={conn_k}, area={min_area}: {len(curves)} curves")
    
    if best_params:
        print(f"\n🏆 Best result: {len(best_curves)} curves")
        print(f"   Parameters: grid={best_params.grid_h_kernel}, "
              f"conn={best_params.connection_kernel_size}, "
              f"min_area={best_params.min_area_threshold}")
        
        # Save best result
        curves, debug_imgs = full_extraction_pipeline(img, best_params, debug=True)
        
        for name, debug_img in debug_imgs.items():
            cv2.imwrite(f"{output_dir}/best_{name}.png", debug_img)
        
        vis = visualize_curves(img, curves)
        cv2.imwrite(f"{output_dir}/best_result.png", vis)
        
        # Save separate colored image (like the example)
        white_bg = np.ones_like(img) * 255
        colors = plt.cm.rainbow(np.linspace(0, 1, len(curves)))
        colors = (colors[:, :3] * 255).astype(int)
        
        for i, curve in enumerate(curves):
            color = tuple(map(int, colors[i][::-1]))  # BGR
            for j in range(len(curve) - 1):
                pt1 = tuple(curve[j].astype(int))
                pt2 = tuple(curve[j + 1].astype(int))
                cv2.line(white_bg, pt1, pt2, color, 2)
        
        cv2.imwrite(f"{output_dir}/best_colored.png", white_bg)
        
        print(f"\n✅ Results saved to {output_dir}/")
    else:
        print("\n❌ Could not find good parameters automatically")
        print("   Try the interactive tuner: streamlit run interactive_tuner.py")


def create_synthetic_test():
    """Create a synthetic test image similar to the aircraft chart."""
    W, H = 800, 600
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    # Draw grid
    for x in range(0, W, 50):
        cv2.line(img, (x, 0), (x, H), (200, 200, 200), 1)
    for y in range(0, H, 50):
        cv2.line(img, (0, y), (W, y), (200, 200, 200), 1)
    
    # Draw curves (similar shape to aircraft performance)
    np.random.seed(42)
    num_curves = 12
    
    for i in range(num_curves):
        # Create a curve shape similar to specific range curves
        x = np.linspace(100, 700, 200)
        
        # Peak around x=250-350
        peak_x = 250 + i * 10
        base_y = 100 + i * 35
        
        # Shape: rises then falls
        y = base_y + 150 * np.exp(-((x - peak_x) / 200) ** 2) - 50 * (x / 700)
        y = np.clip(y, 50, H - 50)
        
        # Draw as slightly thick line
        for j in range(len(x) - 1):
            pt1 = (int(x[j]), int(y[j]))
            pt2 = (int(x[j + 1]), int(y[j + 1]))
            cv2.line(img, pt1, pt2, (0, 0, 0), 2)
    
    # Add some arrows (noise)
    for _ in range(15):
        x1 = np.random.randint(100, 700)
        y1 = np.random.randint(100, 500)
        dx = np.random.randint(-50, 50)
        dy = np.random.randint(-30, 30)
        cv2.arrowedLine(img, (x1, y1), (x1 + dx, y1 + dy), (0, 0, 0), 1)
    
    # Add text
    for _ in range(10):
        x = np.random.randint(100, 700)
        y = np.random.randint(100, 500)
        cv2.putText(img, f"{np.random.randint(1000, 9999)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.imwrite("synthetic_test.png", img)
    print("✅ Created synthetic_test.png")
    return "synthetic_test.png"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Check for common test image names
        test_names = ["test_chart.png", "chart.png", "input.png", "sample.png"]
        image_path = None
        
        for name in test_names:
            if Path(name).exists():
                image_path = name
                break
        
        if image_path is None:
            print("No test image found. Creating synthetic test image...")
            image_path = create_synthetic_test()
    
    test_step_by_step(image_path)
