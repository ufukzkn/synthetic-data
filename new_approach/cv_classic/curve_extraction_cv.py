# curve_extraction_cv.py
"""
Robust curve extraction from engineering charts using classical CV.
Handles: grid removal, text removal, dashed line connection, curve separation.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ExtractionParams:
    """Parameters for curve extraction pipeline."""
    # Grid removal
    grid_h_kernel: int = 40        # Horizontal kernel for grid detection
    grid_v_kernel: int = 40        # Vertical kernel for grid detection
    
    # Dashed line connection
    connection_kernel_size: int = 5  # Kernel size to bridge gaps
    connection_iterations: int = 2   # Morphological closing iterations
    
    # Noise/text removal
    min_area_threshold: int = 500    # Min pixel area to keep component
    max_area_ratio: float = 0.4      # Max area ratio (remove huge blobs)
    
    # Curve refinement
    thin_iterations: int = 1         # Thinning iterations
    smoothing_kernel: int = 3        # Gaussian smoothing before final


def preprocess_for_skeletonization(
    binary_img: np.ndarray,
    connection_kernel_size: int = 5,
    min_area_threshold: int = 500,
    connection_iterations: int = 2
) -> np.ndarray:
    """
    Clean binary image before skeletonization.
    
    Args:
        binary_img: Binary image (numpy array, single channel, 0 or 255).
        connection_kernel_size: Kernel size to connect dashed lines (3-15).
        min_area_threshold: Minimum pixel area to keep a component (50-2000).
        connection_iterations: Number of morphological closing iterations.
    
    Returns:
        Cleaned binary image ready for skeletonization.
    """
    if binary_img is None or binary_img.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Ensure binary
    if len(binary_img.shape) == 3:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    
    # Step 1: Bridge Gaps (Connect Dashes)
    # Use elliptical kernel - better for connecting diagonal dashes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (connection_kernel_size, connection_kernel_size)
    )
    connected = cv2.morphologyEx(
        binary_img, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=connection_iterations
    )
    
    # Step 2: Filter Noise (Remove Text and Small Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )
    
    # Calculate total image area for ratio check
    total_area = binary_img.shape[0] * binary_img.shape[1]
    max_area = total_area * 0.4  # Components larger than 40% are likely borders/artifacts
    
    # Create clean mask
    clean_mask = np.zeros_like(binary_img)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Keep only components within size range
        if min_area_threshold < area < max_area:
            clean_mask[labels == i] = 255
    
    return clean_mask


def remove_grid_lines(
    gray_img: np.ndarray,
    h_kernel_size: int = 40,
    v_kernel_size: int = 40
) -> np.ndarray:
    """
    Remove horizontal and vertical grid lines using morphological operations.
    
    Args:
        gray_img: Grayscale image.
        h_kernel_size: Horizontal kernel length for detecting horizontal lines.
        v_kernel_size: Vertical kernel length for detecting vertical lines.
    
    Returns:
        Image with grid lines removed.
    """
    # Threshold to binary (invert so lines are white)
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # Combine grid lines
    grid = cv2.add(h_lines, v_lines)
    
    # Dilate grid slightly to ensure complete removal
    grid_dilated = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)
    
    # Remove grid from original binary
    no_grid = cv2.subtract(binary, grid_dilated)
    
    return no_grid


def remove_arrows_and_text(
    binary_img: np.ndarray,
    min_curve_length: int = 100,
    max_aspect_ratio: float = 10.0
) -> np.ndarray:
    """
    Remove arrows, text labels, and other non-curve elements.
    
    Uses shape analysis:
    - Arrows have specific aspect ratios
    - Text is usually small or has high aspect ratio
    - Curves are long and have moderate aspect ratio
    
    Args:
        binary_img: Binary image with curves and noise.
        min_curve_length: Minimum length (perimeter) for a curve component.
        max_aspect_ratio: Maximum aspect ratio for curve components.
    
    Returns:
        Binary image with only curve-like components.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    
    clean_mask = np.zeros_like(binary_img)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Skip very small components (text, dots)
        if area < 200:
            continue
        
        # Calculate aspect ratio
        aspect = max(w, h) / (min(w, h) + 1e-6)
        
        # Calculate perimeter
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            perimeter = cv2.arcLength(contours[0], closed=False)
            
            # Curves are long (high perimeter) relative to area
            # Compactness = perimeter^2 / area (high for curves, low for blobs)
            compactness = (perimeter ** 2) / (area + 1e-6)
            
            # Keep components that look like curves:
            # - Long perimeter OR
            # - High compactness (thin, elongated)
            if perimeter > min_curve_length or compactness > 50:
                # But not too square (arrows can be detected by convexity)
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / (hull_area + 1e-6)
                
                # Curves have low solidity (not filled), arrows have high
                if solidity < 0.5 or perimeter > min_curve_length * 2:
                    clean_mask[labels == i] = 255
    
    return clean_mask


def skeletonize(binary_img: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization (thinning).
    
    Args:
        binary_img: Binary image (0 or 255).
    
    Returns:
        Skeletonized image.
    """
    # Ensure binary
    binary = (binary_img > 127).astype(np.uint8)
    
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        
        if cv2.countNonZero(binary) == 0:
            break
    
    return skeleton * 255


def trace_curves(
    skeleton: np.ndarray,
    min_length: int = 50
) -> List[np.ndarray]:
    """
    Trace individual curves from skeleton image.
    
    Args:
        skeleton: Skeletonized binary image.
        min_length: Minimum number of points for a valid curve.
    
    Returns:
        List of curves, each as Nx2 array of (x, y) points.
    """
    # Find all contours
    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    
    curves = []
    for contour in contours:
        if len(contour) >= min_length:
            # Reshape from (N, 1, 2) to (N, 2)
            points = contour.reshape(-1, 2)
            curves.append(points)
    
    return curves


def order_curve_points(points: np.ndarray) -> np.ndarray:
    """
    Order curve points from left to right (by x coordinate).
    Handles curves that may have been traced in wrong direction.
    
    Args:
        points: Nx2 array of (x, y) points.
    
    Returns:
        Ordered Nx2 array.
    """
    if len(points) == 0:
        return points
    
    # Simple approach: sort by x, then connect nearest neighbors
    # For monotonic curves, sorting by x works well
    sorted_indices = np.argsort(points[:, 0])
    return points[sorted_indices]


def full_extraction_pipeline(
    image: np.ndarray,
    params: Optional[ExtractionParams] = None,
    debug: bool = False
) -> Tuple[List[np.ndarray], dict]:
    """
    Full curve extraction pipeline.
    
    Args:
        image: Input BGR or grayscale image.
        params: Extraction parameters.
        debug: If True, return intermediate images.
    
    Returns:
        Tuple of (list of curves, debug_images dict)
    """
    if params is None:
        params = ExtractionParams()
    
    debug_imgs = {}
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if debug:
        debug_imgs['input'] = gray.copy()
    
    # Step 1: Remove grid lines
    no_grid = remove_grid_lines(
        gray, 
        h_kernel_size=params.grid_h_kernel,
        v_kernel_size=params.grid_v_kernel
    )
    
    if debug:
        debug_imgs['no_grid'] = no_grid.copy()
    
    # Step 2: Connect dashed lines and remove small noise
    cleaned = preprocess_for_skeletonization(
        no_grid,
        connection_kernel_size=params.connection_kernel_size,
        min_area_threshold=params.min_area_threshold,
        connection_iterations=params.connection_iterations
    )
    
    if debug:
        debug_imgs['cleaned'] = cleaned.copy()
    
    # Step 3: Additional arrow/text removal
    curves_only = remove_arrows_and_text(
        cleaned,
        min_curve_length=100,
        max_aspect_ratio=8.0
    )
    
    if debug:
        debug_imgs['curves_only'] = curves_only.copy()
    
    # Step 4: Skeletonize
    skeleton = skeletonize(curves_only)
    
    if debug:
        debug_imgs['skeleton'] = skeleton.copy()
    
    # Step 5: Trace curves
    curves = trace_curves(skeleton, min_length=50)
    
    # Step 6: Order points
    ordered_curves = [order_curve_points(c) for c in curves]
    
    return ordered_curves, debug_imgs


def visualize_curves(
    image: np.ndarray,
    curves: List[np.ndarray],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize extracted curves with different colors.
    
    Args:
        image: Original image for background.
        curves: List of curve point arrays.
        save_path: Optional path to save visualization.
    
    Returns:
        Visualization image.
    """
    # Create white background
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Make background lighter
    vis = cv2.addWeighted(vis, 0.3, np.full_like(vis, 255), 0.7, 0)
    
    # Generate distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(curves)))
    colors = (colors[:, :3] * 255).astype(int)
    
    for i, curve in enumerate(curves):
        color = tuple(map(int, colors[i]))
        
        # Draw curve
        for j in range(len(curve) - 1):
            pt1 = tuple(curve[j].astype(int))
            pt2 = tuple(curve[j + 1].astype(int))
            cv2.line(vis, pt1, pt2, color, 2)
    
    if save_path:
        cv2.imwrite(save_path, vis)
    
    return vis


# ============================================================
# Interactive testing with parameter tuning
# ============================================================

def test_on_image(image_path: str, output_dir: str = "cv_output"):
    """
    Test the extraction pipeline on a single image.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Try different parameter combinations
    param_sets = [
        ExtractionParams(
            grid_h_kernel=40, grid_v_kernel=40,
            connection_kernel_size=5, min_area_threshold=300
        ),
        ExtractionParams(
            grid_h_kernel=50, grid_v_kernel=50,
            connection_kernel_size=7, min_area_threshold=500
        ),
        ExtractionParams(
            grid_h_kernel=30, grid_v_kernel=30,
            connection_kernel_size=3, min_area_threshold=200
        ),
    ]
    
    for i, params in enumerate(param_sets):
        print(f"\nTrying parameter set {i+1}...")
        
        curves, debug_imgs = full_extraction_pipeline(image, params, debug=True)
        
        print(f"  Found {len(curves)} curves")
        
        # Save debug images
        for name, img in debug_imgs.items():
            cv2.imwrite(f"{output_dir}/debug_{i+1}_{name}.png", img)
        
        # Save visualization
        vis = visualize_curves(image, curves)
        cv2.imwrite(f"{output_dir}/result_{i+1}.png", vis)
        
        # Save curves as CSV
        for j, curve in enumerate(curves):
            np.savetxt(
                f"{output_dir}/curve_{i+1}_{j:02d}.csv",
                curve,
                delimiter=",",
                header="x,y",
                comments=""
            )
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        print("Usage: python curve_extraction_cv.py <image_path>")
        print("\nTesting with a sample...")
        
        # Create a simple test image
        test_img = np.ones((400, 600), dtype=np.uint8) * 255
        
        # Draw a curve
        x = np.linspace(50, 550, 200)
        y = 200 + 100 * np.sin(x / 50)
        for i in range(len(x) - 1):
            cv2.line(
                test_img,
                (int(x[i]), int(y[i])),
                (int(x[i+1]), int(y[i+1])),
                0, 2
            )
        
        # Draw grid
        for gx in range(0, 600, 50):
            cv2.line(test_img, (gx, 0), (gx, 400), 128, 1)
        for gy in range(0, 400, 50):
            cv2.line(test_img, (0, gy), (600, gy), 128, 1)
        
        cv2.imwrite("test_input.png", test_img)
        
        # Run extraction
        curves, debug = full_extraction_pipeline(
            cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR),
            debug=True
        )
        
        print(f"Found {len(curves)} curves")
        
        for name, img in debug.items():
            cv2.imwrite(f"test_{name}.png", img)
