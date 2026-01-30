"""
Tek seferlik test scripti.
Kendi grafiğinle sentetik veriyi karşılaştır.

Kullanım:
    python test_single_image.py <senin_grafigin.png>
    
Veya direkt çalıştır (varsayılan test görüntüsü üretir):
    python test_single_image.py
"""

import sys
import os

# Add synthetic folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'new_approach', 'synthetic'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import from synthetic_v5
from synthetic_v5 import (
    ChartConfig, random_config, draw_chart_matplotlib, 
    add_scan_artifacts, fig_to_array
)


def load_image(path):
    """Load image as RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compare_with_synthetic(real_img, n_synthetic=3):
    """Compare real image with synthetic samples."""
    
    fig, axes = plt.subplots(2, n_synthetic + 1, figsize=(5 * (n_synthetic + 1), 10))
    
    # Real image
    axes[0, 0].imshow(real_img)
    axes[0, 0].set_title('GERÇEK GRAFİK', fontsize=14, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    # Edge detection on real
    gray = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Gerçek - Edge Detection', fontsize=10)
    axes[1, 0].axis('off')
    
    # Generate synthetic samples
    for i in range(n_synthetic):
        config = random_config()
        h, w = real_img.shape[:2]
        synth_img, mask, curves = draw_chart_matplotlib(config, W=w, H=h)
        synth_img = add_scan_artifacts(synth_img, strength=0.8)
        
        axes[0, i + 1].imshow(synth_img)
        axes[0, i + 1].set_title(f'SENTETİK #{i+1}\n({config.n_curves} curve, lw={config.curve_lw:.2f})', 
                                  fontsize=10, color='blue')
        axes[0, i + 1].axis('off')
        
        axes[1, i + 1].imshow(mask, cmap='gray')
        axes[1, i + 1].set_title(f'Mask #{i+1} (eğitim hedefi)', fontsize=10)
        axes[1, i + 1].axis('off')
    
    plt.suptitle('Gerçek Grafik vs Sentetik Örnekler', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_output.png', dpi=150, bbox_inches='tight')
    print("✅ Karşılaştırma kaydedildi: comparison_output.png")
    plt.show()


def analyze_real_image(real_img):
    """Analyze the real image - edge detection, line detection etc."""
    
    gray = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                            minLineLength=50, maxLineGap=10)
    
    # Draw lines on image
    line_img = real_img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Threshold to find dark pixels (potential curves)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(real_img)
    axes[0, 0].set_title('1. Orijinal Görüntü', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('2. Grayscale', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('3. Canny Edge Detection', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(line_img)
    axes[1, 0].set_title(f'4. Hough Lines ({len(lines) if lines is not None else 0} çizgi)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title('5. Threshold (koyu pikseller)', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cleaned, cmap='gray')
    axes[1, 2].set_title('6. Morfolojik temizlik', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle('Gerçek Grafik Analizi - Klasik CV Yöntemleri', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_output.png', dpi=150, bbox_inches='tight')
    print("✅ Analiz kaydedildi: analysis_output.png")
    plt.show()
    
    return edges, cleaned


def extract_curves_classic(real_img):
    """Attempt to extract curves using classic CV steps similar to synthetic pipeline."""
    gray = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)

    # Enhance dark strokes (curves) using blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to get dark strokes
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove long horizontal/vertical lines (grid/axes) via morphology
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(h_lines, v_lines)
    curves = cv2.bitwise_and(binary, cv2.bitwise_not(lines))

    # Clean up noise
    curves = cv2.morphologyEx(curves, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    curves = cv2.morphologyEx(curves, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return curves


def generate_synthetic_samples(n=5, output_dir='synthetic_samples'):
    """Generate multiple synthetic samples for inspection."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {n} synthetic samples...")
    
    for i in range(n):
        config = random_config()
        full_img, mask, curves = draw_chart_matplotlib(config, W=800, H=600)
        full_with_artifacts = add_scan_artifacts(full_img, strength=0.8)
        
        # Save
        cv2.imwrite(f'{output_dir}/sample_{i:02d}_clean.png', 
                    cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{output_dir}/sample_{i:02d}_artifact.png', 
                    cv2.cvtColor(full_with_artifacts, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{output_dir}/sample_{i:02d}_mask.png', mask)
        
        print(f"  ✓ Sample {i}: {config.n_curves} curves, type={config.curve_type}, lw={config.curve_lw:.2f}")
    
    print(f"\n✅ Samples saved to: {output_dir}/")


def main():
    print("=" * 60)
    print("  F-18 Chart Curve Extraction - Test Script")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # User provided an image
        image_path = sys.argv[1]
        print(f"\n📷 Loading: {image_path}")
        real_img = load_image(image_path)
        print(f"   Size: {real_img.shape[1]}x{real_img.shape[0]}")
        
        print("\n" + "-" * 40)
        print("1. Görüntü Analizi (Klasik CV)")
        print("-" * 40)
        analyze_real_image(real_img)

        print("\n" + "-" * 40)
        print("1b. Curve Extraction (Klasik CV Denemesi)")
        print("-" * 40)
        curves_mask = extract_curves_classic(real_img)
        cv2.imwrite('curves_extracted.png', curves_mask)
        print("✅ Curve mask kaydedildi: curves_extracted.png")
        plt.figure(figsize=(6, 6))
        plt.imshow(curves_mask, cmap='gray')
        plt.title('Extracted Curves (Classic CV)')
        plt.axis('off')
        plt.show()
        
        print("\n" + "-" * 40)
        print("2. Sentetik Veri Karşılaştırması")
        print("-" * 40)
        compare_with_synthetic(real_img, n_synthetic=3)
        
    else:
        print("\n⚠️  Görüntü belirtilmedi. Sentetik örnekler üretiliyor...")
        print("   Kullanım: python test_single_image.py <senin_grafigin.png>")
        
        # Generate synthetic samples for inspection
        generate_synthetic_samples(n=5)
        
        # Also show a comparison with a generated "fake real" image
        print("\n" + "-" * 40)
        print("Demo: Sentetik görüntü üzerinde test")
        print("-" * 40)
        
        config = ChartConfig(
            x_min=0.35, x_max=1.15,
            y_min=0.06, y_max=0.16,
            n_curves=10,
            curve_type='peaked',
            curve_lw=0.5
        )
        demo_img, mask, _ = draw_chart_matplotlib(config, W=800, H=600)
        demo_img = add_scan_artifacts(demo_img)
        
        analyze_real_image(demo_img)
        compare_with_synthetic(demo_img, n_synthetic=3)
    
    print("\n" + "=" * 60)
    print("  Test tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
