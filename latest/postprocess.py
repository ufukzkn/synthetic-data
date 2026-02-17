"""
Postprocess — Arrow leader tracing + Tesseract OCR text detection.

UNet modeli sadece arrow tip + dashed lines algılar (2 kanal).
Bu modül:
  1. Arrow tip mask'tan geriye doğru leader çizgiyi trace eder
  2. Tesseract OCR ile text bölgelerini bulur
  3. Tüm mask'ları birleştirip inpaint mask oluşturur

Pipeline:
  UNet(arrows, dashed) → trace_arrow_leaders() → detect_text_ocr()
  → build_inpaint_mask() → cv2.inpaint()
"""

import cv2
import numpy as np
import math


# ════════════════════════════════════════════════════════════════
# ARROW LEADER LINE TRACING
# ════════════════════════════════════════════════════════════════

def trace_arrow_leaders(img_gray, arrow_mask, max_trace_len=300,
                        line_thickness=3, angle_tolerance=25):
    """
    Arrow tip mask'tan geriye doğru leader çizgiyi trace et.

    Mantık:
      1. Arrow mask'taki bağlı bileşenleri (connected components) bul
      2. Her bileşenin ana yönünü (orientation) hesapla
      3. Üçgenin sivri ucunu bul (ok yönünde en uçtaki piksel)
      4. Sivri uçtan TERS yöne doğru çizgiyi trace et
      5. Çizgi üzerinde yürürken img_gray'de karanlık pikseller varsa devam et

    Args:
        img_gray: Gri tonlama görüntü (uint8) — çizgi takibi için
        arrow_mask: Binary arrow tip mask (uint8, 0/255)
        max_trace_len: Maksimum trace uzunluğu (piksel)
        line_thickness: Trace edilen çizginin kalınlığı (mask genişleme)
        angle_tolerance: Çizgi takibinde açı toleransı (derece)

    Returns:
        leader_mask: Binary mask (uint8, 0/255) — sadece leader çizgiler
    """
    H, W = arrow_mask.shape[:2]
    leader_mask = np.zeros((H, W), dtype=np.uint8)

    # Bağlı bileşenleri bul
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        arrow_mask, connectivity=8)

    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 8:  # Çok küçük bileşenleri atla (gürültü)
            continue

        # Bu bileşenin piksellerini al
        pts = np.argwhere(labels == label_id)  # (row, col) = (y, x)
        if len(pts) < 5:
            continue

        # PCA ile ana yönü bul
        pts_float = pts.astype(np.float32)
        mean = pts_float.mean(axis=0)
        centered = pts_float - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # En büyük eigenvalue'nun eigenvector'ü = ana yön
        main_dir = eigenvectors[:, -1]  # (dy, dx)
        dy_main, dx_main = main_dir

        # Sivri ucu bul — ana yönde en uçtaki piksel
        projections = centered @ main_dir
        tip_idx = np.argmax(projections)  # Sivri uç
        tip_y, tip_x = pts[tip_idx]

        # Centroid'i kullan (argmin wing corner verir, centroid daha doğru)
        center_y, center_x = float(mean[0]), float(mean[1])

        # Ok yönü: centroid -> tip
        arrow_dy = tip_y - center_y
        arrow_dx = tip_x - center_x
        arrow_len = math.sqrt(arrow_dx**2 + arrow_dy**2)
        if arrow_len < 2:
            continue

        # Ters yön — leader çizgiyi trace etmek için (tip'ten uzağa)
        trace_dx = -arrow_dx / arrow_len
        trace_dy = -arrow_dy / arrow_len

        # Arrow mask dışına çıkana kadar centroid'den ilerle
        start_x, start_y = center_x, center_y
        for _ in range(50):
            nx = int(round(start_x + trace_dx))
            ny = int(round(start_y + trace_dy))
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                break
            if arrow_mask[ny, nx] == 0:
                break
            start_x += trace_dx
            start_y += trace_dy

        # Mask dışına çıkış noktasından trace et
        leader_pts = _trace_line(img_gray, start_x, start_y,
                                 trace_dx, trace_dy,
                                 max_trace_len, angle_tolerance)

        if len(leader_pts) > 5:
            # Leader çizgiyi mask'a çiz
            pts_arr = np.array(leader_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(leader_mask, [pts_arr], isClosed=False,
                          color=255, thickness=line_thickness)

    return leader_mask


def _trace_line(img_gray, start_x, start_y, dx, dy,
                max_len=300, angle_tolerance=25):
    """
    Bir noktadan belirli yönde çizgiyi trace et.

    Her adımda:
      - Ana yönde 1px ilerle
      - Yanlardan (±angle_tolerance) da bak — eğri çizgiler için
      - Karanlık piksel varsa devam et, yoksa dur

    Returns:
        list of (x, y) tuples — trace edilen noktalar
    """
    H, W = img_gray.shape[:2]
    points = [(int(start_x), int(start_y))]

    cx, cy = float(start_x), float(start_y)
    cur_dx, cur_dy = dx, dy

    dark_threshold = 180  # Bu değerin altındaki pikseller "çizgi"
    step_size = 1.5
    max_gap = 8  # Maksimum boşluk toleransı (grid removal sonrası kopuklar)
    gap_count = 0

    for _ in range(int(max_len / step_size)):
        # Ana yönde ilerle
        best_x, best_y = None, None
        best_darkness = 255

        # Birkaç farklı açıda dene
        for angle_offset in range(-angle_tolerance, angle_tolerance + 1, 5):
            rad = math.radians(angle_offset)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            test_dx = cur_dx * cos_a - cur_dy * sin_a
            test_dy = cur_dx * sin_a + cur_dy * cos_a

            test_x = cx + test_dx * step_size
            test_y = cy + test_dy * step_size

            ix, iy = int(round(test_x)), int(round(test_y))

            # Sınır kontrolü
            if ix < 1 or ix >= W - 1 or iy < 1 or iy >= H - 1:
                continue

            # 3x3 bölgede en karanlık pikseli kontrol et
            patch = img_gray[iy-1:iy+2, ix-1:ix+2]
            darkness = patch.min()

            if darkness < best_darkness:
                best_darkness = darkness
                best_x, best_y = test_x, test_y
                # Yönü güncelle (çizginin eğrisine uyum)
                new_dx, new_dy = test_dx, test_dy

        if best_x is None:
            break

        if best_darkness < dark_threshold:
            cx, cy = best_x, best_y
            points.append((int(round(cx)), int(round(cy))))
            gap_count = 0

            # Yönü hafifçe güncelle (smooth tracking)
            blend = 0.3
            cur_dx = cur_dx * (1 - blend) + new_dx * blend
            cur_dy = cur_dy * (1 - blend) + new_dy * blend
            norm = math.sqrt(cur_dx**2 + cur_dy**2)
            if norm > 0:
                cur_dx /= norm
                cur_dy /= norm
        else:
            # Boşluk — grid removal sonrası kopukluk olabilir
            gap_count += 1
            if gap_count > max_gap:
                break
            # Boşlukta da ilerle (ama mask'a ekleme)
            cx += cur_dx * step_size
            cy += cur_dy * step_size

    return points


# ════════════════════════════════════════════════════════════════
# TESSERACT OCR TEXT DETECTION
# ════════════════════════════════════════════════════════════════

def detect_text_ocr(img_gray, padding=3, min_conf=30):
    """
    Tesseract OCR ile text bölgelerini bul.

    Args:
        img_gray: Gri tonlama görüntü (uint8)
        padding: Her text bounding box'a eklenecek kenar boşluğu (px)
        min_conf: Minimum güven skoru (0-100)

    Returns:
        text_mask: Binary mask (uint8, 0/255) — text bölgeleri
        texts: list of dict — algılanan metinler [{text, x, y, w, h, conf}, ...]
    """
    H, W = img_gray.shape[:2]
    text_mask = np.zeros((H, W), dtype=np.uint8)
    texts = []

    try:
        import pytesseract
        # Windows varsayilan Tesseract yolu
        import os, shutil
        if os.name == 'nt' and not shutil.which('tesseract'):
            default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.isfile(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
    except ImportError:
        print('  [WARN] pytesseract not installed -- text detection disabled')
        print('         Install: pip install pytesseract')
        return text_mask, texts

    try:
        data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT,
                                         config='--psm 11 --oem 3')
    except Exception as e:
        print(f'  [WARN] Tesseract error: {e}')
        return text_mask, texts

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        conf = int(data['conf'][i])
        txt = str(data['text'][i]).strip()

        if conf < min_conf or len(txt) == 0:
            continue

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        # Padding ekle
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)

        text_mask[y1:y2, x1:x2] = 255
        texts.append({
            'text': txt, 'x': x, 'y': y,
            'w': w, 'h': h, 'conf': conf
        })

    return text_mask, texts


# ════════════════════════════════════════════════════════════════
# COMBINED MASK BUILDER
# ════════════════════════════════════════════════════════════════

def build_inpaint_mask(img_rgb, arrow_mask, dashed_mask,
                       trace_leaders=True, use_ocr=True,
                       max_trace_len=300, ocr_padding=3,
                       ocr_min_conf=30):
    """
    Tüm noise mask'larını birleştir.

    Args:
        img_rgb: RGB görüntü (uint8) — trace ve OCR için
        arrow_mask: Binary arrow tip mask (uint8, 0/255)
        dashed_mask: Binary dashed lines mask (uint8, 0/255)
        trace_leaders: Arrow leader line trace et
        use_ocr: Tesseract OCR ile text bul
        max_trace_len: Leader trace max uzunluk
        ocr_padding: OCR text box padding
        ocr_min_conf: OCR minimum güven skoru

    Returns:
        dict: {
            'combined': birleşik inpaint mask (uint8, 0/255),
            'arrows': arrow tip mask,
            'leaders': leader line mask,
            'dashed': dashed lines mask,
            'text': text mask,
            'ocr_texts': algılanan text listesi,
        }
    """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Leader line trace
    leader_mask = np.zeros_like(arrow_mask)
    if trace_leaders and arrow_mask.any():
        leader_mask = trace_arrow_leaders(
            img_gray, arrow_mask, max_trace_len=max_trace_len)

    # OCR text detection
    text_mask = np.zeros_like(arrow_mask)
    ocr_texts = []
    if use_ocr:
        text_mask, ocr_texts = detect_text_ocr(
            img_gray, padding=ocr_padding, min_conf=ocr_min_conf)

    # Birleştir
    combined = np.zeros_like(arrow_mask)
    combined = np.maximum(combined, arrow_mask)
    combined = np.maximum(combined, leader_mask)
    combined = np.maximum(combined, dashed_mask)
    combined = np.maximum(combined, text_mask)

    return {
        'combined': combined,
        'arrows': arrow_mask,
        'leaders': leader_mask,
        'dashed': dashed_mask,
        'text': text_mask,
        'ocr_texts': ocr_texts,
    }


# ════════════════════════════════════════════════════════════════
# TEST
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    print('=== Postprocess Module Test ===')

    # Test trace fonksiyonu synthetic data ile
    print('\n1. Arrow trace test...')
    test_img = np.ones((256, 256), dtype=np.uint8) * 255

    # Sahte ok çiz — üçgen uç + leader çizgi
    # Leader çizgi (ince)
    cv2.line(test_img, (50, 128), (180, 128), 0, 1)
    # Üçgen ok ucu
    pts = np.array([[180, 128], [165, 120], [165, 136]], dtype=np.int32)
    cv2.fillPoly(test_img, [pts], 0)

    # Arrow mask — sadece üçgen
    arrow_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.fillPoly(arrow_mask, [pts], 255)

    leader = trace_arrow_leaders(test_img, arrow_mask,
                                 max_trace_len=200, line_thickness=2)
    n_leader_px = (leader > 0).sum()
    print(f'  Arrow tip area: {(arrow_mask > 0).sum()} px')
    print(f'  Leader traced:  {n_leader_px} px')
    print(f'  Status: {"OK" if n_leader_px > 20 else "FAIL"}')

    # Test OCR
    print('\n2. OCR test...')
    try:
        import pytesseract
        ocr_test = np.ones((100, 300), dtype=np.uint8) * 255
        cv2.putText(ocr_test, '3000', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 2)
        text_mask, texts = detect_text_ocr(ocr_test)
        print(f'  Detected texts: {[t["text"] for t in texts]}')
        print(f'  Text mask pixels: {(text_mask > 0).sum()}')
        print(f'  Status: OK')
    except ImportError:
        print('  pytesseract not installed — skipping')
        print('  Install: pip install pytesseract')

    print('\nDone!')
