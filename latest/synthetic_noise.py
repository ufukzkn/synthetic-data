"""
Synthetic Noise Data Generator — v2

Oklar ve kesik çizgiler cv2 ile piksel seviyesinde çizilir.
Metin hâlâ matplotlib ile (font kalitesi için).

Input:  Grid kaldırılmış grafik (eğriler + eksenler + gürültü elemanları)
Target: 3-kanallı gürültü mask'ı
        R = oklar (arrows)
        G = kesik çizgiler (dashed lines)
        B = metin (text)
"""

import io
import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
import cv2


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

@dataclass
class ChartConfig:
    """Grafik oluşturma ayarları."""
    x_min: float = 0.30
    x_max: float = 1.00
    y_min: float = 0.04
    y_max: float = 0.15

    n_curves: int = 8
    curve_type: str = 'peaked'
    curve_lw: float = 0.6

    # Noise element toggles
    add_arrows: bool = True
    add_dashed_lines: bool = True
    add_text_labels: bool = True
    add_text_boxes: bool = True


def random_config() -> ChartConfig:
    """Rastgele grafik konfigürasyonu üret."""
    x_ranges = [
        (0.30, 0.95), (0.30, 1.00), (0.40, 1.10), (0.50, 1.20),
        (0.50, 1.30), (0.50, 1.40), (0.60, 1.40),
    ]
    y_ranges = [
        (0.04, 0.15), (0.05, 0.15), (0.06, 0.17), (0.07, 0.18),
        (0.08, 0.19), (0.08, 0.20), (0.05, 0.14),
    ]
    x_min, x_max = random.choice(x_ranges)
    y_min, y_max = random.choice(y_ranges)

    curve_types = (
        ['peaked_oval'] * 30 + ['peaked'] * 30 +
        ['rising'] * 22 + ['falling'] * 18
    )
    curve_type = random.choice(curve_types)

    if curve_type == 'falling':
        n_curves = random.randint(4, 7)
    elif curve_type == 'rising':
        n_curves = random.randint(4, 7)
    else:  # peaked, peaked_oval
        n_curves = random.randint(4, 7)

    return ChartConfig(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        n_curves=n_curves,
        curve_type=curve_type,
        curve_lw=random.uniform(0.3, 0.6),
        add_arrows=random.random() < 0.85,
        add_dashed_lines=random.random() < 0.80,
        add_text_labels=random.random() < 0.80,
        add_text_boxes=random.random() < 0.70,
    )


# ════════════════════════════════════════════════════════════════
# CURVE SHAPES
# ════════════════════════════════════════════════════════════════

def generate_curve_shape(x: np.ndarray, curve_type: str,
                         idx: int, total: int) -> np.ndarray:
    alt = idx / max(total - 1, 1)
    t = (x - x.min()) / (x.max() - x.min() + 1e-8)

    if curve_type == 'peaked':
        pk = 0.30 + random.uniform(-0.05, 0.05)
        s, p, e = (0.10 + alt * 0.06 + random.uniform(-0.02, 0.02),
                    0.35 + alt * 0.45 + random.uniform(-0.03, 0.03),
                    0.15 + alt * 0.30 + random.uniform(-0.02, 0.02))
        y = np.where(t <= pk,
                     s + (p - s) * (1 - np.clip(1 - t / pk, 0, None) ** 2),
                     p - (p - e) * np.clip((t - pk) / (1 - pk), 0, None) ** 0.7)

    elif curve_type == 'peaked_oval':
        pk = 0.45 + random.uniform(-0.07, 0.07)
        s, p, e = (0.10 + alt * 0.06 + random.uniform(-0.02, 0.02),
                    0.35 + alt * 0.45 + random.uniform(-0.03, 0.03),
                    0.15 + alt * 0.30 + random.uniform(-0.02, 0.02))
        y = np.where(t <= pk,
                     s + (p - s) * np.clip(np.sin(np.clip(t / pk, 0, 1) * np.pi / 2), 0, None) ** 1.2,
                     e + (p - e) * np.clip(np.cos(np.clip((t - pk) / (1 - pk), 0, 1) * np.pi / 2), 0, None) ** 1.2)

    elif curve_type == 'rising':
        s = 0.08 + alt * 0.05 + random.uniform(-0.02, 0.02)
        e = 0.55 + alt * 0.30 + random.uniform(-0.03, 0.03)
        c = random.uniform(0.7, 1.3)
        y = s + (e - s) * (np.clip(t, 0, None) ** c)

    elif curve_type == 'falling':
        s = 0.65 + alt * 0.25 + random.uniform(-0.03, 0.03)
        e = 0.12 + alt * 0.10 + random.uniform(-0.02, 0.02)
        c = random.uniform(0.5, 1.0)
        y = s - (s - e) * (np.clip(t, 0, None) ** c)

    else:
        # Bilinmeyen tip — peaked olarak fallback
        return generate_curve_shape(x, 'peaked', idx, total)

    return np.nan_to_num(y, nan=0.5)


# ════════════════════════════════════════════════════════════════
# COORDINATE CONVERSION
# ════════════════════════════════════════════════════════════════

def _data_to_px(dx, dy, cfg, W, H):
    """Data koordinatlarını piksel koordinatlarına çevir."""
    px = int((dx - cfg.x_min) / (cfg.x_max - cfg.x_min) * W)
    py = int((1.0 - (dy - cfg.y_min) / (cfg.y_max - cfg.y_min)) * H)
    return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)


def _px_to_data(px, py, cfg, W, H):
    """Piksel koordinatlarını data koordinatlarına çevir."""
    dx = cfg.x_min + (px / W) * (cfg.x_max - cfg.x_min)
    dy = cfg.y_min + (1.0 - py / H) * (cfg.y_max - cfg.y_min)
    return dx, dy


# ════════════════════════════════════════════════════════════════
# MATPLOTLIB (sadece eğriler + eksenler + metin)
# ════════════════════════════════════════════════════════════════

DPI = 150


def _make_fig(cfg: ChartConfig, W: int, H: int):
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax


def _make_mask_fig(cfg: ChartConfig, W: int, H: int):
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    return fig, ax


def _fig_to_array(fig, dpi=DPI) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def _draw_curves(ax, curves_data, lw=0.6, color='black'):
    for cx, cy in curves_data:
        ax.plot(cx, cy, color=color, linewidth=lw)


def _draw_curves_cv2(img, curves_data, cfg, W, H, thickness=1, color=(0, 0, 0)):
    """
    cv2 polyline ile eğri çizimi — matplotlib'e alternatif.
    Daha ince, daha keskin çizgiler üretir.
    """
    for cx, cy in curves_data:
        pts = []
        for xi, yi in zip(cx, cy):
            px, py = _data_to_px(xi, yi, cfg, W, H)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
    return img


def _draw_axes_cv2(img, cfg, W, H, thickness=2, color=(0, 0, 0)):
    """cv2 ile eksen çizgileri."""
    # X ekseni (alt)
    x1, y1 = _data_to_px(cfg.x_min, cfg.y_min, cfg, W, H)
    x2, y2 = _data_to_px(cfg.x_max, cfg.y_min, cfg, W, H)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    # Y ekseni (sol)
    x3, y3 = _data_to_px(cfg.x_min, cfg.y_min, cfg, W, H)
    x4, y4 = _data_to_px(cfg.x_min, cfg.y_max, cfg, W, H)
    cv2.line(img, (x3, y3), (x4, y4), color, thickness, cv2.LINE_AA)
    return img


def _draw_axes_lines(ax, cfg: ChartConfig):
    ax.axhline(y=cfg.y_min, color='black', linewidth=2.0, zorder=10)
    ax.axvline(x=cfg.x_min, color='black', linewidth=2.0, zorder=10)


# ════════════════════════════════════════════════════════════════
# CV2 İLE OK ÇİZİMİ (gerçekçi teknik çizim stili)
# ════════════════════════════════════════════════════════════════

def _generate_arrow_specs(cfg, curves_data):
    """
    Ok pozisyonlarını hesapla (data koordinatları).

    Gerçek F-18 grafik deseni:
      - Her okun kuyruğu (tail) bir text etiketinin yanında
      - Ok ucu (head) eğrinin sağ ucuna veya ortasına gidiyor
      - Bazı oklar eğrileri ortadan kesebilir
      - Oklar her zaman paralel değil, açıları değişken
      - Her okun yanında text var ama her text'in yanında ok yok

    Döndürür: list of (x_tail, y_tail, x_head, y_head, line_thickness,
                       tip_len, is_hollow, has_label)
    has_label=True ise bu ok'un kuyruğunda bir text etiketi olacak.
    """
    specs = []

    for idx, (cx, cy) in enumerate(reversed(curves_data)):
        if idx >= 12:
            break
        if random.random() < 0.15:
            continue

        # ── Ok ucu hedef noktası ──
        # %65 sağ uca (eğrinin bitişi), %35 ortaya
        if random.random() < 0.65:
            # Sağ uç — son %10-%25 arasından
            aidx = random.randint(int(len(cx) * 0.75), len(cx) - 1)
        else:
            # Orta bölge — eğriyi kesebilir
            aidx = random.randint(len(cx) // 5, int(len(cx) * 0.70))

        x_head, y_head = cx[aidx], cy[aidx]

        # ── Ok kuyruğu — text etiketinin olacağı yer ──
        # Kuyruk her zaman eğrinin dışında, genelde sağ tarafta/yukarıda
        angle = random.uniform(15, 80)
        dist = random.uniform(0.04, 0.12)
        # Genelde kuyruk sağa ve yukarıya doğru (text label bölgesi)
        dx_sign = random.choice([1, 1, 1, -1])  # %75 sağa
        dy_sign = random.choice([1, -1])         # eşit yukarı/aşağı
        dx = dist * math.cos(math.radians(angle)) * dx_sign
        dy = dist * math.sin(math.radians(angle)) * dy_sign
        x_tail = x_head + dx
        y_tail = y_head + dy

        thickness = 1
        tip_len = random.randint(12, 24)
        is_hollow = random.random() < 0.45

        # Her ok'un label'ı var (metnin ucunda ok)
        has_label = True

        specs.append((x_tail, y_tail, x_head, y_head, thickness,
                      tip_len, is_hollow, has_label))

    # ── Ekstra bağımsız oklar (label'sız olabilir) ──
    n_extra = random.randint(0, 2)
    for _ in range(n_extra):
        # Rastgele bir eğriye doğru
        ci = random.randint(0, len(curves_data) - 1)
        cx, cy = curves_data[ci]
        aidx = random.randint(len(cx) // 4, len(cx) - 1)
        xh, yh = cx[aidx], cy[aidx]
        ang = random.uniform(15, 80)
        d = random.uniform(0.04, 0.10)
        xt = xh + d * math.cos(math.radians(ang)) * random.choice([1, -1])
        yt = yh + d * math.sin(math.radians(ang)) * random.choice([1, -1])
        tip_len = random.randint(12, 22)
        is_hollow = random.random() < 0.45
        # Ekstra okların %50'sinde label yok
        has_label = random.random() < 0.50
        specs.append((xt, yt, xh, yh, 1, tip_len, is_hollow, has_label))

    return specs


def _draw_arrow_cv2(img, pt_tail, pt_head, thickness=1,
                    tip_length=18, color=(0, 0, 0), hollow=False):
    """
    Gerçek F-18 grafiklerindeki ok ucu stili:
    Uzun, dar, sivri konik üçgen.
    Bazıları içi dolu (filled), bazıları sadece kenar çizgili (hollow).
    """
    x1, y1 = int(pt_tail[0]), int(pt_tail[1])
    x2, y2 = int(pt_head[0]), int(pt_head[1])

    # Leader çizgisi (ince, düz)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return

    ux = dx / length
    uy = dy / length

    base_x = x2 - ux * tip_length
    base_y = y2 - uy * tip_length

    wing = tip_length * random.uniform(0.18, 0.24)

    nx = -uy
    ny = ux

    tip = (x2, y2)
    left = (int(base_x + nx * wing), int(base_y + ny * wing))
    right = (int(base_x - nx * wing), int(base_y - ny * wing))

    pts = np.array([tip, left, right], dtype=np.int32)

    if hollow:
        cv2.polylines(img, [pts], isClosed=True, color=color,
                      thickness=1, lineType=cv2.LINE_AA)
    else:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def _draw_arrow_pil(img, pt_tail, pt_head, thickness=1,
                    tip_length=18, color=(0, 0, 0), hollow=False):
    """
    PIL ile ok çizimi — alternatif render.
    Daha yumuşak anti-aliasing verir.
    """
    from PIL import ImageDraw
    H, W = img.shape[:2]
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    x1, y1 = int(pt_tail[0]), int(pt_tail[1])
    x2, y2 = int(pt_head[0]), int(pt_head[1])

    # Leader çizgisi
    pil_color = color if len(color) == 3 else (color[0], color[1], color[2])
    draw.line([(x1, y1), (x2, y2)], fill=pil_color, width=thickness)

    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        result = np.array(pil_img)
        np.copyto(img, result)
        return

    ux = dx / length
    uy = dy / length
    base_x = x2 - ux * tip_length
    base_y = y2 - uy * tip_length
    wing = tip_length * random.uniform(0.18, 0.24)
    nx, ny = -uy, ux

    tip = (x2, y2)
    left = (int(base_x + nx * wing), int(base_y + ny * wing))
    right = (int(base_x - nx * wing), int(base_y - ny * wing))

    if hollow:
        draw.polygon([tip, left, right], outline=pil_color, fill=None)
    else:
        draw.polygon([tip, left, right], outline=pil_color, fill=pil_color)

    np.copyto(img, np.array(pil_img))


def _draw_arrows_on_img(img, arrow_specs, cfg, W, H, color=(0, 0, 0),
                        use_pil=None):
    """
    Arrow spec'lerini bir görüntüye çiz.
    use_pil=None → rastgele seç, True → PIL, False → cv2
    """
    if use_pil is None:
        use_pil = random.random() < 0.4  # %40 PIL, %60 cv2

    draw_fn = _draw_arrow_pil if use_pil else _draw_arrow_cv2

    for spec in arrow_specs:
        # Unpack — has_label dahil (ama çizimde kullanılmaz)
        xt, yt, xh, yh, thick, tip_len, is_hollow = spec[:7]
        pt_tail = _data_to_px(xt, yt, cfg, W, H)
        pt_head = _data_to_px(xh, yh, cfg, W, H)
        draw_fn(img, pt_tail, pt_head, thickness=thick,
                tip_length=tip_len, color=color, hollow=is_hollow)
    return img


# ════════════════════════════════════════════════════════════════
# CV2 İLE KESİK ÇİZGİ (düz çizgiler)
# ════════════════════════════════════════════════════════════════

def _generate_dashed_specs(cfg, n_lines=None, W=512, H=512, arrow_specs=None):
    """
    Kesik çizgi pozisyonlarını hesapla.
    Tüm hesap piksel uzayında yapılır → görsel olarak gerçekten paralel.
    Üç mod:
      1) Ok ucundan çıkan — okun kuyruk ucundan aynı doğrultuda uzanan kesikli
      2) Sürü (herd) — aynı açı, aynı uzunluk, paralel
      3) Tekil — bağımsız uzun kesikli çizgi
    Tire = boşluk = eşit uzunluk.
    """
    specs = []

    # ── Ok ucundan (sivri uç) çıkan kesikli çizgiler ──
    if arrow_specs:
        for spec in arrow_specs:
            xt, yt, xh, yh, thick, tip_len, is_hollow = spec[:7]
            if random.random() < 0.45:  # ~%45 ihtimalle ok ucuna kesikli ekle
                # Ok yön vektörü: tail→head (okun işaret ettiği yön)
                pt_tail = _data_to_px(xt, yt, cfg, W, H)
                pt_head = _data_to_px(xh, yh, cfg, W, H)
                dx = pt_head[0] - pt_tail[0]  # tail'den head'e (ok yönü)
                dy = pt_head[1] - pt_tail[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1:
                    continue
                ux = dx / length
                uy = dy / length

                # Kesikli çizgi: ok ucundan (head) aynı doğrultuda devam eder
                seg = random.randint(10, 18)
                n_dashes = random.randint(4, 8)
                dash_len_px = n_dashes * 2 * seg

                # Başlangıç: ok ucu (head), bitiş: ileriye doğru
                sx = pt_head[0]
                sy = pt_head[1]
                ex = sx + ux * dash_len_px
                ey = sy + uy * dash_len_px

                d_sx, d_sy = _px_to_data(sx, sy, cfg, W, H)
                d_ex, d_ey = _px_to_data(ex, ey, cfg, W, H)
                specs.append((d_sx, d_sy, d_ex, d_ey, seg, seg, 1))

    # ── Sürü grubu (herd) — en fazla 1 küme, piksel uzayında ──
    if random.random() < 0.65:  # %65 ihtimalle 1 küme var
        cx_px = random.uniform(W * 0.25, W * 0.75)
        cy_px = random.uniform(H * 0.25, H * 0.75)

        angle_deg = random.choice([25, 30, 40, 45, 50, 55, 60, 65, 75])
        angle = math.radians(angle_deg)

        dir_x = math.cos(angle)
        dir_y = -math.sin(angle)

        perp_x = -dir_y
        perp_y = dir_x

        herd_size = random.randint(5, 15)
        spacing_px = random.uniform(10, 22)

        seg = random.randint(6, 22)
        # Dash:gap oranı — çeşitlendirilmiş
        ratio_choices = [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2)]
        d_ratio, g_ratio = random.choice(ratio_choices)
        gap = max(3, int(seg * g_ratio / d_ratio))
        herd_thick = random.choice([1, 1, 1, 2])  # çoğunlukla 1, bazen 2

        # Her çizgi için rastgele tire sayısı (min 6, max 10)
        # 2-3 farklı uzunluk grubu oluştur
        n_length_groups = random.randint(2, 3)
        length_options = [random.randint(6, 10) for _ in range(n_length_groups)]

        for j in range(herd_size):
            offset = (j - (herd_size - 1) / 2.0) * spacing_px
            mx = cx_px + perp_x * offset
            my = cy_px + perp_y * offset

            # Bu çizginin tire sayısı — rastgele bir uzunluk grubundan
            n_dashes_j = random.choice(length_options)
            line_len_px = n_dashes_j * (seg + gap)

            half = line_len_px / 2
            sx = mx - dir_x * half
            sy = my - dir_y * half
            ex = mx + dir_x * half
            ey = my + dir_y * half
            d_sx, d_sy = _px_to_data(sx, sy, cfg, W, H)
            d_ex, d_ey = _px_to_data(ex, ey, cfg, W, H)
            specs.append((d_sx, d_sy, d_ex, d_ey, seg, gap, herd_thick))

    # ── Tekil çizgiler — uzun, düzgün kesikli çizgi ──
    # Kesişme kontrolü: yeni çizgi mevcut çizgilerle kesişiyorsa atla
    n_singles = random.randint(3, 8)
    for _ in range(n_singles):
        cx_px = random.uniform(W * 0.15, W * 0.85)
        cy_px = random.uniform(H * 0.15, H * 0.85)
        seg = random.randint(6, 22)
        # Dash:gap oranı çeşitlendirilmiş
        ratio_choices = [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2)]
        d_ratio, g_ratio = random.choice(ratio_choices)
        gap = max(3, int(seg * g_ratio / d_ratio))
        thick = random.choice([1, 1, 1, 2])
        n_dashes = random.randint(4, 10)
        line_len_px = n_dashes * (seg + gap)
        angle_deg = random.choice([0, 20, 30, 45, 55, 65, 75, 90])
        angle = math.radians(angle_deg)
        dir_x = math.cos(angle)
        dir_y = -math.sin(angle)
        half = line_len_px / 2
        sx = cx_px - dir_x * half
        sy = cy_px - dir_y * half
        ex = cx_px + dir_x * half
        ey = cy_px + dir_y * half
        d_sx, d_sy = _px_to_data(sx, sy, cfg, W, H)
        d_ex, d_ey = _px_to_data(ex, ey, cfg, W, H)

        # Kesişme kontrolü — mevcut çizgilerle kes. varsa atla
        new_seg = ((sx, sy), (ex, ey))
        intersects = False
        for (os_dx, os_dy, oe_dx, oe_dy, *_rest) in specs:
            os_px = _data_to_px(os_dx, os_dy, cfg, W, H)
            oe_px = _data_to_px(oe_dx, oe_dy, cfg, W, H)
            if _segments_intersect(new_seg[0], new_seg[1], os_px, oe_px):
                intersects = True
                break
        if intersects:
            continue

        specs.append((d_sx, d_sy, d_ex, d_ey, seg, gap, thick))

    return specs


def _segments_intersect(p1, p2, p3, p4):
    """İki doğru parçasının kesişip kesişmediğini kontrol et (CCW yöntemi)."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
        return True
    return False


def _draw_dashed_line_cv2(img, pt1, pt2, dash_len, gap_len, thickness=1,
                           color=(0, 0, 0)):
    """
    Piksel seviyesinde düz kesikli çizgi çiz.
    Hiçbir bombelik/eğim yok — tam doğrusal.
    Float hassasiyetiyle hesaplanarak zoom artifact’ları önlenir.
    """
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 2:
        return

    ux = dx / length
    uy = dy / length

    dist = 0.0
    drawing = True
    while dist < length:
        seg_len = dash_len if drawing else gap_len
        seg_end = min(dist + seg_len, length)

        if drawing:
            sx = round(x1 + ux * dist)
            sy = round(y1 + uy * dist)
            ex = round(x1 + ux * seg_end)
            ey = round(y1 + uy * seg_end)
            # Segment en az 1px olmalı
            if abs(sx - ex) + abs(sy - ey) >= 1:
                cv2.line(img, (sx, sy), (ex, ey), color, thickness,
                         cv2.LINE_AA)

        dist = seg_end
        drawing = not drawing


def _draw_dashed_on_img(img, dashed_specs, cfg, W, H, color=(0, 0, 0)):
    """Dashed spec'lerini bir görüntüye çiz."""
    for (x1, y1, x2, y2, dash_px, gap_px, thick) in dashed_specs:
        pt1 = _data_to_px(x1, y1, cfg, W, H)
        pt2 = _data_to_px(x2, y2, cfg, W, H)
        _draw_dashed_line_cv2(img, pt1, pt2, dash_px, gap_px,
                               thickness=thick, color=color)
    return img


# ════════════════════════════════════════════════════════════════
# METİN (matplotlib ile — font kalitesi önemli)
# ════════════════════════════════════════════════════════════════

def _draw_text_labels(ax, cfg: ChartConfig, arrow_specs=None, color='black'):
    """Metin etiketleri — yakıt akışı sayıları, eksen başlıkları.

    Gerçek F-18 deseni: Her okun kuyruğunda text olabilir.
    has_label=True olan oklara metin eklenir.
    Ayrıca oksuz bağımsız metinler de olabilir.
    """
    fuel_flows = ['3000', '3500', '4000', '4500', '5000', '5500',
                  '6000', '6500', '7000', '7500', '8000', '8500',
                  '9000', '9500', '10000']
    label_idx = 0
    if arrow_specs:
        for spec in arrow_specs:
            xt, yt = spec[0], spec[1]
            has_label = spec[7] if len(spec) > 7 else True
            if not has_label:
                continue
            if label_idx >= len(fuel_flows):
                break
            # Metin okun kuyruğunun hemen yanında
            lx = xt + random.uniform(0.005, 0.025)
            ax.text(lx, yt + random.uniform(-0.003, 0.003),
                    fuel_flows[label_idx], fontsize=random.randint(7, 9),
                    va='center', ha='left', color=color)
            label_idx += 1

    # Oksuz bağımsız metinler (%40)
    n_extra_texts = random.randint(0, 3)
    for _ in range(n_extra_texts):
        if label_idx >= len(fuel_flows):
            break
        tx = cfg.x_min + (cfg.x_max - cfg.x_min) * random.uniform(0.5, 0.92)
        ty = cfg.y_min + (cfg.y_max - cfg.y_min) * random.uniform(0.1, 0.9)
        ax.text(tx, ty, fuel_flows[label_idx],
                fontsize=random.randint(7, 9),
                va='center', ha='left', color=color)
        label_idx += 1

    if random.random() < 0.7:
        ax.text((cfg.x_min + cfg.x_max) / 2,
                cfg.y_min - (cfg.y_max - cfg.y_min) * 0.06,
                'MACH NUMBER', fontsize=random.randint(8, 11),
                ha='center', va='top', fontweight='bold', color=color)
    if random.random() < 0.7:
        ax.text(cfg.x_min - (cfg.x_max - cfg.x_min) * 0.08,
                (cfg.y_min + cfg.y_max) / 2,
                'SPECIFIC RANGE', fontsize=random.randint(7, 9),
                ha='center', va='center', rotation=90, color=color)


def _draw_text_boxes(ax, cfg: ChartConfig, color='black'):
    """Metin kutuları."""
    box_texts = [
        'TOTAL FUEL FLOW—\nPOUNDS PER HOUR',
        'OPTIMUM\nCRUISE',
        '◄─ CRUISE    DASH ─►\n      AOA          AOA',
        'MAXIMUM\nENDURANCE',
        'DRAG INDEX\n(CONFIGURATION)',
    ]
    n_boxes = random.randint(1, 3)
    chosen = random.sample(box_texts, min(n_boxes, len(box_texts)))

    positions = [
        (cfg.x_max - 0.05, cfg.y_max - 0.005, 'right', 'top'),
        (cfg.x_min + 0.05, cfg.y_max - 0.005, 'left', 'top'),
        (cfg.x_max - 0.05, cfg.y_min + 0.005, 'right', 'bottom'),
        (cfg.x_min + 0.05, cfg.y_min + 0.005, 'left', 'bottom'),
        ((cfg.x_min + cfg.x_max) / 2, cfg.y_max - 0.01, 'center', 'top'),
    ]
    random.shuffle(positions)

    for i, txt in enumerate(chosen):
        px, py, ha, va = positions[i % len(positions)]
        has_box = random.random() < 0.6
        bbox_props = dict(boxstyle='square,pad=0.3',
                          facecolor='white', edgecolor=color) if has_box else None
        ax.text(px, py, txt,
                fontsize=random.randint(6, 9), ha=ha, va=va,
                color=color, bbox=bbox_props)

    # Drag index numaraları
    if random.random() < 0.50:
        labels = ['0.00', '25.00', '50.00', '75.00', '100.00']
        bx = cfg.x_min + (cfg.x_max - cfg.x_min) * random.uniform(0.55, 0.75)
        by = cfg.y_min + (cfg.y_max - cfg.y_min) * 0.15
        for j, lbl in enumerate(labels[:random.randint(3, 5)]):
            ax.text(bx + random.uniform(-0.02, 0.02),
                    by + j * (cfg.y_max - cfg.y_min) * 0.05,
                    lbl, fontsize=7, alpha=0.9, color=color)


# ════════════════════════════════════════════════════════════════
# SCAN ARTIFACTS
# ════════════════════════════════════════════════════════════════

def add_scan_artifacts(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    pil = Image.fromarray(img)
    angle = random.uniform(-1.2, 1.2) * strength
    pil = pil.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.90, 1.10))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.88, 1.12))

    arr = np.array(pil).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.012 * strength, arr.shape)
    arr = np.clip(arr + noise, 0, 1)

    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(
        buf, format='JPEG', quality=random.randint(50, 80))
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


# ════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ════════════════════════════════════════════════════════════════

def make_sample(W: int = 512, H: int = 512, seed: int = None,
                add_artifacts: bool = True,
                force_curve_type: str = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eğitim örneği üret.

    Mimari:
      1) Eğriler + eksenler → matplotlib (base image)
      2) Oklar + kesikli çizgiler → cv2 ile piksel seviyesinde
      3) Metin → matplotlib (font kalitesi)

    Aynı pozisyon verileri hem input'a hem mask'a uygulanır → tutarlı eşleşme.

    Returns:
        input_img:  RGB (uint8) — gürültülü grafik
        noise_mask: RGB (uint8) — R=oklar, G=kesik çizgiler, B=metin
        clean_img:  RGB (uint8) — sadece eğriler + eksenler
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cfg = random_config()
    if force_curve_type is not None:
        cfg = ChartConfig(
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            n_curves=cfg.n_curves,
            curve_type=force_curve_type,
            curve_lw=cfg.curve_lw,
            add_arrows=cfg.add_arrows,
            add_dashed_lines=cfg.add_dashed_lines,
            add_text_labels=cfg.add_text_labels,
            add_text_boxes=cfg.add_text_boxes,
        )

    # ── 1. Eğri verileri ──
    x = np.linspace(cfg.x_min + 0.02, cfg.x_max - 0.02, 400)
    curves_data = []
    for i in range(cfg.n_curves):
        y_norm = generate_curve_shape(x, cfg.curve_type, i, cfg.n_curves)
        y = cfg.y_min + y_norm * (cfg.y_max - cfg.y_min)
        y = np.clip(y, cfg.y_min + 0.008, cfg.y_max - 0.008)
        curves_data.append((x.copy(), y))

    # ── 2. Base image (eğriler + eksenler) ──
    # Rastgele: matplotlib VEYA cv2 ile çiz
    use_cv2_curves = random.random() < 0.5
    if use_cv2_curves:
        base_img = np.full((H, W, 3), 255, dtype=np.uint8)
        _draw_curves_cv2(base_img, curves_data, cfg, W, H,
                         thickness=max(1, round(cfg.curve_lw * 1.5)),
                         color=(0, 0, 0))
        _draw_axes_cv2(base_img, cfg, W, H, thickness=2, color=(0, 0, 0))
    else:
        fig_base, ax_base = _make_fig(cfg, W, H)
        _draw_curves(ax_base, curves_data, lw=cfg.curve_lw)
        _draw_axes_lines(ax_base, cfg)
        base_img = _fig_to_array(fig_base)
        base_img = cv2.resize(base_img, (W, H))

    # clean = base (gürültüsüz)
    clean_img = base_img.copy()

    # ── 3. Gürültü pozisyonlarını hesapla (seed'li) ──
    arrow_specs = _generate_arrow_specs(cfg, curves_data) if cfg.add_arrows else []
    dashed_specs = _generate_dashed_specs(cfg, W=W, H=H,
                                          arrow_specs=arrow_specs) if cfg.add_dashed_lines else []

    # ── 4. Input görüntüsü: base + oklar + kesikler + metin ──
    input_img = base_img.copy()

    # Oklar — PIL mi cv2 mi rastgele seç, ikisi de aynı yöntemi kullansın
    arrow_use_pil = random.random() < 0.4
    if arrow_specs:
        _draw_arrows_on_img(input_img, arrow_specs, cfg, W, H,
                            color=(0, 0, 0), use_pil=arrow_use_pil)

    # cv2 ile düz kesik çizgiler
    if dashed_specs:
        _draw_dashed_on_img(input_img, dashed_specs, cfg, W, H, color=(0, 0, 0))

    # matplotlib ile metin (input üzerine overlay)
    if cfg.add_text_labels or cfg.add_text_boxes:
        fig_txt, ax_txt = _make_fig(cfg, W, H)
        # Şeffaf metin katmanı — beyaz arka plan, sadece metin
        if cfg.add_text_labels:
            _draw_text_labels(ax_txt, cfg, arrow_specs, color='black')
        if cfg.add_text_boxes:
            _draw_text_boxes(ax_txt, cfg, color='black')
        text_layer = _fig_to_array(fig_txt)
        text_layer = cv2.resize(text_layer, (W, H))
        # Metin piksellerini input'a overlay (beyaz olmayan pikseller)
        text_gray = cv2.cvtColor(text_layer, cv2.COLOR_RGB2GRAY)
        text_pixels = text_gray < 240  # metin olan pikseller
        input_img[text_pixels] = text_layer[text_pixels]

    # ── 5. Mask katmanları ──
    # Arrow mask (R kanalı) — callout oklar
    arrow_mask = np.zeros((H, W), dtype=np.uint8)
    if arrow_specs:
        arrow_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_arrows_on_img(arrow_canvas, arrow_specs, cfg, W, H,
                            color=(255, 255, 255), use_pil=arrow_use_pil)
        arrow_mask = cv2.cvtColor(arrow_canvas, cv2.COLOR_RGB2GRAY)
        _, arrow_mask = cv2.threshold(arrow_mask, 20, 255, cv2.THRESH_BINARY)

    # Dashed mask (Ch1) — düz kesikler
    dashed_mask = np.zeros((H, W), dtype=np.uint8)
    if dashed_specs:
        dashed_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_dashed_on_img(dashed_canvas, dashed_specs, cfg, W, H,
                            color=(255, 255, 255))
        dashed_mask = cv2.cvtColor(dashed_canvas, cv2.COLOR_RGB2GRAY)
        _, dashed_mask = cv2.threshold(dashed_mask, 20, 255, cv2.THRESH_BINARY)

    # 2-kanal mask: Ch0=arrows, Ch1=dashed
    # (Text mask yok — text detection Tesseract OCR ile postprocess'te yapılır)
    noise_mask = np.stack([arrow_mask, dashed_mask], axis=-1)

    # ── 6. Artifact ──
    if add_artifacts:
        input_img = add_scan_artifacts(input_img)

    return input_img, noise_mask, clean_img


def make_sample_pair(W=512, H=512, seed=None, add_artifacts=True):
    """(input, target) çifti döndürür."""
    inp, mask, _ = make_sample(W, H, seed, add_artifacts)
    return inp, mask


# ════════════════════════════════════════════════════════════════
# TORCH DATASET
# ════════════════════════════════════════════════════════════════

def get_torch_dataset(n_samples: int = 6000, W: int = 512, H: int = 512):
    import torch
    from torch.utils.data import Dataset

    class NoiseDataset(Dataset):
        def __init__(self, n, w, h):
            self.n, self.w, self.h = n, w, h

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            inp, mask = make_sample_pair(self.w, self.h, seed=None,
                                         add_artifacts=True)
            inp_t = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
            return inp_t, mask_t

    return NoiseDataset(n_samples, W, H)


# ════════════════════════════════════════════════════════════════
# TEST
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os

    print('Generating noise removal samples (v2 — cv2 arrows/dashes)...')
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(3):
        inp, mask, clean = make_sample(512, 512, seed=i * 42)
        cv2.imwrite(os.path.join(out_dir, f'test_input_{i}.png'),
                    cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
        # Mask artık 2 kanallı — 3. kanalı siyah yap görselleştirme için
        mask_vis = np.zeros((512, 512, 3), dtype=np.uint8)
        mask_vis[:, :, 0] = mask[:, :, 0]  # arrows
        mask_vis[:, :, 1] = mask[:, :, 1]  # dashed
        cv2.imwrite(os.path.join(out_dir, f'test_mask_{i}.png'),
                    cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, f'test_clean_{i}.png'),
                    cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        print(f'  sample {i}  Ch0(arrow)={mask[:,:,0].sum()>0}  '
              f'Ch1(dashed)={mask[:,:,1].sum()>0}')

    print('Done!')
