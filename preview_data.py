# ══════════════════════════════════════════════════════════════════════════════
# 🔍 LOCAL VERİ ÖNİZLEME - Her curve type'tan 1 örnek
# ══════════════════════════════════════════════════════════════════════════════
# Bu dosyayı local'de çalıştırarak sentetik verinin gerçekçiliğini kontrol et!
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from dataclasses import dataclass
import random
import math
import io
import os

# Görüntü boyutu
IMG_SIZE = 512

# ══════════════════════════════════════════════════════════════════════════════
# 📊 CHART CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChartConfig:
    x_min: float = 0.30
    x_max: float = 1.00
    y_min: float = 0.04
    y_max: float = 0.15
    n_curves: int = 8
    curve_type: str = 'peaked'
    curve_lw: float = 0.6
    add_grid: bool = True
    add_arrows: bool = True
    add_envelope_optimum: bool = True
    add_envelope_endurance: bool = False
    add_vmax_line: bool = False
    add_text_boxes: bool = True
    add_fuel_labels: bool = True
    add_drag_labels: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# 📈 CURVE SHAPE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_curve_shape(x, curve_type, curve_index, total_curves):
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
                
    elif curve_type == 'peaked_oval':
        peak_pos = 0.45 + random.uniform(-0.07, 0.07)
        start_y = 0.12 + random.uniform(-0.02, 0.02)
        peak_y = 0.45 + alt * 0.38 + random.uniform(-0.03, 0.03)
        end_y = 0.20 + alt * 0.25 + random.uniform(-0.02, 0.02)
        y = np.zeros_like(x_norm)
        for i, t in enumerate(x_norm):
            if t <= peak_pos:
                progress = t / peak_pos
                y[i] = start_y + (peak_y - start_y) * (math.sin(progress * math.pi / 2) ** 1.2)
            else:
                progress = (t - peak_pos) / (1 - peak_pos)
                y[i] = end_y + (peak_y - end_y) * (math.cos(progress * math.pi / 2) ** 1.2)
                
    elif curve_type == 'wavy':
        freq = random.choice([1.0, 1.5, 2.0])
        phase = random.uniform(0, 1)
        wave = 0.5 + 0.25 * np.sin(2 * np.pi * (x_norm * freq + phase))
        wave += 0.12 * np.sin(4 * np.pi * (x_norm * freq + phase))
        hump_center = random.uniform(0.55, 0.70)
        hump = 0.12 * np.exp(-((x_norm - hump_center) / 0.22) ** 2)
        y = wave + hump
        y = np.clip(y, 0.05, 0.95)
        
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
        
    else:
        return generate_curve_shape(x, random.choice(['peaked', 'peaked_oval', 'rising', 'falling', 'wavy']),
                                   curve_index, total_curves)
    return y


# ══════════════════════════════════════════════════════════════════════════════
# 🖼️ FIGURE TO ARRAY
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_array(fig, dpi=150, tight=True):
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.02,
                    facecolor='white', edgecolor='none')
    else:
        fig.savefig(buf, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


# ══════════════════════════════════════════════════════════════════════════════
# 🎨 DRAW CHART
# ══════════════════════════════════════════════════════════════════════════════

def draw_chart_matplotlib(config, W=512, H=512):
    """Draw chart and return image + colored mask (RGB)"""
    fig_w, fig_h = W / 100, H / 100
    
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
    
    ax1.axhline(y=config.y_min, color='black', linewidth=2.0, zorder=10)
    ax1.axvline(x=config.x_min, color='black', linewidth=2.0, zorder=10)
    ax1.tick_params(axis='both', which='major', length=6, width=1.5, direction='in')
    ax1.tick_params(axis='both', which='minor', length=3, width=1.0, direction='in')
    
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    ax1.set_xlabel('MACH NUMBER', fontsize=10, fontweight='bold')
    ax1.set_ylabel('SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL', fontsize=8)
    
    # Ana eğriler
    for cx, cy in curves_data:
        ax1.plot(cx, cy, 'k-', linewidth=config.curve_lw)
    
    # OPTIMUM CRUISE ENVELOPE
    if config.add_envelope_optimum:
        if config.curve_type in ['peaked', 'peaked_oval']:
            envelope_pts = [(cx[np.argmax(cy)], cy.max()) for cx, cy in curves_data]
        else:
            envelope_pts = [(cx[int(len(cx)*0.5)], cy[int(len(cy)*0.5)]) for cx, cy in curves_data]
        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2)
        ax1.text(ex[0] - 0.03, ey[-1] + (config.y_max - config.y_min) * 0.02,
                'OPTIMUM\nCRUISE', fontsize=8, ha='right', va='bottom')
    
    # MAXIMUM ENDURANCE ENVELOPE
    if config.add_envelope_endurance:
        envelope_pts = [(cx[int(len(cx)*0.2)], cy[int(len(cy)*0.2)]) for cx, cy in curves_data]
        envelope_pts.sort(key=lambda p: p[1])
        ex, ey = zip(*envelope_pts)
        ax1.plot(ex, ey, 'k-', linewidth=1.2)
        ax1.text(ex[-1] - 0.02, ey[0] - (config.y_max - config.y_min) * 0.02,
                'MAXIMUM\nENDURANCE', fontsize=8, ha='right', va='top')
    
    # ARROWS & FUEL LABELS
    if config.add_arrows:
        fuel_flows = ['3000', '3500', '4000', '4500', '5000', '5500', '6000', '6500', '7000', '7500', '8000', '8500']
        for idx, (cx, cy) in enumerate(reversed(curves_data)):
            if idx >= len(fuel_flows):
                break
            
            if random.random() < 0.5:
                arrow_idx = -1
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]
                dx = random.uniform(0.04, 0.08)
                dy = random.uniform(-0.005, 0.005)
                x_tail = x_head + dx
                y_tail = y_head + dy
            else:
                mid_start = len(cx) // 4
                mid_end = 3 * len(cx) // 4
                arrow_idx = random.randint(mid_start, mid_end)
                x_head = cx[arrow_idx]
                y_head = cy[arrow_idx]
                angle = random.uniform(20, 70)
                dist = random.uniform(0.05, 0.10)
                if random.random() < 0.5:
                    dx = dist * math.cos(math.radians(angle))
                    dy = dist * math.sin(math.radians(angle))
                else:
                    dx = dist * math.cos(math.radians(-angle))
                    dy = dist * math.sin(math.radians(-angle))
                x_tail = x_head + dx
                y_tail = y_head + dy
            
            ax1.plot([x_tail, x_head], [y_tail, y_head], color="black", linewidth=0.6)
            
            if random.random() < 0.4:
                arrow_style = random.choice(["-|>", "->"])
                fill_style = "none"
            else:
                arrow_style = random.choice(["-|>", "-|>", "->"])
                fill_style = "black"
            
            ax1.annotate(
                "",
                xy=(x_head, y_head),
                xytext=(x_tail, y_tail),
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    lw=random.uniform(0.7, 1.1),
                    color="black",
                    fc=fill_style,
                    shrinkA=0, shrinkB=0,
                    mutation_scale=random.uniform(12, 18),
                ),
            )
            
            if config.add_fuel_labels:
                label_x = x_tail + random.uniform(0.02, 0.05)
                ax1.text(label_x, y_tail + random.uniform(-0.002, 0.002),
                        fuel_flows[idx], fontsize=8, va='center', ha='left')
    
    # KESİKLİ ÇİZGİLER (DASHED LINES)
    n_dashed = random.randint(3, 8)
    for _ in range(n_dashed):
        dcx = config.x_min + (config.x_max - config.x_min) * random.uniform(0.15, 0.85)
        dcy = config.y_min + (config.y_max - config.y_min) * random.uniform(0.15, 0.85)
        dash_len = random.uniform(0.04, 0.12)
        dash_angle = math.radians(random.choice([30, 40, 45, 50, 60, 70]))
        dash_dx = dash_len * math.cos(dash_angle)
        dash_dy = dash_len * math.sin(dash_angle)
        dash_style = random.choice([(0, (12, 6)), (0, (8, 4)), (0, (15, 8)), (0, (6, 3))])
        ax1.plot([dcx, dcx + dash_dx], [dcy, dcy + dash_dy],
                color="black", linewidth=random.uniform(0.4, 0.7), linestyle=dash_style)
    
    # TEXT BOXES & LEGEND
    if config.add_text_boxes:
        ax1.text(config.x_max - 0.05, config.y_max - 0.005,
                'TOTAL FUEL FLOW—\nPOUNDS PER HOUR',
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black'))
        
        legend_x = config.x_min + (config.x_max - config.x_min) * 0.12
        legend_y = config.y_max - (config.y_max - config.y_min) * 0.08
        ax1.text(legend_x, legend_y,
                '◄─ CRUISE    DASH ─►\n      AOA          AOA\n(USED FOR INTERFERENCE\n DRAG DETERMINATION)',
                fontsize=7, ha='left', va='top',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black'))
    
    # DRAG INDEX LABELS
    if config.add_drag_labels:
        labels = ['0.00', '25.00', '50.00', '75.00', '100.00', '125.00', '150.00']
        base_x = config.x_min + (config.x_max - config.x_min) * 0.65
        base_y = config.y_min + (config.y_max - config.y_min) * 0.15
        for i, lbl in enumerate(labels[:random.randint(4, 7)]):
            ax1.text(base_x + random.uniform(-0.02, 0.02),
                    base_y + i * (config.y_max - config.y_min) * 0.05,
                    lbl, fontsize=7, alpha=0.9)
    
    # VMAX LINE
    if config.add_vmax_line:
        vmax_pts = [(cx[int(len(cx)*0.85)], cy[int(len(cy)*0.85)]) for cx, cy in curves_data]
        vmax_pts.sort(key=lambda p: p[1])
        vx, vy = zip(*vmax_pts)
        ax1.plot(vx, vy, 'k--', linewidth=0.8)
        ax1.text(vx[-1], vy[-1] + 0.003, r'$V_{max}$(MIL)', fontsize=7)
    
    full_img = fig_to_array(fig1, dpi=150, tight=True)
    full_img = cv2.resize(full_img, (W, H))
    
    # ========== COLORED MASK ==========
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.set_xlim(config.x_min, config.x_max)
    ax2.set_ylim(config.y_min, config.y_max)
    ax2.set_position([0, 0, 1, 1])
    ax2.axis('off')
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')
    
    n_curves = len(curves_data)
    for i, (cx, cy) in enumerate(curves_data):
        hue = int(180 * i / max(n_curves, 1))
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
        ax2.plot(cx, cy, color=np.array(rgb_color) / 255.0, linewidth=config.curve_lw + 0.2, zorder=2)
    
    colored_img = fig_to_array(fig2, dpi=150, tight=False)
    colored_img = cv2.resize(colored_img, (W, H))
    
    return full_img, colored_img, curves_data


# ══════════════════════════════════════════════════════════════════════════════
# 📷 SCAN ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def add_scan_artifacts(img, strength=1.0):
    """Scan/photocopy artifacts ekle"""
    pil_img = Image.fromarray(img)
    angle = random.uniform(-1.2, 1.2) * strength
    pil_img = pil_img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.90, 1.10))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.88, 1.12))
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.012 * strength, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format='JPEG', quality=random.randint(50, 80))
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


# ══════════════════════════════════════════════════════════════════════════════
# 🔍 ANA FONKSİYON - HER TÜRDEN 1 ÖRNEK ÜRET
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("🔍 SENTETİK VERİ ÖNİZLEME - Her Curve Type'tan 1 Örnek")
    print("="*70)
    
    curve_types = ['peaked', 'peaked_oval', 'rising', 'falling', 'wavy']
    
    fig, axes = plt.subplots(len(curve_types), 2, figsize=(14, 5 * len(curve_types)))
    
    for row, ctype in enumerate(curve_types):
        print(f"\n📊 {ctype.upper()} üretiliyor...")
        
        # Config oluştur (tüm özellikler açık)
        cfg = ChartConfig(
            x_min=0.35, x_max=1.15,
            y_min=0.05, y_max=0.16,
            n_curves=4 if ctype == 'wavy' else random.randint(6, 9),
            curve_type=ctype,
            curve_lw=random.uniform(0.4, 0.55),
            add_grid=True,
            add_arrows=True,
            add_envelope_optimum=True,
            add_envelope_endurance=True,
            add_vmax_line=True,
            add_text_boxes=True,
            add_fuel_labels=True,
            add_drag_labels=True
        )
        
        # Grafik üret
        img, colored, _ = draw_chart_matplotlib(cfg, W=IMG_SIZE, H=IMG_SIZE)
        img_noisy = add_scan_artifacts(img, strength=1.0)
        
        # INPUT göster
        axes[row, 0].imshow(img_noisy)
        axes[row, 0].set_title(f'📥 INPUT: {ctype.upper()}\n({cfg.n_curves} eğri, gürültülü)', 
                               fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # TARGET göster
        axes[row, 1].imshow(colored)
        axes[row, 1].set_title(f'🎯 TARGET: RGB Mask\n(HSV renklendirme)', 
                               fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        
        print(f"   ✅ {ctype}: {cfg.n_curves} eğri")
    
    plt.suptitle('📊 SENTETİK VERİ ÖNİZLEME\n(Grid, Dashed Lines, Arrows, Legend Box, Vmax Line)', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Kaydet
    output_path = os.path.join(os.path.dirname(__file__), 'preview_all_types.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\n💾 Kaydedildi: {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("✅ ÖNİZLEME TAMAMLANDI!")
    print("="*70)
    print("\n📋 Kontrol Listesi:")
    print("   ✓ Grid (major + minor çizgiler)")
    print("   ✓ Kesikli çizgiler (dashed lines)")
    print("   ✓ Oklar ve fuel flow etiketleri")
    print("   ✓ Legend box (sol üst)")
    print("   ✓ Text box (sağ üst)")
    print("   ✓ Vmax line")
    print("   ✓ Envelope çizgileri")
    print("   ✓ HSV renklendirme")
    print("   ✓ Scan artifacts")


if __name__ == "__main__":
    main()
