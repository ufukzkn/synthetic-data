# synthetic_data.py
import io
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
import matplotlib.pyplot as plt


# -----------------------------
# Render helper
# -----------------------------
def _fig_to_pil(fig, dpi=180):
    """
    Render a matplotlib figure to a PIL RGB image.
    bbox_inches='tight' removes extra margins, but output size can vary slightly,
    so later we resize to (W,H) explicitly.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# -----------------------------
# Whitespace trimming after rotate(expand=True)
# -----------------------------
def _trim_whitespace(pil_img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """
    Crop near-white borders introduced by rotation with expand=True.
    """
    img = pil_img.convert("RGB")
    bg_img = Image.new("RGB", img.size, bg)
    diff = ImageChops.difference(img, bg_img)

    # boost contrast so almost-white becomes 0, and content becomes visible
    diff = ImageEnhance.Contrast(diff).enhance(12.0)

    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


# -----------------------------
# Scan artifacts
# -----------------------------
def _add_scan_artifacts(pil_img: Image.Image, out_size=None) -> Image.Image:
    """
    Simulate scanned/photocopied chart artifacts:
    - slight rotation/skew (expand=True -> then trim -> resize back)
    - contrast/brightness drift
    - gaussian noise + mild blur
    - JPEG artifacts
    """
    # 1) slight skew/rotation
    angle = random.uniform(-1.5, 1.5)
    pil_img = pil_img.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255)
    )

    # 2) trim expanded white borders and normalize size
    pil_img = _trim_whitespace(pil_img)
    if out_size is not None:
        pil_img = pil_img.resize(out_size, Image.BILINEAR)

    # 3) contrast/brightness drift
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.65, 1.35))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.80, 1.20))

    # 4) gaussian noise
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    sigma = random.uniform(0.0, 0.03)
    arr += np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)

    # 5) mild blur sometimes via down-up sampling
    if random.random() < 0.55:
        w, h = pil_img.size
        scale = random.uniform(0.85, 0.98)
        tmp = Image.fromarray((arr * 255).astype(np.uint8))
        tmp = tmp.resize((max(2, int(w * scale)), max(2, int(h * scale))), Image.BILINEAR)
        tmp = tmp.resize((w, h), Image.BILINEAR)
        arr = np.asarray(tmp).astype(np.float32) / 255.0

    # 6) JPEG artifacts
    if random.random() < 0.65:
        tmp_io = io.BytesIO()
        Image.fromarray((arr * 255).astype(np.uint8)).save(
            tmp_io, format="JPEG", quality=random.randint(25, 70)
        )
        tmp_io.seek(0)
        pil_img = Image.open(tmp_io).convert("RGB")
    else:
        pil_img = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")

    # ensure final size
    if out_size is not None:
        pil_img = pil_img.resize(out_size, Image.BILINEAR)

    return pil_img


# -----------------------------
# Curve synthesis
# -----------------------------
def _generate_smooth_curves(num_curves=8, n_points=450):
    """
    Smooth, mostly horizontal, x-monotonic curves.
    Non-intersecting-ish via ordered base levels + shared low-frequency shape.
    """
    x = np.linspace(0, 1, n_points)

    # Shared family shape
    shape = np.zeros_like(x)
    for _ in range(random.randint(1, 4)):
        freq = random.uniform(0.5, 2.2)
        phase = random.uniform(0, 2 * np.pi)
        amp = random.uniform(0.015, 0.07)
        shape += amp * np.sin(2 * np.pi * freq * x + phase)

    # gentle trend
    trend = random.uniform(-0.08, 0.08) * (x - 0.5)

    base_levels = np.linspace(0.20, 0.70, num_curves) + np.random.uniform(-0.02, 0.02, num_curves)
    base_levels = np.sort(base_levels)

    curves = []
    for i in range(num_curves):
        local = shape + trend
        local += random.uniform(-0.03, 0.03) * np.sin(
            2 * np.pi * random.uniform(0.6, 2.7) * x + random.uniform(0, 2 * np.pi)
        )
        y = base_levels[i] + local
        y = np.clip(y, 0.05, 0.95)
        curves.append((x, y))
    return curves


# -----------------------------
# Hard negative clutter: arrows / leaders / callouts
# -----------------------------
def _add_arrow_clutter(ax, x_min, x_max, y_min, y_max, n_arrows=None):
    """
    Add arrows/leader lines like aircraft charts.
    These must be background (class 0), so they are only on FULL image.
    """
    if n_arrows is None:
        n_arrows = random.randint(10, 35)

    for _ in range(n_arrows):
        x_head = random.uniform(x_min + (x_max-x_min)*0.05, x_max - (x_max-x_min)*0.05)
        y_head = random.uniform(y_min + (y_max-y_min)*0.05, y_max - (y_max-y_min)*0.05)

        dx = random.uniform(-(x_max-x_min)*0.15, (x_max-x_min)*0.15)
        dy = random.uniform(-(y_max-y_min)*0.10, (y_max-y_min)*0.10)

        x_tail = float(np.clip(x_head + dx, x_min, x_max))
        y_tail = float(np.clip(y_head + dy, y_min, y_max))

        lw = random.uniform(0.6, 1.3)  # similar to curve thickness
        ax.annotate(
            "",
            xy=(x_head, y_head),
            xytext=(x_tail, y_tail),
            arrowprops=dict(
                arrowstyle=random.choice(["->", "-|>", "->"]),
                lw=lw,
                color="black",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=random.uniform(8, 14),
                alpha=random.uniform(0.75, 1.0),
            ),
        )

        if random.random() < 0.35:
            ax.text(
                x_head + random.uniform(-(x_max-x_min)*0.03, (x_max-x_min)*0.03),
                y_head + random.uniform(-(y_max-y_min)*0.03, (y_max-y_min)*0.03),
                random.choice(["Vmax", "Opt", "MIL", "CRZ", "MAX", "REF"]),
                fontsize=random.randint(7, 11),
                color="black",
                alpha=random.uniform(0.75, 1.0),
            )


def _add_callout_boxes(ax, x_min, x_max, y_min, y_max):
    """Add boxed legends/callouts similar to aircraft charts."""
    if random.random() < 0.60:
        x = random.uniform(x_min + (x_max-x_min)*0.05, x_max - (x_max-x_min)*0.35)
        y = random.uniform(y_min + (y_max-y_min)*0.15, y_max - (y_max-y_min)*0.10)
        txt = random.choice([
            "TOTAL FUEL FLOW\nPOUNDS PER HOUR",
            "OPTIMUM CRUISE",
            "MAXIMUM ENDURANCE",
            "USE FOR INTERFERENCE\nDRAG DETERMINATION",
            "STANDARD DAY\nSEA LEVEL",
        ])
        ax.text(
            x, y, txt,
            fontsize=random.randint(8, 12),
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="square,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8
            ),
            alpha=1.0
        )


# -----------------------------
# Main: generate a synthetic (image, mask)
# -----------------------------
def make_synth_sample(W=900, H=650, num_curves=None, seed=None):
    """
    Returns:
        full_img: PIL RGB (curves + grid + text + arrows + scan noise)
        mask_bin: np.uint8 {0,1} (ONLY curves are 1)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if num_curves is None:
        num_curves = random.randint(6, 12)

    curves = _generate_smooth_curves(num_curves=num_curves)

    # Axis ranges roughly matching aircraft chart feel
    x_min, x_max = 0.10, 1.00
    y_min, y_max = 0.01, 0.10

    # -------- FULL PLOT --------
    fig = plt.figure(figsize=(W / 180, H / 180))
    ax = fig.add_subplot(111)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Dense grid (major + minor)
    major_x = random.choice([0.10, 0.05])
    minor_x = major_x / 2
    major_y = random.choice([0.01, 0.02])
    minor_y = major_y / 2

    ax.set_xticks(np.arange(x_min, x_max + 1e-9, major_x))
    ax.set_xticks(np.arange(x_min, x_max + 1e-9, minor_x), minor=True)
    ax.set_yticks(np.arange(y_min, y_max + 1e-9, major_y))
    ax.set_yticks(np.arange(y_min, y_max + 1e-9, minor_y), minor=True)

    grid_lw_major = random.uniform(0.7, 1.2)
    grid_lw_minor = random.uniform(0.5, 1.0)
    grid_alpha_major = random.uniform(0.30, 0.65)
    grid_alpha_minor = random.uniform(0.20, 0.50)

    ax.grid(which="major", linewidth=grid_lw_major, alpha=grid_alpha_major, color="black")
    ax.grid(which="minor", linewidth=grid_lw_minor, alpha=grid_alpha_minor, color="black")

    # Curves (thickness similar to grid)
    curve_lw = random.uniform(0.9, 1.6)
    for (x, y) in curves:
        xx = x_min + (x_max - x_min) * x
        yy = y_min + (y_max - y_min) * y
        ax.plot(xx, yy, color="black", linewidth=curve_lw)

    # Axis labels
    ax.set_xlabel("MACH NUMBER", fontsize=random.randint(9, 13))
    ax.set_ylabel("SPECIFIC RANGE — NAUTICAL MILES PER POUND OF FUEL", fontsize=random.randint(9, 13))

    # Title (sometimes)
    if random.random() < 0.6:
        ax.set_title(random.choice(["CRUISE PERFORMANCE", "SPECIFIC RANGE", ""]), fontsize=random.randint(10, 14))

    # Extra random text clutter
    for _ in range(random.randint(4, 14)):
        tx = random.uniform(x_min + (x_max-x_min)*0.03, x_max - (x_max-x_min)*0.03)
        ty = random.uniform(y_min + (y_max-y_min)*0.08, y_max - (y_max-y_min)*0.08)
        ax.text(
            tx, ty,
            random.choice(["4500", "5000", "6500", "9000", "10,000", "12,000", "17,000", "Vmax", "MIL"]),
            fontsize=random.randint(7, 11),
            color="black",
            alpha=random.uniform(0.75, 1.0),
        )

    _add_callout_boxes(ax, x_min, x_max, y_min, y_max)

    # IMPORTANT hard negatives
    _add_arrow_clutter(ax, x_min, x_max, y_min, y_max)

    full_img = _fig_to_pil(fig)

    # Normalize size BEFORE artifacts to keep stable framing
    full_img = full_img.resize((W, H), Image.BILINEAR)

    # Apply scan artifacts (keeps size stable)
    full_img = _add_scan_artifacts(full_img, out_size=(W, H))

    # -------- MASK (CURVES ONLY) --------
    fig2 = plt.figure(figsize=(W / 180, H / 180))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    ax2.axis("off")
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")

    for (x, y) in curves:
        xx = x_min + (x_max - x_min) * x
        yy = y_min + (y_max - y_min) * y
        ax2.plot(xx, yy, color="white", linewidth=curve_lw)

    mask_img = _fig_to_pil(fig2).resize((W, H), Image.NEAREST)

    mask_arr = np.asarray(mask_img.convert("L"))
    mask_bin = (mask_arr > 30).astype(np.uint8)

    return full_img, mask_bin


# Optional quick test
if __name__ == "__main__":
    img, mask = make_synth_sample(seed=0)
    img.save("debug_full.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save("debug_mask.png")
    print("Wrote debug_full.png and debug_mask.png")
