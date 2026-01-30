import numpy as np
import cv2
from PIL import Image



def detect_plot_bbox(pil_img):
    """
    Returns bbox in pixel coords of the resized image: (x0,y0,x1,y1)
    Simple heuristic: find dense line region by edge projection.
    """
    img = np.asarray(pil_img.convert("L"))
    # edges
    edges = cv2.Canny(img, 50, 150)

    # sum projections
    col_sum = edges.sum(axis=0)
    row_sum = edges.sum(axis=1)

    # find central mass (ignore margins)
    W = edges.shape[1]
    H = edges.shape[0]
    mx0, mx1 = int(W*0.05), int(W*0.95)
    my0, my1 = int(H*0.05), int(H*0.95)

    col_sum[:mx0] = 0; col_sum[mx1:] = 0
    row_sum[:my0] = 0; row_sum[my1:] = 0

    # thresholds relative to max
    cx = np.where(col_sum > 0.25*col_sum.max())[0]
    cy = np.where(row_sum > 0.25*row_sum.max())[0]
    if len(cx) < 10 or len(cy) < 10:
        # fallback: full image
        return (0,0,W-1,H-1)

    x0, x1 = int(cx.min()), int(cx.max())
    y0, y1 = int(cy.min()), int(cy.max())

    # pad slightly inward
    pad = 3
    x0 = max(0, x0+pad); y0 = max(0, y0+pad)
    x1 = min(W-1, x1-pad); y1 = min(H-1, y1-pad)
    return (x0,y0,x1,y1)

def pixels_to_normalized_xy(path_rc, bbox):
    # path_rc: (K,2) rows, cols
    x0,y0,x1,y1 = bbox
    cols = path_rc[:,1].astype(np.float32)
    rows = path_rc[:,0].astype(np.float32)

    xn = (cols - x0) / max(1e-6, (x1 - x0))
    yn = 1.0 - (rows - y0) / max(1e-6, (y1 - y0))  # invert y

    return np.stack([xn, yn], axis=1)

def normalized_to_axis(xy_norm, x_min, x_max, y_min, y_max):
    x = x_min + (x_max - x_min) * xy_norm[:,0]
    y = y_min + (y_max - y_min) * xy_norm[:,1]
    return np.stack([x,y], axis=1)
