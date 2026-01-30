import os
import csv
import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev

import os
import csv

def save_curve_csv(xy, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x,y in xy:
            w.writerow([float(x), float(y)])
import svgwrite
from scipy.interpolate import splprep, splev

def smooth_polyline(points_xy, smoothing=2.0, n_out=300):
    """
    points_xy: (N,2) float
    returns smoothed (n_out,2)
    """
    if len(points_xy) < 6:
        return points_xy

    x = points_xy[:,0]
    y = points_xy[:,1]
    # param spline
    tck, _ = splprep([x, y], s=smoothing)
    u = np.linspace(0, 1, n_out)
    xo, yo = splev(u, tck)
    return np.stack([xo, yo], axis=1)

def catmull_rom_to_beziers(P):
    """
    P: (N,2) points
    Returns list of cubic bezier segments: [(p0,c1,c2,p3), ...]
    """
    if len(P) < 4:
        return []

    segs = []
    for i in range(1, len(P)-2):
        p0 = P[i]
        p_1 = P[i-1]
        p1 = P[i+1]
        p2 = P[i+2]

        # Catmull-Rom to cubic Bezier
        c1 = p0 + (p1 - p_1) / 6.0
        c2 = p1 - (p2 - p0) / 6.0
        segs.append((p0, c1, c2, p1))
    return segs

def export_svg(curves_points, out_svg_path, width=512, height=512):
    """
    curves_points: list of (M,2) points in SVG pixel coords (x,y)
    """
    dwg = svgwrite.Drawing(out_svg_path, size=(width, height))
    dwg.add(dwg.rect(insert=(0,0), size=(width,height), fill="white"))

    # distinct colors (simple palette)
    palette = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#999999"]

    for i, pts in enumerate(curves_points):
        if len(pts) < 4:
            continue
        pts_s = smooth_polyline(pts, smoothing=2.0, n_out=240)
        segs = catmull_rom_to_beziers(pts_s)

        if not segs:
            continue

        color = palette[i % len(palette)]
        path = dwg.path(d=f"M {segs[0][0][0]:.2f},{segs[0][0][1]:.2f}",
                        fill="none", stroke=color, stroke_width=2, stroke_linecap="round")

        for (p0,c1,c2,p3) in segs:
            path.push(f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}")

        dwg.add(path)

    dwg.save()
