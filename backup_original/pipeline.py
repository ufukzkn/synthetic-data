import os
import numpy as np
from PIL import Image
import torch

from postprocess import load_model, predict_mask, cleanup_and_skeletonize
from curve_separation import connected_components_skeleton, stitch_fragments
from curve_ordering import order_curve_pixels_left_to_right
from coordinates import detect_plot_bbox, pixels_to_normalized_xy, normalized_to_axis
from export import save_curve_csv, export_svg



def extract_curves_to_csv_and_svg(
    image_path,
    weights_path="curve_unet.pt",
    out_dir="out_curves",
    input_size=512,
    prob_thr=0.5,
    axis_limits=None  # optional (x_min,x_max,y_min,y_max)
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(weights_path, device)

    pil_img = Image.open(image_path).convert("RGB")

    # Predict on resized image
    bin_mask, img_res = predict_mask(model, pil_img, device, input_size=input_size, thr=prob_thr)
    Image.fromarray((bin_mask*255).astype(np.uint8)).save("debug_pred_mask.png")





    skel = cleanup_and_skeletonize(bin_mask)
    Image.fromarray((skel*255).astype(np.uint8)).save("debug_skeleton.png")
    # components -> stitch -> tracks
    fragments = connected_components_skeleton(skel)
    tracks = stitch_fragments(fragments, max_y_gap=7, max_x_back=12)

    # detect plot bbox on resized image (for normalized mapping)
    bbox = detect_plot_bbox(img_res)

    curve_points_for_svg = []
    csv_paths = []

    for curve_id, frag_ids in enumerate(tracks):
        # merge pixels
        coords = np.concatenate([fragments[j] for j in frag_ids], axis=0)
        # order left-to-right
        ordered_rc = order_curve_pixels_left_to_right(coords)

        # map to normalized axis
        xy_norm = pixels_to_normalized_xy(ordered_rc, bbox)  # (N,2) in [0,1]
        if axis_limits is not None:
            x_min,x_max,y_min,y_max = axis_limits
            xy = normalized_to_axis(xy_norm, x_min,x_max,y_min,y_max)
        else:
            xy = xy_norm  # still useful + reproducible

        # save CSV
        csv_path = os.path.join(out_dir, f"curve_{curve_id:02d}.csv")
        save_curve_csv(xy, csv_path)
        csv_paths.append(csv_path)

        # SVG points in pixel coords of resized image
        # Use the ordered pixels directly in SVG coordinate system (x=col, y=row)
        pts_svg = np.stack([ordered_rc[:,1].astype(np.float32), ordered_rc[:,0].astype(np.float32)], axis=1)
        curve_points_for_svg.append(pts_svg)

    # Export SVG
    svg_path = os.path.join(out_dir, "curves.svg")
    export_svg(curve_points_for_svg, svg_path, width=input_size, height=input_size)

    return {"csv_files": csv_paths, "svg_file": svg_path}
