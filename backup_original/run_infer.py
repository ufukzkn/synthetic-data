from pipeline import extract_curves_to_csv_and_svg

if __name__ == "__main__":
    image_path = "debug_full.png"   # real image
    # image_path = "debug_full.png" # sanity check

    result = extract_curves_to_csv_and_svg(
        image_path=image_path,
        weights_path="curve_unet_overfit.pt",
        out_dir="out_curves",
        input_size=512,
        prob_thr=0.15,
        axis_limits=None
    )

    print("SVG:", result["svg_file"])
    print("CSVs:")
    for p in result["csv_files"]:
        print(" -", p)
