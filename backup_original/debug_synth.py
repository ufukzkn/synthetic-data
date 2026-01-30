from synthetic_data import make_synth_sample
import numpy as np
from PIL import Image

if __name__ == "__main__":
    img, mask = make_synth_sample(seed=0)
    img.save("debug_full.png")
    Image.fromarray((mask*255).astype(np.uint8)).save("debug_mask.png")
    print("Wrote debug_full.png and debug_mask.png")
