import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import numpy as np
import torch
from PIL import Image
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from torchvision import transforms

from model import UNetSmall


def load_model(weights_path, device):
    model = UNetSmall(in_ch=3, base=32).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def predict_mask(model, pil_img, device, input_size=512, thr=0.5):
    # preprocess
    img = pil_img.convert("RGB")
    img_res = img.resize((input_size, input_size), Image.BILINEAR)
    x = transforms.ToTensor()(img_res).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    bin_mask = (prob > thr).astype(np.uint8)

    return bin_mask, img_res

def cleanup_and_skeletonize(bin_mask):
    # Morph cleanup in OpenCV (expects 0/1)
    m = (bin_mask*255).astype(np.uint8)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # OPEN bazen curve koparıyor -> şimdilik kaldırıyoruz
    # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1, iterations=1)

    # Strong bridge close (dilate->close->erode)
    m = cv2.dilate(m, k1, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=3)
    m = cv2.erode(m, k1, iterations=1)

    # remove tiny components (noise)
    lab = label(m > 0)
    out = np.zeros_like(m, dtype=np.uint8)
    for r in regionprops(lab):
        if r.area >= 40:  # gerekirse 20 yaparız
            out[lab == r.label] = 255

    skel = skeletonize(out > 0).astype(np.uint8)  # 0/1

    # debug (geçici)
    Image.fromarray(out).save("debug_cleaned_mask.png")
    Image.fromarray((skel*255).astype(np.uint8)).save("debug_skeleton.png")

    return skel
