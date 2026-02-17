"""Quick batch test: 10 sample_out images through trained UNet."""
import cv2, numpy as np, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNet
from test_local import predict_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load best model
model_path = 'models/noise_unet.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=3).to(device)
ckpt = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
ep = ckpt.get('epoch', '?')
loss_val = ckpt.get('loss', 0)
print(f'Model loaded: epoch={ep}, loss={loss_val:.4f}')

# First 10 images from sample_out
sample_dir = 'sample_out'
all_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])
picks = all_files[:10]
n = len(picks)
print(f'Testing {n} images from {sample_dir}/')

fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))
if n == 1:
    axes = axes.reshape(1, -1)

for i, fname in enumerate(picks):
    path = os.path.join(sample_dir, fname)
    data = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Predict
    pred_raw = predict_mask(model, img_rgb, device, img_size=512)

    # Threshold
    threshold = 0.5
    arrows = (pred_raw[:, :, 0] > threshold).astype(np.uint8) * 255
    dashed = (pred_raw[:, :, 1] > threshold).astype(np.uint8) * 255
    text   = (pred_raw[:, :, 2] > threshold).astype(np.uint8) * 255
    combined = np.maximum(np.maximum(arrows, dashed), text)

    # Inpaint
    img_bgr2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr2, combined, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    # Stats
    n_a = int(arrows.sum() // 255)
    n_d = int(dashed.sum() // 255)
    n_t = int(text.sum() // 255)

    short = fname[:25] + '..' if len(fname) > 27 else fname

    axes[i, 0].imshow(img_rgb)
    axes[i, 0].set_title(short, fontsize=7)

    mask_rgb = np.stack([arrows, dashed, text], axis=-1)
    axes[i, 1].imshow(mask_rgb)
    axes[i, 1].set_title(f'Mask R={n_a} G={n_d} B={n_t}', fontsize=6)

    axes[i, 2].imshow(combined, cmap='gray')
    axes[i, 2].set_title('Combined', fontsize=7)

    axes[i, 3].imshow(inpainted_rgb)
    axes[i, 3].set_title('Inpainted', fontsize=7)

    print(f'  {i+1}/{n} {short}: arrows={n_a}, dashed={n_d}, text={n_t}')

for ax in axes.flat:
    ax.axis('off')
plt.suptitle(f'UNet Noise Removal (ep{ep}, loss={loss_val:.4f}) - grid already removed', fontsize=12)
plt.tight_layout()
plt.savefig('test_results.png', dpi=100, bbox_inches='tight')
plt.close()
print(f'\nSaved: test_results.png')
