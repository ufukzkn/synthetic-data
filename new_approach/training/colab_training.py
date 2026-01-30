# Curve Segmentation Training - Google Colab Notebook
# 
# Bu notebook'u Google Colab'a yükleyip çalıştırabilirsiniz.
# GPU runtime seçmeyi unutmayın: Runtime > Change runtime type > GPU
#
# Dosyaları yüklemek için:
# 1. synthetic_aircraft_chart.py
# 2. train_curve_segmentation.py 
# 3. (opsiyonel) Gerçek test görselleriniz

# %% [markdown]
# # 🚀 Curve Segmentation Model Training
# 
# Bu notebook, uçak performans grafiklerindeki eğrileri segmente etmek için
# Attention U-Net modelini eğitir.

# %% 
# Setup - Gerekli kütüphaneleri yükle
!pip install -q torch torchvision matplotlib pillow opencv-python-headless tqdm

# %%
# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
# Upload files from your local machine
from google.colab import files

print("📁 Upload synthetic_aircraft_chart.py and train_curve_segmentation.py")
uploaded = files.upload()

# %%
# Test synthetic data generation
from synthetic_aircraft_chart import make_aircraft_chart_sample, generate_dataset
import matplotlib.pyplot as plt
import numpy as np

# Generate a sample
img, mask = make_aircraft_chart_sample(W=512, H=512, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img)
axes[0].set_title("Synthetic Chart")
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Curve Mask")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# %%
# Pre-generate a dataset (optional - increases initial load but faster training)
# generate_dataset("synth_data", n_samples=2000, W=512, H=512)

# %%
# Train the model
from train_curve_segmentation import train, AttentionUNet, DEVICE

model = train(
    epochs=50,           # Başlangıç için 50, sonra artırabilirsiniz
    batch_size=8,        # Colab'da 8-12 arası iyi çalışır
    img_size=512,
    lr=1e-4,
    save_dir="checkpoints",
    samples_per_epoch=500  # Her epoch için 500 örnek
)

# %%
# Test on synthetic image
from train_curve_segmentation import predict
import cv2

# Generate test image
test_img, test_mask = make_aircraft_chart_sample(W=512, H=512, seed=999)
cv2.imwrite("test_input.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))

# Load best model
model = AttentionUNet(in_ch=3, out_ch=1, base_filters=32)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model = model.to(DEVICE)

# Predict
pred_mask = predict(model, "test_input.png", "test_prediction.png")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_img)
axes[0].set_title("Input")
axes[0].axis('off')

axes[1].imshow(test_mask, cmap='gray')
axes[1].set_title("Ground Truth")
axes[1].axis('off')

axes[2].imshow(pred_mask, cmap='gray')
axes[2].set_title("Prediction")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %%
# Test on your real image
# Upload your real chart image
print("📁 Upload your real chart image (test_chart.png)")
uploaded = files.upload()

if uploaded:
    filename = list(uploaded.keys())[0]
    pred = predict(model, filename, "real_prediction.png")
    
    from PIL import Image
    real_img = Image.open(filename)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(real_img)
    axes[0].set_title("Your Image")
    axes[0].axis('off')
    
    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title("Predicted Curves")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
# Download trained model
from google.colab import files

files.download("checkpoints/best_model.pt")
print("✅ Model downloaded!")

# %%
# Full pipeline test: Extract and colorize curves
import cv2
import numpy as np
from scipy import ndimage

def extract_and_colorize_curves(pred_mask, min_curve_size=100):
    """
    Extract individual curves from prediction mask and colorize them.
    """
    # Threshold
    binary = (pred_mask > 127).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Create colored output
    colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))
    colored = np.ones((*binary.shape, 3), dtype=np.uint8) * 255
    
    valid_curves = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area > min_curve_size:
            color = (colors[i][:3] * 255).astype(np.uint8)
            colored[labels == i] = color
            valid_curves += 1
    
    return colored, valid_curves

# Test
test_img, test_mask = make_aircraft_chart_sample(W=512, H=512, seed=123)
cv2.imwrite("final_test.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
pred = predict(model, "final_test.png", "final_pred.png")

colored, n_curves = extract_and_colorize_curves(pred)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_img)
axes[0].set_title("Input Chart")
axes[0].axis('off')

axes[1].imshow(pred, cmap='gray')
axes[1].set_title("Segmentation Mask")
axes[1].axis('off')

axes[2].imshow(colored)
axes[2].set_title(f"Extracted Curves ({n_curves})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("final_result.png", dpi=150)
plt.show()

print(f"✅ Extracted {n_curves} curves!")
