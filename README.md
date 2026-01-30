# F-18 Performans Grafiği Eğri Çıkarıcı (Curve Extractor)

## 🎯 Amaç
Taranmış/fotokopisi çekilmiş F-18 performans grafiklerinden eğrileri otomatik olarak çıkaran bir U-Net modeli eğitmek.

**Problem:** Gerçek eğitim verisi yok → Sentetik veri üretiyoruz!

---

## ✅ KULLANILAN DOSYALAR

### 📓 `colab_training.ipynb` - ANA DOSYA
**Google Colab'da çalıştır!**

Bu notebook her şeyi içeriyor:
- Sentetik grafik üretici (matplotlib ile)
- U-Net segmentasyon modeli
- Eğitim döngüsü (Dice + BCE loss)
- Çıkarım (inference) kodu

**Özellikler:**
- Rastgele eğriler (peak, rising, falling, step fonksiyonları)
- Oklar (eğri üzerinde rastgele konumlarda)
- Grid çizgileri, text box'lar, envelope'lar
- Tarama artifact'leri (rotasyon, gürültü, JPEG sıkıştırma)
- Data augmentation (flip, brightness, contrast)

**Kullanım:**
1. Colab'a yükle
2. Runtime → Run all
3. Model eğitilecek ve `curve_unet.pt` kaydedilecek

---

### 🖼️ `input_plot.png`
Test için kullanılan gerçek grafik görüntüsü.

### 🧪 `test_single_image.py`
Eğitilmiş modeli gerçek görüntü üzerinde test etmek için script.

---

## ⚠️ ESKİ DENEMELERDEN KALAN DOSYALAR (BAKMA!)

Aşağıdaki dosyalar/klasörler eski denemelerden kaldı, aktif olarak kullanılmıyor:

```
new_approach/          → Önceki yaklaşım denemeleri
backup_original/       → Yedek dosyalar
__pycache__/          → Python cache
.vscode/              → VS Code ayarları
generated_images/     → Test için üretilen örnek görseller
*_output.png          → Çeşitli test çıktıları
```

---

## 🚀 Hızlı Başlangıç

```python
# 1. Colab'da notebook'u aç ve çalıştır
# 2. Eğitim bitince modeli indir (curve_unet.pt)
# 3. Gerçek görüntüde test et:
python test_single_image.py
```

---

## 📊 Model Mimarisi

```
U-Net (encoder-decoder with skip connections)
├── Encoder: 64 → 128 → 256 → 512
├── Bottleneck: 1024
└── Decoder: 512 → 256 → 128 → 64 → 1 (sigmoid)
```

**Loss:** Dice Loss + Binary Cross Entropy (combined)

---

## 🎨 Sentetik Veri Özellikleri

| Özellik | Değer Aralığı |
|---------|---------------|
| Eğri sayısı | 1-6 |
| Eğri kalınlığı | 0.3-0.6 |
| Ok sayısı | 0-3 |
| Grid rengi | Açık mavi/yeşil/gri |
| Rotasyon | ±2° |
| Gürültü | Gaussian + salt-pepper |

---

## 📝 Notlar

- GPU olmadan eğitim çok yavaş, Colab'ın ücretsiz GPU'sunu kullan
- `num_workers=0` ayarı Colab crash'lerini önlüyor
- Epoch sayısını artırarak daha iyi sonuç alabilirsin (default: 50)

---

*Son güncelleme: Ocak 2026*
