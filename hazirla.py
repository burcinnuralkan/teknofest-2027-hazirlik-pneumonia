import os
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from torchvision import transforms

# 1. Bilgisayarındaki 'data' klasörünün içinden train ve test klasörlerini otomatik bul
ana_klasor = None
for kok, klasorler, _ in os.walk("data"):
    if "train" in klasorler and "test" in klasorler:
        ana_klasor = kok
        break

if ana_klasor is None:
    raise FileNotFoundError("data klasörünün içinde train/test klasörleri bulunamadı! Zip dosyasını data içine çıkardığından emin ol.")

print("Veri klasörü bulundu:", ana_klasor)

# 2. Tüm fotoğrafları tara ve meta verileri topla
veriler = []
for set_adi in ["train", "val", "test"]:
    for etiket in ["NORMAL", "PNEUMONIA"]:
        klasor_yolu = os.path.join(ana_klasor, set_adi, etiket)
        if not os.path.exists(klasor_yolu):
            continue
        for dosya in os.listdir(klasor_yolu):
            if dosya.lower().endswith(('.jpeg', '.jpg', '.png')):
                tam_yol = os.path.join(klasor_yolu, dosya)
                with Image.open(tam_yol) as img:
                    genislik, yukseklik = img.size
                    mod = img.mode
                    parmak_izi = str(imagehash.phash(img))
                    gri = np.array(img.convert("L"))
                    parlaklik = round(float(gri.mean()), 2)
                    kontrast = round(float(gri.std()), 2)

                # Hasta ID çıkarımı
                if "person" in dosya.lower():
                    hasta_id = dosya.split("_")[0]
                else:
                    hasta_id = dosya.split("-")[0] + "-" + dosya.split("-")[1]

                # Alt tür çıkarımı
                ad_kucuk = dosya.lower()
                if "bacteria" in ad_kucuk:
                    alt_tur = "bacteria"
                elif "virus" in ad_kucuk:
                    alt_tur = "virus"
                else:
                    alt_tur = "normal"

                veriler.append({
                    "dosya_adi": dosya,
                    "dosya_yolu": tam_yol,
                    "orijinal_set": set_adi,
                    "etiket": etiket,
                    "alt_tur": alt_tur,
                    "hasta_id": hasta_id,
                    "genislik": genislik,
                    "yukseklik": yukseklik,
                    "kanal_modu": mod,
                    "parlaklik": parlaklik,
                    "kontrast": kontrast,
                    "parmak_izi": parmak_izi
                })

df = pd.DataFrame(veriler)
print(f"Toplam taranan fotoğraf: {len(df)}")

# 3. Kopya fotoğrafları temizle
df_temiz = df.drop_duplicates(subset=["parmak_izi"], keep="first").copy()
print(f"Kopyalar silindikten sonra kalan tekil fotoğraf: {len(df_temiz)}")

# 4. Sızıntısız hasta bazlı bölme (StratifiedGroupKFold)
test_df = df_temiz[df_temiz["orijinal_set"] == "test"].reset_index(drop=True)
havuz_df = df_temiz[df_temiz["orijinal_set"].isin(["train", "val"])].reset_index(drop=True)

sgkf = StratifiedGroupKFold(n_splits=5)
for train_idx, val_idx in sgkf.split(havuz_df, y=havuz_df["etiket"], groups=havuz_df["hasta_id"]):
    train_df = havuz_df.iloc[train_idx].reset_index(drop=True)
    val_df = havuz_df.iloc[val_idx].reset_index(drop=True)
    break

ortak = len(set(train_df["hasta_id"]).intersection(set(val_df["hasta_id"])))
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Ortak Hasta: {ortak}")

# 5. CSV dosyalarını doğrudan klasöre kaydet
train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)
print("CSV dosyaları oluşturuldu!")

# 6. Augmentation örneği görselini oluştur ve kaydet
gorsel_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15)
])

ornek_yol = train_df[train_df["etiket"] == "PNEUMONIA"].iloc[0]["dosya_yolu"]
orijinal_img = Image.open(ornek_yol).convert("RGB")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes[0].imshow(orijinal_img.resize((224, 224)))
axes[0].set_title("Orijinal Röntgen")
axes[0].axis("off")

for i in range(1, 5):
    axes[i].imshow(gorsel_aug(orijinal_img))
    axes[i].set_title(f"Augmentation #{i}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("augmentation_ornekleri.png", dpi=150)
print("TÜM İŞLEM TAMAMLANDI! Tüm dosyalar klasörüne kaydedildi.")