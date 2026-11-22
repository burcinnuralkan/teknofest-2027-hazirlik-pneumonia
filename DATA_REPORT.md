# Veri Keşfi (EDA), Temizlik ve Sızıntısız Bölme Raporu

## 1. Veri Seti Genel Yapısı ve Piksel Kalitesi Analizi
* **Toplam İncelenen Görsel:** 5.856 adet (0 bozuk/okunamayan dosya).
* **Çözünürlük ve Kanal Bulguları:** Görseller `384x127` ile `2916x2713` piksel arasında değişken boyutlardadır. 5.573 görsel tek kanallı (`L`), 283 görsel 3 kanallı (`RGB`) formatındadır; ön işlemede tüm görseller standart `RGB` ve `224x224` boyutuna sabitlenmiştir.
* **Piksel Parlaklık ve Kontrast Kontrolü:** Tüm görsellerin piksel yoğunluk ortalamaları (`parlaklik`) ve standart sapmaları (`kontrast`) hesaplanarak veri tablolarına eklenmiş, aşırı karanlık veya tamamen yanmış hatalı çekim bulunmadığı doğrulanmıştır.

## 2. Kopya (Duplicate) Görsel Temizliği
* Görsellerin dijital parmak izleri (`imagehash.phash`) çıkarılarak tekrarlı kayıtlar taranmıştır.
* Aynı parmak izine sahip **62** tekrarlı kayıt elenerek veri seti **5.824** tekil görsele indirilmiştir.

## 3. Hasta Bazlı Sızıntısız Veri Bölme (Data Splitting)
* **Kilitli Test Seti:** Orijinal `test` klasörü (618 görsel) `test_split.csv` olarak ayrılmıştır.
* **Eğitim & Doğrulama Havuzu:** Orijinal `train` ve `val` havuzu birleştirilip `StratifiedGroupKFold (n_splits=5)` ile **%80 Eğitim (4.164 görsel) / %20 Doğrulama (1.042 görsel)** olarak bölünmüştür.
* **Veri Sızıntısı Kontrolü:** Bölme işlemi `hasta_id` bazlı yapıldığı için `train_split.csv` ve `val_split.csv` arasındaki **ortak hasta sayısı 0'dır**. Hata analizi için `bacteria`, `virus` ve `normal` alt türleri `alt_tur` sütununa eklenmiştir.

## 4. Veri Artırımı (Data Augmentation) Kurgusu
* Akciğer anatomisini bozabilecek yatay/dikey çevirme (Flip) uygulanmamış; yalnızca eğitim seti için `±10°` döndürme, `%5` ölçekleme/kaydırma ve `±%15` parlaklık/kontrast değişimi tanımlanmıştır (`augmentation_ornekleri.png`).