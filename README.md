# 🛡️ Oltalama (Phishing) Web Sitelerinin Tespiti — Veri Madenciliği Projesi


## 📌 Proje Özeti

Bu projede, kullanıcıları dolandırmayı hedefleyen **sahte (Phishing) web siteleri** ile **gerçek (Legitimate) web sitelerini** makine öğrenmesi algoritmaları kullanarak otomatik ve yüksek doğrulukla ayırt edebilen modeller geliştirilmiştir.

**Ana algoritma:** Random Forest  
**Veri seti:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI Machine Learning Repository)

**Sunum Videosu (YouTube):** [Random Forest ile Phishing Web Sitelerinin Tespiti](https://www.youtube.com/watch?v=yq_4fRWAo_8)




---

## İçindekiler

1. [Projenin Amacı ve Önemi](#projenin-amacı-ve-önemi)
2. [Veri Seti Bilgileri](#veri-seti-bilgileri)
3. [Yöntem ve İş Akışı](#yöntem-ve-iş-akışı)
4. [Deneysel Sonuçlar](#deneysel-sonuçlar)
5. [Dayanıklılık (Robustness) Testleri](#dayanıklılık-robustness-testleri)
6. [Proje Yapısı](#proje-yapısı)
7. [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
8. [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
9. [Referanslar](#referanslar)

---

## Projenin Amacı ve Önemi

Siber savunma sistemlerinde oltalama sitelerinin tespiti yapılırken **False Negative** (phishing sitesinin güvenli/meşru olarak sınıflandırılması) hatasının maliyeti çok yüksektir; çünkü bu durum doğrudan kullanıcı verilerinin sızdırılmasına veya finansal kayıplara yol açar. 

Bu çalışmada, yalnızca yüksek doğruluklu bir model eğitilmemiş; aynı zamanda siber saldırganların tespit sistemlerini atlatmak için web site özniteliklerinde yapabileceği manipülasyonlar (adversarial noise) simüle edilerek modelin savunma hattındaki dayanıklılığı test edilmiştir.

---

## Veri Seti Bilgileri

| Özellik | Değer / Açıklama |
| :--- | :--- |
| **Veri Seti Adı** | PhiUSIIL Phishing URL Dataset |
| **Kaynak** | UCI Machine Learning Repository |
| **Toplam Kayıt Sayısı** | ~235.000 URL |
| **Öznitelik Sayısı** | 54+ Yapısal, URL tabanlı ve istatistiksel nitelik |
| **Hedef Değişken (label)** | 0 = Phishing (Oltalama), 1 = Legitimate (Güvenli) |
| **Eksik Değer Durumu** | Yok (Ön işleme aşamasında temizlenmiştir) |

---

## Yöntem ve iş Akışı

Proje, verinin ham halinden alınıp savunma sistemine entegre edilebilir bir kararlılığa ulaştırılmasına kadar 7 temel aşamadan oluşmaktadır:

```	ext
┌─────────────────────────┐
│   1. VERİ YÜKLEME       │ -> CSV dosyasının DataFrame olarak yüklenmesi
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│   2. VERİ ÖN İŞLEME      │ -> Metin sütunlarının çıkarılması, ölçekleme
└──────────┬──────────────┘ -> Train / Validation / Test (%70, %15, %15) ayrımı
           ▼
┌─────────────────────────┐
│   3. RF MODEL EĞİTİMİ   │ -> Tüm özellikler ve kontrollü özellik setleri
└──────────┬──────────────┘ -> ile Random Forest modellerinin kurulması
           ▼
┌─────────────────────────┐
│   4. DEĞERLENDİRME      │ -> Confusion Matrix, ROC-AUC hesaplamaları
└──────────┬──────────────┘ -> Feature Importance (Özellik Önem) analizleri
           ▼
┌─────────────────────────┐
│   5. MODEL KIYASLAMA    │ -> DT, RF, AdaBoost, KNN, Gaussian NB, 
└──────────┬──────────────┘ -> Bernoulli NB, MLP, Deep MLP
           ▼
┌─────────────────────────┐
│   6. DAYANIKLILIK TESTİ │ -> %10 ve %30 Gauss gürültüsü eklenmesi,
└──────────┬──────────────┘ -> Rastgele veriyle ezber (overfitting) kontrolü
           ▼
┌─────────────────────────┐
│   7. SONUÇ & RAPORLAMA  │ -> Grafikler, çıktılar ve akademik raporlama
└─────────────────────────┘
```

## Deneysel Sonuçlar

### 1. Random Forest Öznitelik Seti Karşılaştırması

| Özellik Seti | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Tüm Sayısal Özellikler** | ~%99.9 | ~%99.9 | ~%99.9 |
| **Kontrollü Özellik Seti** | ~%99.8 | ~%99.8 | ~%99.9 |

### 2. Çoklu Algoritma Kıyaslaması (Kapsamlı Analiz)
Veri seti 9 farklı makine öğrenmesi ve yapay sinir ağı mimarisiyle eğitilmiş olup, elde edilen kalitatif performans özeti aşağıdadır:

| Algoritma Grubu | Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Ensemble Learning** | **Random Forest** | **En Yüksek** | **En Yüksek** | **En Yüksek** |
| **Ensemble Learning** | AdaBoost | Yüksek | Yüksek | Yüksek |
| **Ağaç Tabanlı Sınıflandırıcı**| Decision Tree | Yüksek | Yüksek | Yüksek |
| **Yapay Sinir Ağları** | Deep MLP & MLP | Yüksek | Yüksek | Yüksek |
| **Mesafe Tabanlı Modeller** | K-Nearest Neighbors (KNN)| Yüksek | Yüksek | Yüksek |
| **Olasılıksal Modeller** | Gaussian & Bernoulli NB | Orta | Orta | Orta |

*Not: Kesin ondalıklı değerler, proje çalıştırıldığında üretime bağlı olarak outputs_compare/ klasöründe metrik bazlı olarak tutulmaktadır.*

## Dayanıklılık (Robustness) Testleri
Siber güvenlik modellerinin kararlılığını ölçmek adına, eğitilen en başarılı modele 3 farklı stres ve manipülasyon senaryosu uygulanmıştır:

| Test Senaryosu | Uygulanan Değişim | Beklenen / Elde Edilen Model Refleksi |
| :--- | :--- | :--- |
| **Orijinal Test Verisi** | Değişim yok | ~%99.8 Doğruluk oranı ile kararlı koruma. |
| **%10 Gauss Gürültüsü** | Özniteliklere hafif gürültü eklenmesi | Hafif düşüş; model gürültüyü absorbe edebiliyor (Sağlam). |
| **%30 Gauss Gürültüsü** | Yüksek oranda veri manipülasyonu | Belirgin düşüş; modelin yapısal sınırları doğrulanıyor. |
| **Tamamen Rastgele Veri**| Anlamsız/Rastgele matris girdisi | ~%50 Doğruluk (Modelin ezber yapmadığı, rastgele veriye yazı-tura cevabı verdiği kanıtlanmıştır). |

## Proje Yapısı

```plaintext
PhishingURLDatasetModels/
│
├── 📓 sunum_notebook.ipynb            # Adım adım yürütülen ana Jupyter sunum dosyası
│
├── 🐍 01_random_forest_model.py       # RF modeli eğitimi (3 farklı özellik seti ile)
├── 🐍 02_model_comparison.py          # 9 farklı ML modelinin karşılaştırılması
├── 🐍 03_robustness_test.py           # Model dayanıklılık ve gürültü testleri
│
├── 📂 outputs/                        # Random Forest analiz çıktıları (.json & grafikler)
│   ├── all_features/                  # Tüm özelliklerle üretilen sonuçlar
│   └── controlled_features/           # Seçilmiş özelliklerle üretilen sonuçlar
│ 
├── 📂 outputs_compare/                # 9 Modelin kıyaslama grafikleri ve CSV tabloları
│
├── 📄 PhiUSIIL_Phishing_URL_Dataset.csv # Orijinal veri seti dosyası
├── 📄 requirements.txt                # Bağımlılık listesi
├── 📄 README.md                       # Proje dokümantasyonu
└── 📄 BLM463_Proje_KorayGarip_22360859088.docx # Detaylı akademik rapor
```

## Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin
Projenin çalışması için gerekli kütüphaneleri yüklemek adına terminalde aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

### 2. Sunum Notebook'u (Önerilen Çalıştırma Yöntemi)
Projeyi görsel grafiklerle, adım adım izlemek ve sunumunu gerçekleştirmek için Jupyter ortamını başlatabilirsiniz:

```bash
jupyter notebook sunum_notebook.ipynb
```

### 3. Modülleri Script Üzerinden Çalıştırma
Her bir analiz adımını bağımsız Python scriptleri olarak koşturmak isterseniz:

```bash
# 1. Ana Random Forest modelini eğitin ve çıktıları kaydedin
python 01_random_forest_model.py --csv PhiUSIIL_Phishing_URL_Dataset.csv --output outputs

# 2. 8 farklı modeli birbiriyle kıyaslayın
python 02_model_comparison.py --csv PhiUSIIL_Phishing_URL_Dataset.csv --output outputs_compare

# 3. Eğitilen modelin gürültü ve dayanıklılık testlerini simüle edin
python 03_robustness_test.py
```

## Kullanılan Teknolojiler

- **Programlama Dili:** Python 3.10+
- **Veri İşleme & Analiz:** pandas, 
umpy
- **Makine Öğrenmesi & YSA:** scikit-learn (Ensemble, Tree, Neighbors, Naive Bayes, Neural Network modülleri)
- **Görselleştirme:** matplotlib, seaborn
- **Geliştirme Ortamları:** Jupyter Notebook, Python CLI

## Referanslar

- **Veri Seti:** Abutaha, M. et al. (2024). *PhiUSIIL Phishing URL Dataset*. UCI Machine Learning Repository.
- **Kütüphane:** Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
- **Teorik Altyapı:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

## Geliştirici
**LinkedIn:** [Koray Garip | LinkedIn](https://www.linkedin.com/in/koray-garip/)
