# 🛡️ Oltalama (Phishing) Web Sitelerinin Tespiti — Veri Madenciliği Projesi


## 📌 Proje Özeti

Bu projede, kullanıcıları dolandırmayı hedefleyen **sahte (Phishing) web siteleri** ile **gerçek (Legitimate) web sitelerini** makine öğrenmesi algoritmaları kullanarak otomatik ve yüksek doğrulukla ayırt edebilen modeller geliştirilmiştir.

**Ana algoritma:** Random Forest  
**Veri seti:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI Machine Learning Repository)

---

## 📊 Veri Seti Bilgileri

| Özellik | Değer |
|---|---|
| **Dosya Adı** | `PhiUSIIL_Phishing_URL_Dataset.csv` |
| **Toplam Kayıt** | ~235.000 URL |
| **Özellik Sayısı** | 54+ sütun |
| **Hedef Değişken** | `label` (0 = Phishing, 1 = Legitimate) |
| **Kaynak** | UCI ML Repository / Kaggle |

---

## 📁 Proje Yapısı

```
veri_madenciliği/
│
├── 📓 sunum_notebook.ipynb          # Ana sunum notebook'u (adım adım çalıştırılır)
│
├── 🐍 01_random_forest_model.py     # Random Forest — 3 farklı özellik seti ile eğitim
├── 🐍 02_model_comparison.py        # 9 farklı ML modeli karşılaştırması
├── 🐍 03_robustness_test.py         # Gürültülü ve rastgele veri ile dayanıklılık testi
│
├── 📂 outputs/                      # Random Forest model çıktıları
│   ├── all_features/                #   ├── Tüm özellikler ile sonuçlar
│   └──  controlled_features/         #  ├── Kontrollü özellik seti sonuçları
│ 
│
├── 📂 outputs_compare/              # Model karşılaştırma çıktıları (grafik + CSV)
│
├── 📄 PhiUSIIL_Phishing_URL_Dataset.csv 
├── 📄 requirements.txt            
├── 📄 README.md                  
└── 📄 BLM463_Proje_KorayGarip_22360859088.docx  
```

---

## 🔄 Proje Akış Diyagramı

```
┌─────────────────────────┐
│   1. VERİ YÜKLEME       │
│   CSV → DataFrame       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   2. VERİ ÖN İŞLEME      │
│   • Metin sütunları çıkar│
│   • özellik seti oluştur │
│   • Train/Val/Test böl   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   3. MODEL EĞİTİMİ      │
│   Random Forest         │
│   • Tüm özellikler      │
│   • Kontrollü set       │         
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   4. DEĞERLENDİRME      │
│   • Confusion Matrix    │
│   • ROC-AUC             │
│   • Feature Importance  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   5. MODEL KARŞILAŞTIRMA │
│   9 farklı ML algoritması│
│   DT, RF, KNN, NB,      │
│   SVM, MLP, AdaBoost    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   6. DAYANIKLILIK TESTİ │
│   • %10 / %30 gürültü   │
│   • Rastgele veri       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   7. SONUÇ & RAPORLAMA  │
│   Grafik, tablo, rapor  │
└─────────────────────────┘
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. Sunum Notebook'u (Önerilen)

Tüm projeyi tek bir notebook üzerinden adım adım çalıştırıp sunabilirsiniz:

```bash
jupyter notebook sunum_notebook.ipynb
```

### 3. Python Scriptleri (Ayrı Ayrı Çalıştırma)

```bash
# Ana Random Forest modeli 
python 01_random_forest_model.py --csv PhiUSIIL_Phishing_URL_Dataset.csv --output outputs

# Model karşılaştırması
python 02_model_comparison.py --csv PhiUSIIL_Phishing_URL_Dataset.csv --output outputs_compare

# Dayanıklılık testi (gürültülü + rastgele veri)
python 03_robustness_test.py

```

---

## 📈 Sonuçlar Özeti

### Random Forest — Özellik Seti Karşılaştırması

| Özellik Seti | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Tüm Sayısal Özellikler | ~%99.9 | ~%99.9 | ~%99.9 |
| Kontrollü Özellik Seti | ~%99.8 | ~%99.8 | ~%99.9 |


### Çok Modelli Karşılaştırma

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Random Forest** | **En Yüksek** | **En Yüksek** | **En Yüksek** |
| Decision Tree | Yüksek | Yüksek | Yüksek |
| Neural Network (MLP) | Yüksek | Yüksek | Yüksek |
| K-Nearest Neighbors | Yüksek | Yüksek | Yüksek |
| Naive Bayes | Orta | Orta | Orta |


> **Not:** Kesin değerler notebook çalıştırıldığında hesaplanır ve gösterilir.

---

## 🧪 Dayanıklılık Test Sonuçları

| Test Senaryosu | Beklenen Sonuç |
|---|---|
| Orijinal test verisi | ~%99.8 doğruluk |
| %10 gürültü eklenmiş | Hafif düşüş (model sağlam) |
| %30 gürültü eklenmiş | Belirgin düşüş (beklenen) |
| Tamamen rastgele veri | ~%50 (model ezber yapmıyor) |


---

## 📚 Dosya Açıklamaları

### `01_random_forest_model.py`
Ana model scripti. Veri setini yükler, 3 farklı özellik seti hazırlar ve her biri için Random Forest modeli eğitir. Confusion matrix, ROC eğrisi, feature importance grafikleri ve metrik JSON dosyalarını `outputs/` klasörüne kaydeder.

### `02_model_comparison.py`
9 farklı makine öğrenmesi algoritmasını (Decision Tree, Random Forest, AdaBoost, KNN, Gaussian NB, Bernoulli NB, MLP, Deep MLP, SVM) aynı veri seti üzerinde eğitip karşılaştırır. Sonuçları `outputs_compare/` klasörüne kaydeder.

### `03_robustness_test.py`
Eğitilmiş modelin farklı gürültü seviyelerine dayanıklılığını test eder. %10 ve %30 Gauss gürültüsü ile tamamen rastgele üretilmiş veri kullanarak modelin ezbere dayalı olmadığını doğrular.

### `sunum_notebook.ipynb`
Tüm projeyi 7 bölümde, adım adım çalıştırılabilir şekilde sunan Jupyter Notebook. Sunumda hücre hücre ilerleyerek anlatılacak şekilde tasarlanmıştır.

---

## 🔗 Referanslar

- **Veri Seti:** Abutaha, M. et al. (2024). PhiUSIIL Phishing URL Dataset. UCI Machine Learning Repository.
- **Scikit-learn:** Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
- **Random Forest:** Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

---


