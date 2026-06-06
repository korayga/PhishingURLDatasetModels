import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_noisy_data(X, noise_level=0.1):
    # mevcut veriye rastgele gauss gurultusu ekler
    X_noisy = X.copy()
    for col in X.columns:
        std = X[col].std()
        noise = np.random.normal(0, std * noise_level, size=len(X))
        X_noisy[col] = X_noisy[col] + noise
    return X_noisy

def create_random_data(X):
    # ozelliklerin min-max araliginda tamamen rastgele (anlamsiz) veri uretir
    X_random = X.copy()
    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()
        X_random[col] = np.random.uniform(min_val, max_val, size=len(X))
    return X_random

def main():
    # veri hazirligi
    df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
    
    target_column = "label"
    text_columns = ["FILENAME", "URL", "Domain", "TLD", "Title"]
    suspicious_columns = [
        "URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb",
        "URLCharProb", "DomainTitleMatchScore", "URLTitleMatchScore"
    ]
    drop_cols = [c for c in text_columns + suspicious_columns + [target_column] if c in df.columns]
    
    y = df[target_column]
    X = df.drop(columns=drop_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # model egitimi
    print("random forest modeli egitiliyor...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, class_weight="balanced")
    model.fit(X_train, y_train)
    
    print("\ntest senaryolari")
    # 1. orijinal test verisi
    y_pred_orig = model.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    print(f"1. orijinal test verisi basarisi: %{acc_orig*100:.2f}")
    
    # 2. %10 gurultulu test verisi
    X_noise_10 = create_noisy_data(X_test, noise_level=0.1)
    y_pred_noise_10 = model.predict(X_noise_10)
    acc_noise_10 = accuracy_score(y_test, y_pred_noise_10)
    print(f"2. orijinal veriye %10 gurultu eklenmis basari: %{acc_noise_10*100:.2f}")
    
    # 3. %30 gurultulu test verisi
    X_noise_30 = create_noisy_data(X_test, noise_level=0.3)
    y_pred_noise_30 = model.predict(X_noise_30)
    acc_noise_30 = accuracy_score(y_test, y_pred_noise_30)
    print(f"3. orijinal veriye %30 gurultu eklenmis basari: %{acc_noise_30*100:.2f}")
    
    # 4. tamamen rastgele veri
    X_random = create_random_data(X_test)
    y_pred_random = model.predict(X_random)
    acc_random = accuracy_score(y_test, y_pred_random)
    print(f"4. tamamen rastgele uretilmis yapay veri basarisi: %{acc_random*100:.2f}")
    
    print("\nsonuc incelemesi")
    print("en onemli ozellikler modeli nasil etkiliyor?")
    feature_importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    
    print(feature_importances.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
