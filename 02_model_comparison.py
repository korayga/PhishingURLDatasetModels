import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # kontrollu ozellik setini kullanalim (ana model yaklasimi ile uyumlu)
    target_column = "label"
    text_columns = ["FILENAME", "URL", "Domain", "TLD", "Title"]
    suspicious_columns = [
        "URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb",
        "URLCharProb", "DomainTitleMatchScore", "URLTitleMatchScore"
    ]
    
    drop_cols = [c for c in text_columns + suspicious_columns + [target_column] if c in df.columns]
    
    y = df[target_column]
    X = df.drop(columns=drop_cols)
    
    return X, y

def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, class_weight="balanced"),
        "Rule-Based (AdaBoost)": AdaBoostClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Naive Bayes (Bernoulli)": BernoulliNB(),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
        "Deep Learning (Deep MLP)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
    }

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(PHISHING_LABEL)]
    
    y_true_phishing = (y_test == PHISHING_LABEL).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=PHISHING_LABEL, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=PHISHING_LABEL, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=PHISHING_LABEL, zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_true_phishing, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "FPR": fpr,
        "TPR": tpr
    }

def plot_metrics(results, output_dir):
    metrics_to_plot = ["Accuracy", "F1 Score", "ROC AUC"]
    
    df_plot = pd.DataFrame({
        model_name: [res[m] for m in metrics_to_plot]
        for model_name, res in results.items()
    }, index=metrics_to_plot)
    
    df_plot = df_plot.T
    
    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind="bar", width=0.8, figsize=(10, 6))
    plt.title("Model Karsilastirmasi")
    plt.ylabel("Skor")
    plt.ylim(0.9, 1.0)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_metrics_comparison.png", dpi=300)
    plt.close()

def plot_roc_curves(results, output_dir):
    plt.figure(figsize=(8, 6))
    for model_name, res in results.items():
        plt.plot(res["FPR"], res["TPR"], label=f"{model_name} (AUC = {res['ROC AUC']:.4f})")
        
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")
    plt.title("ROC Egrisi Karsilastirmasi")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_comparison.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Farkli ML modelleri ile performans karsilastirmasi.")
    parser.add_argument("--csv", type=str, default="PhiUSIIL_Phishing_URL_Dataset.csv", help="CSV dosyasinin yolu")
    parser.add_argument("--output", type=str, default="outputs_compare", help="Cikti klasoru")
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"veri seti yukleniyor: {csv_path}")
    if not csv_path.exists():
        print(f"hata: {csv_path} bulunamadi!")
        return
        
    X, y = load_and_prepare_data(csv_path)
    
    # %80 train, %20 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = get_models()
    results = {}
    
    print("\nmodeller egitiliyor ve degerlendiriliyor...\n")
    for name, model in models.items():
        print(f"{name} egitiliyor...")
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test)
        results[name] = res
        print(f"{name} -> Accuracy: {res['Accuracy']:.4f}, F1: {res['F1 Score']:.4f}, AUC: {res['ROC AUC']:.4f}")
        
    # sonuclari gorsellestir ve kaydet
    plot_metrics(results, output_dir)
    plot_roc_curves(results, output_dir)
    
    # metrikleri csv olarak kaydet
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            "Model": name,
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1 Score": res["F1 Score"],
            "ROC AUC": res["ROC AUC"]
        })
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / "model_comparison_results.csv", index=False)
    
    print(f"\nkarsilastirma tamamlandi. gorseller ve sonuclar '{output_dir}' klasorune kaydedildi.")
    print("raporunuza 'model_metrics_comparison.png' ve 'roc_curves_comparison.png' ekleyebilirsiniz.")

if __name__ == "__main__":
    main()
