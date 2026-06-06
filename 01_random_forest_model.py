
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


PHISHING_LABEL = 0
LEGITIMATE_LABEL = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="PhiUSIIL Phishing URL veri seti için Random Forest deneyleri"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="PhiUSIIL_Phishing_URL_Dataset.csv path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Sonuçların kaydedileceği klasör",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Random Forest ağaç sayısı default 100",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Tekrarlanabilir sonuç için random_state default 42",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Veri setinde 'label' isimli hedef sütun bulunamadı.")

    return df


def print_dataset_info(df):
    print(f"\nveri boyutu: {df.shape}")
    print(f"eksik deger sayisi: {df.isnull().sum().sum()}")
    print("\nsinif dagilimi:")
    print(df["label"].value_counts())
    
    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    print("\nmetinsel sutunlar:", object_columns)


def prepare_feature_sets(df):
    target_column = "label"

    text_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # hedef bilgisi sizdirma riski tasiyan sutunlar
    suspicious_columns = [
        "URLSimilarityIndex",
        "CharContinuationRate",
        "TLDLegitimateProb",
        "URLCharProb",
        "DomainTitleMatchScore",
        "URLTitleMatchScore",
    ]
    suspicious_columns = [col for col in suspicious_columns if col in df.columns]

    y = df[target_column]

    # tum sayisal ozellikler (metin sutunlari cikarilmis)
    X_all = df.drop(columns=text_columns + [target_column])

    # kontrollu ozellik seti (metin + supheli sutunlar cikarilmis)
    X_controlled = df.drop(columns=text_columns + suspicious_columns + [target_column])

    feature_sets = {
        "all_features": {
            "display_name": "Random Forest - Tum Sayisal Ozellikler",
            "X": X_all,
            "dropped_columns": text_columns,
        },
        "controlled_features": {
            "display_name": "Random Forest - Kontrollu Ozellik Seti",
            "X": X_controlled,
            "dropped_columns": text_columns + suspicious_columns,
        },
    }

    print(f"\ntum sayisal ozellikler: {X_all.shape}")
    print(f"kontrollu ozellik seti: {X_controlled.shape}")

    return feature_sets, y


def train_val_test_split(X, y, random_state):
    # veri setini 70 train, 15 val, 15 test olarak boluyoruz. oranlari stratify ile koruyoruz.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(model, X_data, y_true):
    y_pred = model.predict(X_data)
    y_proba = model.predict_proba(X_data) 

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[PHISHING_LABEL, LEGITIMATE_LABEL],
    )

    accuracy = accuracy_score(y_true, y_pred)

    phishing_precision = precision_score(
        y_true,
        y_pred,
        pos_label=PHISHING_LABEL,
        zero_division=0,
    )
    phishing_recall = recall_score(
        y_true,
        y_pred,
        pos_label=PHISHING_LABEL,
        zero_division=0,
    )
    phishing_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=PHISHING_LABEL,
        zero_division=0,
    )

    # confusion matrix: tp, fn, fp, tn ayiklama
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    phishing_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    phishing_class_index = list(model.classes_).index(PHISHING_LABEL)
    phishing_score = y_proba[:, phishing_class_index]
    y_true_phishing = (y_true == PHISHING_LABEL).astype(int)

    fpr, tpr, thresholds = roc_curve(y_true_phishing, phishing_score)
    roc_auc = auc(fpr, tpr)

    report = classification_report(
        y_true,
        y_pred,
        labels=[PHISHING_LABEL, LEGITIMATE_LABEL],
        target_names=["Phishing", "Legitimate"],
        digits=6,
        zero_division=0,
    )

    metrics = {
        "Accuracy": accuracy,
        "Phishing Precision": phishing_precision,
        "Phishing Recall / Sensitivity": phishing_recall,
        "Phishing Specificity": phishing_specificity,
        "Phishing F1-score": phishing_f1,
        "Phishing ROC-AUC": roc_auc,
        "TP - Phishing Dogru": tp,
        "FN - Phishing Kacirilan": fn,
        "FP - Legitimate Yanlis Alarm": fp,
        "TN - Legitimate Dogru": tn,
        "Confusion Matrix": cm,
        "Classification Report": report,
        "FPR": fpr,
        "TPR": tpr,
    }

    return metrics


def train_and_evaluate(feature_key, feature_info, y, args, output_dir):
    display_name = feature_info["display_name"]
    X = feature_info["X"]

    print(f"\n{display_name}")
    print(f"ozellik matrisi boyutu: {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y,
        random_state=args.random_state,
    )

    print(f"train: {X_train.shape}")
    print(f"validation: {X_val.shape}")
    print(f"test: {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    val_metrics = calculate_metrics(model, X_val, y_val)
    test_metrics = calculate_metrics(model, X_test, y_test)

    print("\nvalidation metrikleri:")
    print_metrics_short(val_metrics)

    print("\ntest metrikleri:")
    print_metrics_short(test_metrics)

    save_model_artifacts(
        feature_key=feature_key,
        display_name=display_name,
        model=model,
        X_columns=X.columns.tolist(),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        output_dir=output_dir,
    )

    return {
        "feature_key": feature_key,
        "display_name": display_name,
        "model": model,
        "feature_names": X.columns.tolist(),
        "dropped_columns": feature_info["dropped_columns"],
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def print_metrics_short(metrics):
    keys = [
        "Accuracy",
        "Phishing Precision",
        "Phishing Recall / Sensitivity",
        "Phishing Specificity",
        "Phishing F1-score",
        "Phishing ROC-AUC",
    ]

    for key in keys:
        print(f"{key}: {metrics[key]:.6f}")


def save_model_artifacts(
    feature_key,
    display_name,
    model,
    X_columns,
    val_metrics,
    test_metrics,
    output_dir,
):
    model_dir = output_dir / feature_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # model kaydet
    joblib.dump(model, model_dir / "random_forest_model.joblib")

    # feature importance csv
    feature_importance = pd.DataFrame(
        {
            "Feature": X_columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    feature_importance.to_csv(
        model_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # feature importance png
    save_feature_importance_plot(
        feature_importance,
        title=f"{display_name}\nFeature Importance",
        path=model_dir / "feature_importance_top20.png",
        top_n=min(20, len(feature_importance)),
    )

    # confusion matrix png
    save_confusion_matrix_plot(
        test_metrics["Confusion Matrix"],
        title=f"{display_name}\nFinal Test Confusion Matrix",
        path=model_dir / "confusion_matrix_final_test.png",
    )

    # roc curve png
    save_roc_curve_plot(
        test_metrics["FPR"],
        test_metrics["TPR"],
        test_metrics["Phishing ROC-AUC"],
        title=f"{display_name}\nFinal Test ROC Curve",
        path=model_dir / "roc_curve_final_test.png",
    )

    # classification report txt
    with open(model_dir / "classification_report_final_test.txt", "w", encoding="utf-8") as f:
        f.write(test_metrics["Classification Report"])

    # json metrik
    serializable = {
        "display_name": display_name,
        "validation_metrics": clean_metrics_for_json(val_metrics),
        "test_metrics": clean_metrics_for_json(test_metrics),
    }

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)


def clean_metrics_for_json(metrics):
    cleaned = {}
    skip_keys = {"Confusion Matrix", "Classification Report", "FPR", "TPR"}

    for key, value in metrics.items():
        if key in skip_keys:
            continue

        if isinstance(value, (np.integer, np.floating)):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value

    cleaned["Confusion Matrix"] = metrics["Confusion Matrix"].tolist()
    return cleaned


def save_feature_importance_plot(feature_importance, title, path, top_n=20):
    top_features = feature_importance.head(top_n).sort_values(
        by="Importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_confusion_matrix_plot(cm, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Tahmin Edilen Sinif")
    plt.ylabel("Gercek Sinif")
    plt.xticks([0, 1], ["Phishing", "Legitimate"])
    plt.yticks([0, 1], ["Phishing", "Legitimate"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_roc_curve_plot(fpr, tpr, roc_auc, title, path):
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Rastgele siniflandirici")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def create_summary_tables(results, output_dir):
    rows = []
    confusion_rows = []

    for result in results:
        model_name = result["display_name"]

        for split_name, metrics in [
            ("Validation", result["validation_metrics"]),
            ("Final Test", result["test_metrics"]),
        ]:
            rows.append(
                {
                    "Model": model_name,
                    "Split": split_name,
                    "Accuracy": metrics["Accuracy"],
                    "Phishing Precision": metrics["Phishing Precision"],
                    "Phishing Recall / Sensitivity": metrics["Phishing Recall / Sensitivity"],
                    "Phishing Specificity": metrics["Phishing Specificity"],
                    "Phishing F1-score": metrics["Phishing F1-score"],
                    "Phishing ROC-AUC": metrics["Phishing ROC-AUC"],
                }
            )

        test_metrics = result["test_metrics"]
        confusion_rows.append(
            {
                "Model": model_name,
                "TP - Phishing Dogru": test_metrics["TP - Phishing Dogru"],
                "FN - Phishing Kacirilan": test_metrics["FN - Phishing Kacirilan"],
                "FP - Legitimate Yanlis Alarm": test_metrics["FP - Legitimate Yanlis Alarm"],
                "TN - Legitimate Dogru": test_metrics["TN - Legitimate Dogru"],
            }
        )

    summary_df = pd.DataFrame(rows)
    confusion_df = pd.DataFrame(confusion_rows)

    summary_df.to_csv(
        output_dir / "model_comparison_validation_final_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_percent_df = summary_df.copy()
    metric_cols = [
        "Accuracy",
        "Phishing Precision",
        "Phishing Recall / Sensitivity",
        "Phishing Specificity",
        "Phishing F1-score",
        "Phishing ROC-AUC",
    ]
    summary_percent_df[metric_cols] = summary_percent_df[metric_cols] * 100
    summary_percent_df.to_csv(
        output_dir / "model_comparison_percent.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df.to_csv(
        output_dir / "confusion_summary_final_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_model_comparison_plot(summary_percent_df, output_dir / "model_comparison_percent.png")
    save_combined_roc_plot(results, output_dir / "roc_curves_final_test_comparison.png")

    return summary_df, summary_percent_df, confusion_df


def save_model_comparison_plot(summary_percent_df, path):
    final_test_df = summary_percent_df[summary_percent_df["Split"] == "Final Test"].copy()

    metric_cols = [
        "Accuracy",
        "Phishing Precision",
        "Phishing Recall / Sensitivity",
        "Phishing Specificity",
        "Phishing F1-score",
        "Phishing ROC-AUC",
    ]

    x = np.arange(len(metric_cols))
    width = 0.35

    plt.figure(figsize=(14, 7))

    for i, (_, row) in enumerate(final_test_df.iterrows()):
        values = [row[col] for col in metric_cols]
        plt.bar(x + i * width, values, width, label=row["Model"])

    plt.xticks(x + width / 2, metric_cols, rotation=25, ha="right")
    plt.ylabel("Basari Orani (%)")
    plt.title("Final Test Model Karsilastirmasi")
    plt.ylim(99, 100.1)
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_combined_roc_plot(results, path):
    plt.figure(figsize=(8, 6))

    for result in results:
        metrics = result["test_metrics"]
        plt.plot(
            metrics["FPR"],
            metrics["TPR"],
            label=f"{result['display_name']} AUC={metrics['Phishing ROC-AUC']:.6f}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Rastgele siniflandirici")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Test ROC Egrileri Karsilastirmasi")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def select_main_model(results):
    if not results:
        raise ValueError("Ana model secimi icin sonuc listesi bos olamaz.")

    # ana modeli dogrulama performansina gore seciyoruz
    # esitlikte F1, ROC-AUC ve Accuracy sirasi ile karar veriyoruz
    return max(
        results,
        key=lambda r: (
            r["validation_metrics"]["Phishing F1-score"],
            r["validation_metrics"]["Phishing ROC-AUC"],
            r["validation_metrics"]["Accuracy"],
        ),
    )


def save_project_notes(main_model, output_dir):
    # proje icin kucuk notlar
    notes = f"secilen ana model: {main_model['display_name']}\n"
    with open(output_dir / "project_notes.txt", "w", encoding="utf-8") as f:
        f.write(notes)


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output)

    df = load_dataset(args.csv)
    print_dataset_info(df)

    feature_sets, y = prepare_feature_sets(df)

    results = []
    for feature_key, feature_info in feature_sets.items():
        result = train_and_evaluate(
            feature_key=feature_key,
            feature_info=feature_info,
            y=y,
            args=args,
            output_dir=output_dir,
        )
        results.append(result)

    summary_df, summary_percent_df, confusion_df = create_summary_tables(
        results,
        output_dir,
    )

    main_model = select_main_model(results)
    save_project_notes(main_model, output_dir)

    print("\ngenel karsilastirma:")
    print(summary_df.to_string(index=False))

    print("\ngenel karsilastirma (yuzde):")
    print(summary_percent_df.round(4).to_string(index=False))

    print("\nconfusion matris ozeti:")
    print(confusion_df.to_string(index=False))

    print("\nonerilen ana model:")
    print(main_model["display_name"])

    print(f"\nciktilar {output_dir.resolve()} klasorune kaydedildi.")


if __name__ == "__main__":
    main()
