# Train ML models for IBS symptom severity prediction based on trigger vectors

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


from src.utils.paths import DATA_INTERIM, REPORT_TBL, REPORT_FIG, PROJ_ROOT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from xgboost import XGBClassifier

from src.utils.paths import DATA_INTERIM, REPORT_TBL, REPORT_FIG


TRIGGER_COLS = ["gluten", "lactose", "caffeine", "highfat", "fodmap"]
RANDOM_SEED = 42


def load_trigger_mapping() -> pd.DataFrame:
    """Load trigger_mapping.csv created in RQ2."""
    path = DATA_INTERIM / "trigger_mapping.csv"
    if not path.exists():
        raise FileNotFoundError("trigger_mapping.csv not found. Run build_trigger_mapping step first.")
    df = pd.read_csv(path)
    return df


def generate_severity_labels(df: pd.DataFrame, samples_per_class: int = 150) -> pd.DataFrame:
    """
    Generate synthetic symptom severity labels for each food class based on trigger count.
    This is a proof-of-concept heuristic, not clinical ground truth.

    Rule of thumb (can be explained in thesis):
      - 0 active triggers  -> 'None'
      - 1-2 active triggers -> 'Mild'
      - 3-5 active triggers -> 'Moderate/Severe'
    A small amount of noise is added to avoid perfectly deterministic labels.
    """
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for _, row in df.iterrows():
        cls = row["class"]
        trigger_vec = row[TRIGGER_COLS].values.astype(int)
        trigger_count = trigger_vec.sum()

        # choose base severity from trigger count
        if trigger_count == 0:
            base = "None"
        elif trigger_count <= 2:
            base = "Mild"
        else:
            base = "Moderate/Severe"

        for _ in range(samples_per_class):
            # add small label noise (e.g. 10% chance to shift one level)
            sev = base
            noise = rng.random()
            if base == "Mild" and noise < 0.05:
                sev = "None"
            elif base == "Mild" and noise > 0.95:
                sev = "Moderate/Severe"
            elif base == "Moderate/Severe" and noise < 0.05:
                sev = "Mild"

            rows.append({
                "class": cls,
                **{k: int(v) for k, v in zip(TRIGGER_COLS, trigger_vec)},
                "severity": sev
            })

    out = pd.DataFrame(rows)
    return out


def train_models(df: pd.DataFrame):
    """Train Logistic Regression, Random Forest, and XGBoost models and compute metrics."""
    X = df[TRIGGER_COLS].values
    y = df["severity"].values

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc
    )

    models = {
        "logreg": LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_SEED
        ),
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED
        )
    }

    metrics = {}
    probas = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        probas[name] = y_proba

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        try:
            # multi-class ROC-AUC (one-vs-rest)
            roc_auc = roc_auc_score(
                label_binarize(y_test, classes=np.arange(len(class_names))),
                y_proba,
                average="macro",
                multi_class="ovo"
            )
        except Exception:
            roc_auc = None

        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            digits=3
        )

        metrics[name] = {
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "roc_auc_macro": float(roc_auc) if roc_auc is not None else None,
            "classification_report": report
        }

    return models, metrics, probas, class_names, (X_test, y_test)


def save_metrics(metrics: dict, class_names):
    REPORT_TBL.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_TBL / "rq3_symptom_metrics.json"
    payload = {
        "class_names": class_names,
        "models": metrics
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("✅ Saved metrics to:", out_path)


def plot_feature_importance(models: dict, class_names):
    """
    Plot feature importance for a tree-based model (prefer XGB, else RF).
    """
    REPORT_FIG.mkdir(parents=True, exist_ok=True)

    if "xgb" in models:
        model = models["xgb"]
        title = "Feature Importance (XGBoost)"
    elif "rf" in models:
        model = models["rf"]
        title = "Feature Importance (Random Forest)"
    else:
        print("No tree-based model found for feature importance plot.")
        return

    importances = model.feature_importances_
    fig = plt.figure(figsize=(6, 4))
    plt.bar(TRIGGER_COLS, importances)
    plt.ylabel("Importance")
    plt.title(title)
    fig.tight_layout()
    out_path = REPORT_FIG / "rq3_feature_importance.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("✅ Saved feature importance figure to:", out_path)


def plot_roc_curves(probas: dict, y_test, class_names):
    """
    Plot macro ROC curves for the best-probability model (prefer XGB, else RF, else LogReg).
    """
    if "xgb" in probas:
        name = "xgb"
    elif "rf" in probas:
        name = "rf"
    else:
        name = "logreg"

    y_proba = probas[name]
    y_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

    fig = plt.figure(figsize=(6, 5))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-class ROC Curves ({name})")
    plt.legend(fontsize=8)
    plt.tight_layout()

    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_FIG / "rq3_roc_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("✅ Saved ROC curves figure to:", out_path)


def save_confusion_matrix(y_true, y_pred, class_names, out_png):
    """Plot and save a normalized confusion matrix."""
    # y_true/y_pred are integer-encoded; labels must be integer range
    labels_int = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels_int)

    # avoid division by zero if a row sum is 0
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(np.float64) / row_sums

    fig = plt.figure(figsize=(5.2, 4.6))
    ax = plt.gca()
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # 1) Load trigger mapping
    df_trig = load_trigger_mapping()

    # 2) Generate synthetic severity dataset
    df = generate_severity_labels(df_trig, samples_per_class=150)
    print(f"Dataset shape: {df.shape}")
    print(df["severity"].value_counts())

    # 3) Train models and compute metrics
    models, metrics, probas, class_names, (X_test, y_test) = train_models(df)

    # 3.1) Persist XGBoost for pipeline inference
    (PROJ_ROOT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(models["xgb"], PROJ_ROOT / "models" / "xgb_symptom_model.pkl")
    print("✅ Saved XGBoost model to models/xgb_symptom_model.pkl")

    # 4) Save metrics to JSON
    save_metrics(metrics, class_names)

    # 5) Plots: feature importance + ROC curves
    plot_feature_importance(models, class_names)
    plot_roc_curves(probas, y_test, class_names)

    # 6) Classification report + confusion matrix per model
    for name, model in models.items():
        y_pred = model.predict(X_test)

        # y_true/y_pred integer; labels must be integer range; names are class_names
        report_txt = classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        (REPORT_TBL / f"rq3_classification_report_{name}.txt").write_text(report_txt)

        save_confusion_matrix(
            y_test,
            y_pred,
            class_names,
            REPORT_FIG / f"rq3_confusion_matrix_{name}.png",
        )
    print("✅ Saved all metrics, reports, and confusion matrices successfully.")
    
if __name__ == "__main__":
    # Run the full RQ3 pipeline (do not re-implement logic here)
    main()