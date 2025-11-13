# Explain XGBoost symptom-severity model with SHAP

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.utils.paths import PROJ_ROOT, DATA_INTERIM, REPORT_FIG, REPORT_TBL

MODEL_PATH = PROJ_ROOT / "models" / "xgb_symptom_model.pkl"
TRIGGER_CSV = DATA_INTERIM / "trigger_mapping.csv"

SEVERITY_LABELS = {0: "Mild", 1: "Moderate/Severe", 2: "None"}
FEATURES = ["gluten", "lactose", "caffeine", "highfat", "fodmap"]


def load_model_and_data(n_samples: int = 800) -> tuple[pd.DataFrame, object, np.ndarray]:
    """Load trained XGB model + build an input matrix X for SHAP.
    We replicate trigger patterns to get enough samples for stable SHAP plots.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train RQ3 and save the model first.")

    model = joblib.load(MODEL_PATH)

    if not TRIGGER_CSV.exists():
        raise FileNotFoundError(f"Trigger mapping CSV not found: {TRIGGER_CSV}")

    base = pd.read_csv(TRIGGER_CSV)  # columns: class, gluten..fodmap
    X_base = base[FEATURES].copy()

    # Repeat/replicate to reach desired sample size (binary features, so repetition is OK)
    reps = int(np.ceil(n_samples / len(X_base)))
    X = pd.concat([X_base] * reps, ignore_index=True).iloc[:n_samples, :].reset_index(drop=True)

    return X, model, np.array([SEVERITY_LABELS[k] for k in getattr(model, "classes_", [0,1,2])], dtype=object)


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=800, help="Number of synthetic rows for SHAP (default: 800)")
    args = ap.parse_args()

    # 1) Load model + input matrix
    X, model, class_names = load_model_and_data(args.samples)

    # 2) SHAP explainer (Tree -> fallback Kernel)  ----------------------------
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"[WARN] TreeExplainer failed: {e}. Falling back to KernelExplainer (slower).")
        bg = shap.kmeans(X, 50)  # küçük bir arkaplan
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X, nsamples=200)

    # --- Normalize 'shap_values' to a list of (n_samples, n_features) arrays
    sv_list: list[np.ndarray]
    if isinstance(shap_values, list):
        sv_list = shap_values  # already per-class list
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # guess layout and split to list per class
            if arr.shape[0] in (3, 2):          # (n_classes, n_samples, n_features)
                sv_list = [arr[c, :, :] for c in range(arr.shape[0])]
            elif arr.shape[2] in (3, 2):        # (n_samples, n_features, n_classes)
                sv_list = [arr[:, :, c] for c in range(arr.shape[2])]
            else:
                raise RuntimeError(f"Unexpected SHAP shape: {arr.shape}")
        elif arr.ndim == 2:
            sv_list = [arr]  # single-output
        else:
            raise RuntimeError(f"Unexpected SHAP ndim: {arr.ndim}")

    # 3) Global importance: mean(|SHAP|) per feature, macro-avg over classes ---
    abs_means = np.mean([np.mean(np.abs(sv), axis=0) for sv in sv_list], axis=0)  # -> (n_features,)
    imp = pd.Series(abs_means, index=FEATURES).sort_values(ascending=False)
    REPORT_TBL.mkdir(parents=True, exist_ok=True)
    (REPORT_TBL / "rq5_shap_importance.json").write_text(json.dumps(imp.to_dict(), indent=2))

    # 4) Plots -----------------------------------------------------------------
    # Summary (beeswarm) and bar: pass per-class list
    plt.figure()
    shap.summary_plot(sv_list, X, feature_names=FEATURES,
                      class_names=list(SEVERITY_LABELS.values()), show=False)
    save_fig(REPORT_FIG / "rq5_shap_summary.png")

    plt.figure()
    shap.summary_plot(sv_list, X, feature_names=FEATURES,
                      class_names=list(SEVERITY_LABELS.values()), plot_type="bar", show=False)
    save_fig(REPORT_FIG / "rq5_shap_bar.png")

    # Dependence plots for top-2 features — kullanışlı olsun diye 'Moderate/Severe' sınıfını çizelim
    class_idx = 1  # 0: Mild, 1: Moderate/Severe, 2: None
    top2 = list(imp.index[:2])
    for feat in top2:
        plt.figure()
        shap.dependence_plot(feat, sv_list[class_idx], X, feature_names=FEATURES, show=False)
        save_fig(REPORT_FIG / f"rq5_shap_dependence_{feat}.png")

        
    print("✅ Saved:")
    print(f"  - {REPORT_TBL / 'rq5_shap_importance.json'}")
    print(f"  - {REPORT_FIG / 'rq5_shap_summary.png'}")
    print(f"  - {REPORT_FIG / 'rq5_shap_bar.png'}")
    for feat in top2:
        print(f"  - {REPORT_FIG / f'rq5_shap_dependence_{feat}.png'}")


if __name__ == "__main__":
    main()