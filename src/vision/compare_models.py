# Compare CNN backbones (ResNet50 vs MobileNetV2)

from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch

from src.utils.paths import PROJ_ROOT, REPORT_TBL, REPORT_FIG
from src.vision.train_cnn import build_model

CANDIDATES = ["resnet50", "mobilenetv2"]  # extend if needed


def load_summary(arch: str) -> dict | None:
    """Load RQ1 summary JSON produced by train_cnn.py."""
    path = REPORT_TBL / f"rq1_{arch}_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def count_params(arch: str, n_classes: int) -> int:
    """Build model and return the number of trainable parameters."""
    m = build_model(arch, n_classes)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def build_rows() -> list[dict]:
    """
    Build comparison rows for all candidate architectures and
    write model_comparison.json to REPORT_TBL.
    """
    REPORT_TBL.mkdir(parents=True, exist_ok=True)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for arch in CANDIDATES:
        summ = load_summary(arch)
        if summ is None:
            # skip missing model
            continue

        n_classes = len(summ.get("classes", [])) or None
        params = count_params(arch, n_classes or 17)  # fallback to 17 if not present

        rows.append(
            {
                "arch": arch,
                "val_best_acc": safe_float(summ.get("val_best_acc")),
                "test_acc": safe_float(summ.get("test_acc")),
                "num_params": int(params),
                "num_classes": n_classes,
            }
        )

    # write comparison table
    out_tbl = REPORT_TBL / "model_comparison.json"
    out_tbl.write_text(json.dumps(rows, indent=2))
    print(f"✅ Saved table: {out_tbl}")

    return rows


def plot_model_accuracy(
    rows: list[dict] | None = None,
    suffix: str = "png",
    dpi: int = 220,
) -> Path | None:
    """
    Plot test accuracy by architecture.

    Parameters
    ----------
    rows : list[dict] | None
        Precomputed rows. If None, they will be built with build_rows().
    suffix : str
        File extension to use (e.g. 'png' or 'pdf').
    dpi : int
        Dots per inch for the saved figure.
    """
    if rows is None:
        rows = build_rows()

    if not rows:
        print("No models found. Train at least one model first (make rq1).")
        return None

    plt.figure()
    x = [r["arch"] for r in rows]
    y = [r["test_acc"] for r in rows]
    plt.bar(x, y)
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.0)
    plt.title("CNN Test Accuracy by Architecture")
    acc_fig = REPORT_FIG / f"model_acc_bar.{suffix}"
    plt.savefig(acc_fig, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"✅ Saved figure: {acc_fig}")
    return acc_fig


def plot_model_params(
    rows: list[dict] | None = None,
    suffix: str = "png",
    dpi: int = 220,
) -> Path | None:
    """
    Plot parameter count (in millions) by architecture.

    Parameters
    ----------
    rows : list[dict] | None
        Precomputed rows. If None, they will be built with build_rows().
    suffix : str
        File extension to use (e.g. 'png' or 'pdf').
    dpi : int
        Dots per inch for the saved figure.
    """
    if rows is None:
        rows = build_rows()

    if not rows:
        print("No models found. Train at least one model first (make rq1).")
        return None

    plt.figure()
    x = [r["arch"] for r in rows]
    y = [r["num_params"] / 1e6 for r in rows]
    plt.bar(x, y)
    plt.ylabel("Parameters (Millions)")
    plt.title("CNN Size by Architecture")
    par_fig = REPORT_FIG / f"model_params_bar.{suffix}"
    plt.savefig(par_fig, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"✅ Saved figure: {par_fig}")
    return par_fig


def main():
    """
    CLI entry point.

    - Builds comparison rows + JSON table
    - Generates PNG figures for accuracy and parameter count
    """
    rows = build_rows()
    if not rows:
        # build_rows already printed a message; nothing else to do.
        return

    plot_model_accuracy(rows=rows, suffix="png", dpi=220)
    plot_model_params(rows=rows, suffix="png", dpi=220)


if __name__ == "__main__":
    main()