# IBS Expert System
### AI-based food intake & symptom severity prediction for IBS patients

This repository delivers an **AI-driven expert system** that estimates **IBS symptom severity** directly from a meal photo. The solution combines:

- **Food recognition (ResNet50 / MobileNetV2)** trained on a curated Food-101 subset.
- **Ontology-based trigger mapping** that turns dish predictions into IBS trigger vectors (gluten, lactose, caffeine, high-fat, FODMAP).
- **Symptom modeling (XGBoost + baselines)** with explainability via **Grad-CAM** and **SHAP**.

Every step is orchestrated through the Makefile so datasets, checkpoints, figures, and tables stay reproducible.

----

## High-Level Workflow

```
        +------------+     +---------------------+     +--------------------+
        |  Data Prep | --> | CNN Training (RQ1)   | --> | Food Predictions   |
        +------------+     +---------------------+     +--------------------+
               |                       |                        |
               v                       v                        v
   +------------------+      +--------------------+    +-----------------------+
   | Trigger Mapping  | -->  | Symptom Models      | -> | Pipeline Inference    |
   | & Ontology (RQ2) |      | (LogReg/RF/XGB, RQ3)|    | + Explainability RQ4/5|
   +------------------+      +--------------------+    +-----------------------+
```

Artifacts flow from `data/` → `models/` → `reports/epoch{EPOCHS}` so you can regenerate everything with a single command.

----

## Repository Layout

```
ibs-expert-system/
├── data/              # raw / interim / processed assets (Food-101 subset, mappings)
├── models/            # trained CNN + symptom-severity checkpoints
├── reports/           # figures/epochX + tables/epochX (auto-created)
├── src/               # research modules (data_prep, vision, knowledge, ml, integration, xai)
├── Makefile           # single entry point for every RQ
├── requirements.txt   # Python dependencies
└── README.md
```

----

## Setup

```bash
# 1) Clone + enter
git clone <your-fork-or-ssh-url> ibs-expert-system
cd ibs-expert-system

# 2) Create a virtualenv (Python ≥3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3) Install requirements via the Makefile helper
make setup
```

> **Tip:** all `make …` targets reuse `.venv/bin/python`, so you can stay inside the repo root without re-activating the environment later.

----

## Make Targets (cheat sheet)

| Command            | Description                                                     |
|--------------------|-----------------------------------------------------------------|
| `make data`        | Download + extract Food-101 into `data/raw`.                    |
| `make subset`      | Build the curated class subset and split into train/val/test.   |
| `make mapping`     | Export trigger mappings + ontology plot (RQ2).                  |
| `make rq1`         | Train and evaluate the CNN for a chosen `ARCH`.                 |
| `make rq2`         | Re-run ontology evaluation/plots without rebuilding CSV.        |
| `make rq3`         | Train symptom-severity ML models and save metrics/plots.        |
| `make rq4`         | Execute end-to-end inference (Food → Trigger → Severity).       |
| `make rq5`         | Generate Grad-CAM overlays and SHAP visualizations.             |
| `make demo`        | Quick showcase (`rq4` + `rq5`).                                  |
| `make figures`     | Produce every figure for RQ1–RQ5.                               |
| `make tables`      | Export all JSON/txt tables (RQ1, RQ3, RQ5).                      |
| `make clean`       | Remove generated reports only.                                  |
| `make clean_data`  | Remove processed data (raw dataset is preserved).               |

Targets accept overrides, e.g. `make rq1 ARCH=mobilenetv2 EPOCHS=20 BATCH=16`.

----

## Research Questions

| RQ  | Description                                              | Module             |
|-----|----------------------------------------------------------|--------------------|
| RQ1 | Food image classification with CNN backbones.            | `src/vision`       |
| RQ2 | Food-trigger ontology & mapping fidelity.                | `src/knowledge`    |
| RQ3 | Symptom severity modeling (LogReg, RF, XGBoost).         | `src/ml`           |
| RQ4 | End-to-end pipeline inference (Food → Trigger → Symptom) | `src/integration`  |
| RQ5 | Explainability bundle (Grad-CAM + SHAP).                 | `src/xai`          |

----

## Visual Preview

Below are selected results generated automatically from the system:

### RQ1 — CNN Food Classification
| Model | Accuracy Comparison |
|--------|---------------------|
| ![Model Accuracy Bar](reports/figures/model_acc_bar.png) | ![Model Parameters](reports/figures/model_params_bar.png) |

**Confusion Matrices**
| ResNet50 | MobileNetV2 |
|-----------|--------------|
| ![Confusion Matrix ResNet50](reports/figures/rq1_confusion_matrix_resnet50.png) | ![Confusion Matrix MobileNetV2](reports/figures/rq1_confusion_matrix_mobilenetv2.png) |

----

### RQ2 — Ontology Mapping & Trigger Correlations
| Mapping Accuracy | Ontology Graph |
|------------------|----------------|
| ![RQ2 Mapping Accuracy](reports/figures/rq2_mapping_accuracy.png) | ![RQ2 Ontology Graph](reports/figures/rq2_ontology_graph.png) |

----

### RQ3 — Symptom Severity Models
| ROC Curves | Feature Importance |
|-------------|--------------------|
| ![RQ3 ROC](reports/figures/rq3_roc_curves.png) | ![RQ3 Feature Importance](reports/figures/rq3_feature_importance.png) |

**Normalized Confusion Matrices**
| LogReg | Random Forest | XGBoost |
|--------|----------------|----------|
| ![LogReg](reports/figures/rq3_confusion_matrix_logreg.png) | ![RF](reports/figures/rq3_confusion_matrix_rf.png) | ![XGB](reports/figures/rq3_confusion_matrix_xgb.png) |

----

### RQ4 — End-to-End Pipeline Inference
The integrated pipeline combines food recognition, trigger ontology mapping, and XGBoost-based symptom prediction.

Example console output:

```
{
  "class": "steak",
  "triggers": [0, 0, 0, 1, 0],
  "severity": "Mild",
  "probabilities": {
    "Mild": 0.907,
    "Moderate/Severe": 0.042,
    "None": 0.050
  }
}
```

----

### RQ5 — Explainable AI (XAI)
| Grad-CAM Examples | SHAP Summary |
|-------------------|--------------|
| ![GradCAM Cheesecake](reports/figures/rq5_gradcam_cheesecake.png) | ![SHAP Summary](reports/figures/rq5_shap_summary.png) |

**Feature Dependence & Importance**
| Gluten | High Fat | SHAP Bar |
|--------|-----------|-----------|
| ![SHAP Gluten](reports/figures/rq5_shap_dependence_gluten.png) | ![SHAP HighFat](reports/figures/rq5_shap_dependence_highfat.png) | ![SHAP Bar](reports/figures/rq5_shap_bar.png) |

----

## Generated Reports

Outputs are grouped by epoch (default `EPOCHS=5`, override via `make … EPOCHS=10`):

```
reports/
├── figures/
│   └── epoch5/
│       ├── rq1_*.png
│       ├── rq2_*.png
│       ├── rq3_*.png
│       └── rq5_*.png
└── tables/
    └── epoch5/
        ├── rq1_*.json
        ├── rq3_*.json
        └── rq5_*.json
```

----

## Key Dependencies

- Python 3.12
- PyTorch 2.3.1 / TorchVision 0.18.1
- XGBoost 2.0.3
- scikit-learn 1.4.2
- SHAP 0.45.1
- NetworkX 3.2.1 + Graphviz 0.20.3
- Matplotlib 3.8.4, Pandas 2.2.2

----

## Author

Bekir Bozoklar  
M.Sc. Software Engineering  
University of Europe for Applied Sciences, Germany

