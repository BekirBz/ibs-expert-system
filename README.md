# IBS Expert System  
### AI-based Food Intake & Symptom Severity Prediction for IBS Patients  

This project develops an **AI-driven Expert System** that predicts **symptom severity in Irritable Bowel Syndrome (IBS)** patients based on their dietary intake.  
It integrates **food recognition (CNN)**, **ontology-based trigger mapping**, and **machine learning (XGBoost)** to produce interpretable, patient-centered insights.  
Explainability is achieved using **Grad-CAM** for CNN visualization and **SHAP** for feature-level interpretability.  

---

## Project Structure

ibs-expert-system/
│
├── data/
│   ├── raw/                     # Food-101 dataset
│   ├── processed/               # Preprocessed images
│   └── interim/                 # Trigger mapping CSV
│
├── models/                      # Trained model checkpoints
├── reports/
│   ├── figures/                 # Visual outputs (GradCAM, SHAP, etc.)
│   └── tables/                  # JSON + text reports
│
├── src/
│   ├── data_prep/               # Dataset download & preprocessing
│   ├── knowledge/               # Ontology and trigger mapping
│   ├── ml/                      # Symptom severity models (RQ3)
│   ├── integration/             # End-to-end pipeline (RQ4)
│   ├── vision/                  # CNN models (RQ1)
│   └── xai/                     # Explainable AI (RQ5)
│
├── requirements.txt             # Dependencies
├── Makefile                     # Automated workflow runner
└── README.md                    # Project overview

---

## Installation

```bash
# 1. Clone this repository
git clone 
cd ibs-expert-system

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
make setup


⸻

## Main Research Questions (RQs)

RQ	Description	Module
RQ1	Food image classification using CNN (ResNet50, MobileNetV2)	src/vision
RQ2	Food-trigger mapping and ontology evaluation	src/knowledge
RQ3	Symptom severity prediction via ML (LogReg, RF, XGBoost)	src/ml
RQ4	Full pipeline inference (Food → Trigger → Symptom)	src/integration
RQ5	Explainable AI with Grad-CAM and SHAP visualizations	src/xai


⸻

## Usage (Makefile Commands)

Command	Description
make data	Download Food-101 dataset
make subset	Create training subset
make mapping	Build trigger mapping + ontology
make rq1	Train & evaluate CNN
make rq2	Generate ontology figures
make rq3	Train symptom severity models
make rq4	Run full inference pipeline
make rq5	Run explainability bundle (Grad-CAM + SHAP)
make demo	Run complete demo (Pipeline + XAI)
make figures	Generate all figures (RQ1–RQ5)
make tables	Export all tables
make clean	Clear generated reports
make clean_data	Clear processed data


---

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

---

### RQ2 — Ontology Mapping & Trigger Correlations
| Mapping Accuracy | Ontology Graph |
|------------------|----------------|
| ![RQ2 Mapping Accuracy](reports/figures/rq2_mapping_accuracy.png) | ![RQ2 Ontology Graph](reports/figures/rq2_ontology_graph.png) |

---

### RQ3 — Symptom Severity Models
| ROC Curves | Feature Importance |
|-------------|--------------------|
| ![RQ3 ROC](reports/figures/rq3_roc_curves.png) | ![RQ3 Feature Importance](reports/figures/rq3_feature_importance.png) |

**Normalized Confusion Matrices**
| LogReg | Random Forest | XGBoost |
|--------|----------------|----------|
| ![LogReg](reports/figures/rq3_confusion_matrix_logreg.png) | ![RF](reports/figures/rq3_confusion_matrix_rf.png) | ![XGB](reports/figures/rq3_confusion_matrix_xgb.png) |

---

### RQ4 — End-to-End Pipeline Inference
The integrated pipeline combines food recognition, trigger ontology mapping, and XGBoost-based symptom prediction.

Example output (terminal):

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


---

### RQ5 — Explainable AI (XAI)
| Grad-CAM Examples | SHAP Summary |
|-------------------|---------------|
| ![GradCAM Cheesecake](reports/figures/rq5_gradcam_cheesecake.png) | ![SHAP Summary](reports/figures/rq5_shap_summary.png) |

**Feature Dependence & Importance**
| Gluten | High Fat | SHAP Bar |
|--------|-----------|-----------|
| ![SHAP Gluten](reports/figures/rq5_shap_dependence_gluten.png) | ![SHAP HighFat](reports/figures/rq5_shap_dependence_highfat.png) | ![SHAP Bar](reports/figures/rq5_shap_bar.png) |

---

## Generated Reports

All figures and tables are automatically saved under:

reports/
├── figures/
│   ├── rq1_*.png
│   ├── rq2_*.png
│   ├── rq3_*.png
│   ├── rq5_*.png
│   └── ...
└── tables/
    ├── rq1_*.json
    ├── rq3_*.json
    ├── rq5_*.json
    └── ...


⸻

## All outputs are reproducible using Makefile commands:

make rq4     # End-to-end inference
make rq5     # Explainable AI (GradCAM + SHAP)
make demo    # Combined pipeline + XAI demo
make figures # Generate all visual results


⸻

## Key Dependencies
	•	Python 3.12
	•	PyTorch 2.3.1
	•	TorchVision 0.18.1
	•	XGBoost 2.0.3
	•	scikit-learn 1.4.2
	•	SHAP 0.45.1
	•	NetworkX 3.2.1
	•	Graphviz 0.20.3
	•	Matplotlib 3.8.4
	•	Pandas 2.2.2

⸻

 ## Author

Bekir Bozoklar
M.Sc. Software Engineering
University of Europe for Applied Sciences, Germany

⸻


