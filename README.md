IBS Expert System

AI-based Food Intake & Symptom Severity Prediction for IBS Patients

This project develops an AI-driven Expert System that predicts symptom severity in Irritable Bowel Syndrome (IBS) patients based on their dietary intake.
It integrates food recognition (CNN), ontology-based trigger mapping, and machine learning (XGBoost) to produce interpretable, patient-centered insights.
Explainability is provided using Grad-CAM (CNN decisions) and SHAP (feature-level interpretability).

⸻

Project Structure

ibs-expert-system/
│
├── data/
│   ├── raw/                     # Food-101 dataset
│   ├── processed/               # Preprocessed images
│   └── interim/                 # Trigger mapping CSV
│
├── models/                      # Trained model checkpoints
│
├── reports/
│   ├── figures/                 # Visual outputs (GradCAM, SHAP, etc.)
│   └── tables/                  # JSON + text reports
│
├── src/
│   ├── data_prep/               # Dataset download & preprocessing
│   ├── knowledge/               # Ontology & trigger mapping
│   ├── ml/                      # Symptom severity models (RQ3)
│   ├── integration/             # End-to-end pipeline (RQ4)
│   ├── vision/                  # CNN classification models (RQ1)
│   └── xai/                     # Explainable AI (RQ5)
│
├── requirements.txt             # Dependencies
├── Makefile                     # Automated workflow
└── README.md                    # Project overview


⸻

Installation

# 1. Clone this repository
git clone https://github.com/BekirBz/ibs-expert-system.git
cd ibs-expert-system

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
make setup


⸻

Main Research Questions (RQs)

RQ	Description	Module
RQ1	Food image classification using CNN (ResNet50, MobileNetV2)	src/vision
RQ2	Food-trigger mapping & ontology evaluation	src/knowledge
RQ3	Symptom severity prediction via ML (LogReg, RF, XGB)	src/ml
RQ4	Full inference pipeline (Food → Trigger → Symptom)	src/integration
RQ5	Explainable AI (Grad-CAM & SHAP visualizations)	src/xai


⸻

Usage (Makefile Commands)

Command	Description
make data	Download Food-101 dataset
make subset	Create training subset
make mapping	Build trigger mapping + ontology graph
make rq1	Train & evaluate CNN
make rq2	Generate ontology reports
make rq3	Train symptom severity models
make rq4	Run full inference pipeline
make rq5_gradcam	Generate Grad-CAM visualizations
make rq5_shap	Generate SHAP plots
make figures	Generate all figures (RQ1–RQ5)
make tables	Export all tables
make clean	Clean generated reports
make clean_data	Remove processed dataset


⸻

Visual Preview

RQ1 — CNN Food Classification
	•	Accuracy Comparison
	•	reports/figures/model_acc_bar.png
	•	reports/figures/model_params_bar.png
	•	Confusion Matrices
	•	reports/figures/rq1_confusion_matrix_resnet50.png
	•	reports/figures/rq1_confusion_matrix_mobilenetv2.png

⸻

RQ2 — Ontology Mapping
	•	Mapping Accuracy
reports/figures/rq2_mapping_accuracy.png
	•	Ontology Graph
reports/figures/rq2_ontology_graph.png

⸻

RQ3 — Symptom Severity Models
	•	ROC Curves
reports/figures/rq3_roc_curves.png
	•	Feature Importance
reports/figures/rq3_feature_importance.png
	•	Confusion Matrices
	•	rq3_confusion_matrix_logreg.png
	•	rq3_confusion_matrix_rf.png
	•	rq3_confusion_matrix_xgb.png

⸻

RQ4 — End-to-End Pipeline Inference

Sample output:

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


⸻

RQ5 — Explainable AI
	•	Grad-CAM example:
reports/figures/rq5_gradcam_cheesecake.png
	•	SHAP summary:
reports/figures/rq5_shap_summary.png
	•	SHAP dependence & bar plots:
	•	rq5_shap_dependence_gluten.png
	•	rq5_shap_dependence_highfat.png
	•	rq5_shap_bar.png

⸻

Generated Reports

reports/
│
├── figures/
│   ├── rq1_*.png
│   ├── rq2_*.png
│   ├── rq3_*.png
│   ├── rq5_*.png
│   └── ...
│
└── tables/
    ├── rq1_*.json
    ├── rq3_*.json
    ├── rq5_*.json
    └── ...


⸻

Key Dependencies
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

Author

Bekir Bozoklar
M.Sc. Software Engineering
University of Europe for Applied Sciences, Germany
