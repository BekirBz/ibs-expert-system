# ===== IBS Expert System — Makefile =====
# Unified automation for all research stages (RQ1–RQ5)
# Compatible with macOS + Python venv
PY := .venv/bin/python

# ---- Default parameters (can be overridden from CLI) ----
ARCH   ?= resnet50
EPOCHS ?= 5
BATCH  ?= 32
LR     ?= 1e-4

# EPOCHS değişkenini Python'a environment olarak gönder
# Örn: make rq1 ARCH=resnet50 EPOCHS=20
export EPOCHS

.PHONY: help setup data subset mapping rq1 rq1_eval rq2 rq3 rq4 rq5_gradcam rq5_shap rq5 compare figures tables all demo clean clean_data export_pdf

# === HELP ===
help:
	@echo "Available commands:"
	@echo "  setup        - Install dependencies from requirements.txt"
	@echo "  data         - Download Food-101 dataset (data/raw)"
	@echo "  subset       - Select subset classes and split into train/val/test"
	@echo "  mapping      - Build trigger mapping CSV + ontology graph"
	@echo "  rq1          - Train CNN (ResNet50 or MobileNetV2)"
	@echo "  rq1_eval     - Evaluate the trained CNN and generate figures"
	@echo "  rq2          - Evaluate trigger mapping and plot ontology"
	@echo "  rq3          - Train symptom severity models (LogReg, RF, XGB)"
	@echo "  rq4          - Run end-to-end inference pipeline"
	@echo "  rq5_gradcam  - Generate Grad-CAM heatmaps for visual explanations"
	@echo "  rq5_shap     - Generate SHAP explainability plots"
	@echo "  rq5          - Run both Grad-CAM and SHAP (XAI bundle)"
	@echo "  compare      - Compare model architectures (ResNet50 vs MobileNetV2)"
	@echo "  figures      - Generate all figures (RQ1–RQ5)"
	@echo "  tables       - Generate table outputs (RQ1, RQ3, RQ5-SHAP)"
	@echo "  demo         - Pipeline inference + XAI bundle (rq4 + rq5)"
	@echo "  all          - Run the entire workflow sequentially"
	@echo "  clean        - Delete generated reports (keep best models)"
	@echo "  clean_data   - Delete processed data (keep raw dataset)"
	@echo "  export_pdf   - Convert all PNG figures to PDF (uses src.utils.export_results)"

# === SETUP ENVIRONMENT ===
setup:
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -r requirements.txt

# === DATA PREPARATION ===
data:
	$(PY) -m src.data_prep.download_food101

subset:
	$(PY) -m src.data_prep.select_classes

mapping:
	$(PY) -m src.data_prep.build_trigger_mapping
	$(PY) -m src.knowledge.eval_mapping
	$(PY) -m src.knowledge.plot_ontology

# === RQ1: CNN TRAINING & EVALUATION ===
rq1:
	$(PY) -m src.vision.train_cnn --arch $(ARCH) --epochs $(EPOCHS) --batch_size $(BATCH) --lr $(LR)
	$(PY) -m src.vision.eval_cnn  --arch $(ARCH)

rq1_eval:
	$(PY) -m src.vision.eval_cnn  --arch $(ARCH)

# === RQ2: TRIGGER MAPPING + ONTOLOGY ===
rq2:
	$(PY) -m src.knowledge.eval_mapping
	$(PY) -m src.knowledge.plot_ontology

# === RQ3: SYMPTOM SEVERITY MODELS ===
rq3:
	$(PY) -m src.ml.train_symptom_model

# === RQ4: FULL PIPELINE INFERENCE ===
rq4:
	$(PY) -m src.integration.pipeline_infer

# === RQ5: EXPLAINABLE AI (XAI) ===
rq5_gradcam:
	$(PY) -m src.xai.run_gradcam
	sleep 2

rq5_shap:
	$(PY) -m src.xai.shap_explain

# Bundle target for XAI
rq5: rq5_gradcam rq5_shap

# --- Model comparison (ResNet50 vs MobileNetV2) ---
compare:
	$(PY) -m src.vision.compare_models

# === COMBINED OUTPUTS ===
figures: rq1 rq2 rq3 rq5_gradcam rq5_shap
tables: rq1 rq3 rq5_shap

# Quick demo: pipeline + XAI
demo: rq4 rq5

# === COMPLETE PIPELINE ===
all: data subset mapping rq1 rq2 rq3 rq4 rq5

# === CLEANING ===
clean:
	@echo "Removing generated reports (keeping best model checkpoints)..."
	@rm -rf reports/figures/* reports/tables/*

clean_data:
	@echo "Removing processed data (keeping raw dataset)..."
	@rm -rf data/processed/*

# === EXPORT PNG -> PDF (dpi=600, epoch klasörlerine göre) ===
export_pdf:
	$(PY) -m src.utils.export_results