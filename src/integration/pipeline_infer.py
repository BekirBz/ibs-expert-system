# End-to-end IBS Expert System inference pipeline

from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import json

from src.utils.paths import DATA_PROC, PROJ_ROOT
from src.vision.train_cnn import build_model
from src.knowledge.trigger_mapper import map_class_to_triggers
from src.ml.train_symptom_model import TRIGGER_COLS
import joblib


SEVERITY_LABELS = {
    0: "Mild",
    1: "Moderate/Severe",
    2: "None",
}

# CONFIG

ARCH = "resnet50"
CNN_CKPT = PROJ_ROOT / "models" / f"{ARCH}_best.pt"
SYMPTOM_MODEL = PROJ_ROOT / "models" / "xgb_symptom_model.pkl"
CLASS_LIST_PATH = PROJ_ROOT / "data" / "interim" / "class_list.csv"

# LOAD MODELS

def load_cnn_model(device="mps"):
    print("[INFO] Loading CNN model...")
    ckpt = torch.load(CNN_CKPT, map_location=device)
    # build_model(arch, n_classes)
    model = build_model(ckpt["arch"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    classes = ckpt["classes"]
    return model, classes

def load_symptom_model():
    print("[INFO] Loading XGBoost model...")
    model = joblib.load(SYMPTOM_MODEL)
    return model


# PIPELINE INFERENCE

def infer_image(image_path: Path, device="mps"):
    # load cnn + symptom model
    cnn_model, classes = load_cnn_model(device)
    symptom_model = load_symptom_model()

    # preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # CNN prediction
    with torch.no_grad():
        logits = cnn_model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    pred_class = classes[pred_idx]
    print(f"[CNN] Predicted food class: {pred_class}")
    

    # Trigger mapping
    triggers = map_class_to_triggers(pred_class)
    print(f"[Ontology] Trigger vector: {triggers}")

    # Severity prediction
    X = pd.DataFrame([triggers], columns=TRIGGER_COLS)
    raw_pred = symptom_model.predict(X)[0]
    severity_label = SEVERITY_LABELS.get(int(raw_pred), str(raw_pred))
    print(f"[Symptom Model] Predicted severity: {severity_label}")


    # Optional: probability distribution
    proba = symptom_model.predict_proba(X)
    raw_labels = symptom_model.classes_

    proba_dict = {
        SEVERITY_LABELS.get(int(lbl), str(int(lbl))): float(prob)
        for lbl, prob in zip(raw_labels, proba[0])
    }

    print(f"[Symptom Model] Probabilities: {json.dumps(proba_dict, indent=2)}")

    return {
        "class": pred_class,
        "triggers": triggers,
        "severity": severity_label,
        "probabilities": proba_dict
    }


if __name__ == "__main__":
    import random
    root = DATA_PROC / "images" / "test"
    all_images = list(root.rglob("*.jpg"))
    if not all_images:
        print("No test images found under:", root)
    else:
        sample = random.choice(all_images)
        print("[INFO] Using sample image:", sample)
        result = infer_image(sample)
        print("\n✅ Final Inference Result:")
        print(json.dumps(result, indent=2))


