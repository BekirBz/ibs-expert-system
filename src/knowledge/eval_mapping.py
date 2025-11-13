# Evaluate knowledge-based trigger mapping on test set predictions

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score

from src.utils.paths import DATA_PROC, REPORT_TBL, REPORT_FIG, PROJ_ROOT
from src.knowledge.trigger_mapper import TRIGGERS, to_vector
from src.vision.train_cnn import build_model

def load_best(arch: str, device):
    ckpt = torch.load(PROJ_ROOT / "models" / f"{arch}_best.pt", map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    classes = ckpt["classes"]
    return model, classes

@torch.no_grad()
def get_true_pred_classes(model, classes, device):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(DATA_PROC / "images" / "test", transform=tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    y_true_cls, y_pred_cls = [], []
    model.eval()
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_pred_cls.extend(pred.tolist())
        y_true_cls.extend(y.numpy().tolist())

    true_names = [classes[i] for i in y_true_cls]
    pred_names = [classes[i] for i in y_pred_cls]
    return true_names, pred_names

def mapping_metrics(true_names, pred_names):
    # map classes -> trigger vectors
    true_vecs = np.array([to_vector(c) for c in true_names], dtype=int)
    pred_vecs = np.array([to_vector(c) for c in pred_names], dtype=int)

    # per-trigger accuracy
    per_trigger_acc = {}
    for i, t in enumerate(TRIGGERS):
        per_trigger_acc[t] = float((true_vecs[:, i] == pred_vecs[:, i]).mean())

    # exact vector match accuracy
    exact_match = float((true_vecs == pred_vecs).all(axis=1).mean())

    return per_trigger_acc, exact_match

def plot_trigger_bar(per_trigger_acc, save_path):
    fig = plt.figure(figsize=(6,4))
    keys = list(per_trigger_acc.keys())
    vals = [per_trigger_acc[k] for k in keys]
    plt.bar(keys, vals)
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Trigger-wise Mapping Accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

def main(arch="resnet50"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, classes = load_best(arch, device)
    true_names, pred_names = get_true_pred_classes(model, classes, device)

    per_trigger_acc, exact_match = mapping_metrics(true_names, pred_names)

    REPORT_TBL.mkdir(parents=True, exist_ok=True)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)

    (REPORT_TBL / "rq2_mapping_metrics.json").write_text(
        json.dumps({"per_trigger_acc": per_trigger_acc,
                    "exact_match": exact_match}, indent=2)
    )
    plot_trigger_bar(per_trigger_acc, REPORT_FIG / "rq2_mapping_accuracy.png")

    print("Saved:",
          REPORT_TBL / "rq2_mapping_metrics.json",
          REPORT_FIG / "rq2_mapping_accuracy.png")

if __name__ == "__main__":
    main()