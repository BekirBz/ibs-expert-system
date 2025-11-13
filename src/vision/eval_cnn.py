# Evaluate saved best model: classification report + confusion matrix figure
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import DATA_PROC, REPORT_FIG, REPORT_TBL, PROJ_ROOT

def load_best(arch: str, device):
    ckpt = torch.load(PROJ_ROOT / "models" / f"{arch}_best.pt", map_location=device)
    from src.vision.train_cnn import build_model  # reuse
    model = build_model(ckpt["arch"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    classes = ckpt["classes"]
    return model, classes

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=["resnet50","mobilenetv2"])
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    test_ds = datasets.ImageFolder(DATA_PROC / "images" / "test", transform=tf)
    test_ld = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model, classes = load_best(args.arch, device)
    y_true, y_pred = predict_all(model, test_ld, device)

    # classification report (csv-like txt)
    rep = classification_report(y_true, y_pred, target_names=classes, digits=3)
    (REPORT_TBL / f"rq1_{args.arch}_classification_report.txt").write_text(rep)

    # confusion matrix fig
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({args.arch})")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIG / f"rq1_confusion_matrix_{args.arch}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:",
          REPORT_TBL / f"rq1_{args.arch}_classification_report.txt",
          REPORT_FIG / f"rq1_confusion_matrix_{args.arch}.png")

if __name__ == "__main__":
    main()