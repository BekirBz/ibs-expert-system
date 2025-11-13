# Train a CNN (ResNet50 or MobileNetV2) on the selected Food-101 subset

import argparse, time, json
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import numpy as np

from src.utils.paths import DATA_PROC, REPORT_FIG, REPORT_TBL, PROJ_ROOT

def get_device():
    # Prefer Apple Silicon GPU (MPS), else CUDA, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_dataloaders(img_root: Path, img_size=224, batch_size=32):
    # standard augmentations for train; light eval transforms for val/test
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(img_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(img_root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(img_root / "test",  transform=eval_tf)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld, train_ds.classes

def build_model(arch: str, n_classes: int):
    # create model and replace classifier head
    if arch.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, n_classes)
    elif arch.lower() == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, n_classes)
    else:
        raise ValueError("Unsupported arch. Use 'resnet50' or 'mobilenetv2'.")
    return model

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=["resnet50","mobilenetv2"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = get_device()
    img_root = DATA_PROC / "images"
    train_ld, val_ld, test_ld, classes = build_dataloaders(img_root, batch_size=args.batch_size)
    n_classes = len(classes)

    REPORT_TBL.mkdir(parents=True, exist_ok=True)
    (PROJ_ROOT / "models").mkdir(exist_ok=True)

    model = build_model(args.arch, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_path = 0.0, PROJ_ROOT / "models" / f"{args.arch}_best.pt"
    print(f"Device: {device} | Classes: {n_classes} | Arch: {args.arch}")

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_ld, device, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_ld, device, criterion)
        dt = time.time() - t0
        print(f"[{epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "classes": classes
            }, best_path)

    # final test eval with best weights
    ckpt = torch.load(best_path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_acc = evaluate(model, test_ld, device, criterion)
    print(f"TEST: loss={te_loss:.4f} acc={te_acc:.4f}")

    # save quick metrics json (used by eval script)
    metrics = {"arch": args.arch, "val_best_acc": float(best_val), "test_acc": float(te_acc), "classes": classes}
    (REPORT_TBL / f"rq1_{args.arch}_summary.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()