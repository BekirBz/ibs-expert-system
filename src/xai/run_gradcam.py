# Run Grad-CAM on a sample image from the test split (or a given path)
# Comments in English

from pathlib import Path
import argparse
import torch
from PIL import Image

from src.utils.paths import DATA_PROC, REPORT_FIG, PROJ_ROOT
from src.vision.train_cnn import build_model
from src.xai.gradcam import GradCAM, default_transform, denormalize, save_cam_figure


ACTIVE_CKPT = PROJ_ROOT / "models" / "active_cnn.pt"  # symlink to the active CNN


def get_device():
    # Prefer Apple Silicon (MPS), then CUDA, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cnn_from_active(device):
    """
    Load the 'active' CNN checkpoint and return (model, classes, target_layer).
    Picks the correct target conv block per architecture.
    """
    ckpt = torch.load(ACTIVE_CKPT, map_location=device)
    arch = ckpt["arch"]
    classes = ckpt["classes"]

    model = build_model(arch, len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # pick last meaningful conv block for Grad-CAM
    if arch.lower() == "resnet50":
        target_layer = model.layer4[-1]          # last bottleneck block
    elif arch.lower() == "mobilenetv2":
        target_layer = model.features[-1]        # last feature block
    else:
        raise ValueError(f"Unsupported arch for Grad-CAM: {arch}")

    return model, classes, target_layer


def pick_sample_image() -> Path:
    """Pick a random test image if no path is provided."""
    import random
    root = DATA_PROC / "images" / "test"
    all_jpg = list(root.rglob("*.jpg"))
    if not all_jpg:
        raise FileNotFoundError(f"No test images under {root}")
    return random.choice(all_jpg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None,
                    help="Path to an image. If omitted, a random test image is used.")
    ap.add_argument("--size", type=int, default=224, help="Input size for the model transforms.")
    args = ap.parse_args()

    device = get_device()
    model, classes, target_layer = load_cnn_from_active(device)
    tfm = default_transform(args.size)

    # read image
    img_path = Path(args.image) if args.image else pick_sample_image()
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    # forward + Grad-CAM
    camgen = GradCAM(model, target_layer)
    cam, cls_idx = camgen.generate(x)
    camgen.remove()

    pred_class = classes[cls_idx]
    print(f"[Grad-CAM] Image: {img_path}")
    print(f"[Grad-CAM] Predicted class: {pred_class} (idx={cls_idx})")

    # save overlay figure
    rgb = denormalize(x[0])
    out = REPORT_FIG / f"rq5_gradcam_{pred_class}.png"
    save_cam_figure(rgb, cam, out, title=f"Grad-CAM: {pred_class}")
    print(f"✅ Saved: {out}")


if __name__ == "__main__":
    main()