# Run Grad-CAM on a sample image from the test split (or a given path)
# Comments in English

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Optional

import torch
from PIL import Image

from src.utils.paths import DATA_PROC, REPORT_FIG, PROJ_ROOT
from src.vision.train_cnn import build_model
from src.xai.gradcam import GradCAM, default_transform, denormalize, save_cam_figure


# Symlink to the currently "active" CNN checkpoint
ACTIVE_CKPT = PROJ_ROOT / "models" / "active_cnn.pt"


def get_device():
    """Return best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cnn_from_active(device):
    """
    Load the 'active' CNN checkpoint and return (model, classes, target_layer).

    The checkpoint is expected to contain:
        - arch: backbone name ("resnet50", "mobilenetv2", ...)
        - state_dict: model weights
        - classes: list of food class names
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


def run_gradcam_single(
    image: Optional[str | Path] = None,
    out_dir: Optional[Path] = None,
    suffix: str = "png",
    size: int = 224,
    **kwargs,
) -> Path:
    """
    Run Grad-CAM for a single image and save the overlay figure.

    Parameters
    ----------
    image : str | Path | None
        Input image path. If None, a random test image is chosen from test split.
    out_dir : Path | None
        Output directory for the figure. If None, uses REPORT_FIG.
    suffix : str
        File extension ("png", "pdf", ...).
    size : int
        Input image size for the transform.
    **kwargs :
        Extra keyword args (e.g., dpi) are accepted for compatibility but ignored.

    Returns
    -------
    Path
        Path to the saved Grad-CAM figure.
    """
    device = get_device()
    model, classes, target_layer = load_cnn_from_active(device)
    tfm = default_transform(size)

    # choose image
    img_path = Path(image) if image is not None else pick_sample_image()
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    # forward + Grad-CAM
    camgen = GradCAM(model, target_layer)
    cam, cls_idx = camgen.generate(x)
    camgen.remove()

    pred_class = classes[cls_idx]
    print(f"[Grad-CAM] Image: {img_path}")
    print(f"[Grad-CAM] Predicted class: {pred_class} (idx={cls_idx})")

    # output path
    if out_dir is None:
        out_dir = REPORT_FIG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rq5_gradcam_{pred_class}.{suffix}"

    # save overlay figure (save_cam_figure handles figure + saving)
    rgb = denormalize(x[0])
    save_cam_figure(rgb, cam, out_path, title=f"Grad-CAM: {pred_class}")
    print(f"✅ Saved: {out_path}")

    return out_path


def main():
    """
    CLI entry point.

    Examples:
        python -m src.xai.run_gradcam
        python -m src.xai.run_gradcam --image path/to/img.jpg --size 224
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image. If omitted, a random test image is used.",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=224,
        help="Input size for the model transforms.",
    )
    args = ap.parse_args()

    # For CLI usage we keep default PNG output under reports/figures
    run_gradcam_single(
        image=args.image,
        out_dir=REPORT_FIG,
        suffix="png",
        size=args.size,
    )


if __name__ == "__main__":
    main()