# Simple Grad-CAM utility for ResNet50 / MobileNetV2

from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """[C,H,W] tensor in ImageNet norm -> uint8 HxWx3 numpy."""
    x = img_t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        x[c] = x[c] * s + m
    x = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return x


class GradCAM:
    """Hook-based Grad-CAM for a given target layer."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.fmap = None
        self.grad = None

        def f_hook(_m, _i, o):
            self.fmap = o.detach()

        def b_hook(_m, gi, go):
            # go is a tuple; we need gradients wrt the layer output
            self.grad = go[0].detach()

        self._fh = target_layer.register_forward_hook(f_hook)
        self._bh = target_layer.register_full_backward_hook(b_hook)

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Return (cam[H,W] in [0,1], class_idx)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx]
        score.backward()

        # weights: global-average-pooled gradients
        weights = self.grad.mean(dim=(2, 3), keepdim=True)  # [N,C,1,1]
        cam = (weights * self.fmap).sum(dim=1, keepdim=True)  # [N,1,h,w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay heatmap onto RGB image (both HxW or HxWx3)."""
    import cv2  # comes with opencv-python if you have it; if not, fallback to matplotlib
    h, w = cam.shape
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    overlay = (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)
    return overlay


def default_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def save_cam_figure(orig_rgb: np.ndarray, cam: np.ndarray, out_path: Path, title: str = "Grad-CAM"):
    plt.figure(figsize=(6, 6))
    plt.imshow(orig_rgb)
    plt.axis("off")
    plt.title(f"{title} (overlay)")
    # fallback overlay without OpenCV
    try:
        import cv2  # noqa: F401
        over = overlay_cam(orig_rgb, cam)
        plt.imshow(over)
    except Exception:
        plt.imshow(cam, cmap="jet", alpha=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()