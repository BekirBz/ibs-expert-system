# Utility to export all PNG figures under reports/figures/** into matching PDF folders
# Comments in English

from pathlib import Path
from PIL import Image

from src.utils.paths import REPORT_FIG


def convert_pngs_recursively(dpi: int = 600) -> None:
    """
    Recursively convert all PNG files under reports/figures/** into PDF,
    preserving the same subfolder structure under reports/figures/pdf/.
    """

    src_root = REPORT_FIG
    dst_root = REPORT_FIG / "pdf"
    dst_root.mkdir(parents=True, exist_ok=True)

    png_files = sorted(src_root.rglob("*.png"))
    if not png_files:
        print(f"[WARN] No PNG files found under: {src_root}")
        return

    print(f"[INFO] Converting {len(png_files)} PNG files to PDF (dpi={dpi})...")

    for png_path in png_files:
        # Calculate relative path (e.g., epoch5/rq1_confusion...)
        rel = png_path.relative_to(src_root)

        # Build PDF output path (inside reports/figures/pdf/…)
        out_path = dst_root / rel.with_suffix(".pdf")

        # Ensure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(png_path).convert("RGB")
            img.save(out_path, "PDF", resolution=dpi)
            print(f"  ✅ {rel}  ->  pdf/{rel.with_suffix('.pdf')}")
        except Exception as e:
            print(f"[ERROR] Failed to convert {png_path}: {e}")

    print(f"[DONE] PDF export finished. Output root: {dst_root}")


def main():
    convert_pngs_recursively(dpi=600)


if __name__ == "__main__":
    main()