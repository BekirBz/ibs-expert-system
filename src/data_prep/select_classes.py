# Build a smaller subset of Food-101 into processed/{train,val,test}
from pathlib import Path
import shutil, random, csv
from typing import List
from src.utils.paths import DATA_RAW, DATA_PROC, DATA_INTERIM

RANDOM_SEED = 42
SPLIT = (0.7, 0.15, 0.15)  # train, val, test

# You can edit this list; keep 20–30 classes for faster training
DEFAULT_SELECTED = [
    "pizza","lasagna","spaghetti_bolognese","steak","hamburger","hot_dog",
    "sushi","omelette","waffles","ice_cream","cheesecake",
    "tiramisu","fried_rice","ramen","pad_thai",
    "fish_and_chips","greek_salad","pancakes","apple_pie","french_fries"
]

def read_all_classes():
    meta = DATA_RAW / "food-101" / "meta" / "classes.txt"
    return [c.strip() for c in meta.read_text().splitlines() if c.strip()]

def main():
    random.seed(RANDOM_SEED)
    images_dir = DATA_RAW / "food-101" / "images"
    assert images_dir.exists(), "Food-101 images not found. Run download step first."

    # Read all valid class names from dataset
    all_classes = set(read_all_classes())

    # If you already created a CSV of classes, use it; else write default
    class_csv = DATA_INTERIM / "class_list.csv"
    if class_csv.exists():
        selected = [row.strip() for row in class_csv.read_text().splitlines()[1:]]
    else:
        selected = DEFAULT_SELECTED

    # Filter to only valid Food-101 classes
    missing = [c for c in selected if c not in all_classes]
    selected = [c for c in selected if c in all_classes]

    if missing:
        print("[INFO] The following classes are not in Food-101 and will be skipped:", missing)

    # Save final class list (overwrite to keep consistent)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    with open(class_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"])
        w.writerows([[c] for c in selected])

    # Prepare output directories
    for split in ["train", "val", "test"]:
        (DATA_PROC / "images" / split).mkdir(parents=True, exist_ok=True)

    # Copy images for each selected class
    for cls in selected:
        src = images_dir / cls
        imgs = list(src.glob("*.jpg"))
        if len(imgs) == 0:
            print(f"[WARN] No images found for class: {cls}")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        parts = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:]
        }

        for split, items in parts.items():
            out_dir = DATA_PROC / "images" / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in items:
                dst = out_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)

    print("✅ Subset created under:", DATA_PROC / "images")
    print("   Splits:", [d.name for d in (DATA_PROC / "images").iterdir()])

if __name__ == "__main__":
    main()