# Download and extract Food-101 dataset 
import tarfile
import urllib.request
from pathlib import Path
from src.utils.paths import DATA_RAW

URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DEST = DATA_RAW / "food-101.tar.gz"
EXTRACT_DIR = DATA_RAW / "food-101"

def main():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    if (EXTRACT_DIR / "images").exists():
        print(f"Already extracted at: {EXTRACT_DIR}")
        return

    if not DEST.exists():
        print("Downloading Food-101 (~5 GB)...")
        # simple progress hook
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 / (total_size or 1))
            print(f"\r  {pct:5.1f}% ({downloaded/1e6:,.1f} MB)", end="")
        urllib.request.urlretrieve(URL, DEST, _progress)
        print("\n Download complete:", DEST)

    print(" Extracting .tar.gz ... (this may take a few minutes)")
    with tarfile.open(DEST, "r:gz") as tar:
        tar.extractall(DATA_RAW)
    print("Extracted to:", EXTRACT_DIR)

if __name__ == "__main__":
    main()
