# Centralized project paths with epoch-based folders
from pathlib import Path
import os

# --- Root folder ---
PROJ_ROOT = Path(__file__).resolve().parents[2]

# --- Base data folders ---
DATA_RAW     = PROJ_ROOT / "data" / "raw"
DATA_INTERIM = PROJ_ROOT / "data" / "interim"
DATA_PROC    = PROJ_ROOT / "data" / "processed"

# --- Epoch value from environment (default: 5) ---
EPOCHS = os.getenv("EPOCHS", "5")

# --- Epoch-specific report folders ---
REPORT_FIG = PROJ_ROOT / "reports" / "figures" / f"epoch{EPOCHS}"
REPORT_TBL = PROJ_ROOT / "reports" / "tables"  / f"epoch{EPOCHS}"

# Create directories if missing
for p in [DATA_RAW, DATA_INTERIM, DATA_PROC, REPORT_FIG, REPORT_TBL]:
    p.mkdir(parents=True, exist_ok=True)