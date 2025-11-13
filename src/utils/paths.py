# Centralized project paths 
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJ_ROOT / "data" / "raw"
DATA_INTERIM = PROJ_ROOT / "data" / "interim"
DATA_PROC = PROJ_ROOT / "data" / "processed"
REPORT_FIG = PROJ_ROOT / "reports" / "figures"
REPORT_TBL = PROJ_ROOT / "reports" / "tables"

for p in [DATA_RAW, DATA_INTERIM, DATA_PROC, REPORT_FIG, REPORT_TBL]:
    p.mkdir(parents=True, exist_ok=True)