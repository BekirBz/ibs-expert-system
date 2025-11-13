# Create trigger_mapping.csv 
import csv
from src.utils.paths import DATA_INTERIM
from src.knowledge.trigger_mapper import TRIGGERS, get_rules, to_vector

CLASS_LIST_CSV = DATA_INTERIM / "class_list.csv"
OUT_CSV = DATA_INTERIM / "trigger_mapping.csv"

def main():
    # read selected classes
    if CLASS_LIST_CSV.exists():
        classes = [row.strip() for row in CLASS_LIST_CSV.read_text().splitlines()[1:]]
    else:
        raise FileNotFoundError("class_list.csv not found. Run select_classes step first.")

    rules = get_rules()
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + TRIGGERS)
        for c in classes:
            vec = to_vector(c)
            w.writerow([c] + vec)

    print("✅ trigger_mapping.csv written to:", OUT_CSV)

if __name__ == "__main__":
    main()