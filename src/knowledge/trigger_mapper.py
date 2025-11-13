# Rule-based mapping from food class to IBS trigger vector
# Vector order: [gluten, lactose, caffeine, highfat, fodmap]
# NOTE: These rules are heuristic, designed for proof-of-concept (not medical use).

from typing import Dict, List

TRIGGERS: List[str] = ["gluten", "lactose", "caffeine", "highfat", "fodmap"]

# You can refine these rules later based on literature
RULES: Dict[str, Dict[str, int]] = {
    # Pasta / bakery (gluten, often FODMAP via wheat/onion/garlic sauces)
    "pizza":                 {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},  # cheese + wheat + fat
    "lasagna":               {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},
    "spaghetti_bolognese":   {"gluten":1,"lactose":0,"caffeine":0,"highfat":1,"fodmap":1},
    "pancakes":              {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},
    "apple_pie":             {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},  # apple (FODMAP fructose)
    "waffles":               {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},

    # Meats / fried
    "steak":                 {"gluten":0,"lactose":0,"caffeine":0,"highfat":1,"fodmap":0},
    "hamburger":             {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},  # bun + cheese + sauces
    "hot_dog":               {"gluten":1,"lactose":0,"caffeine":0,"highfat":1,"fodmap":1},
    "fish_and_chips":        {"gluten":1,"lactose":0,"caffeine":0,"highfat":1,"fodmap":1},
    "french_fries":          {"gluten":0,"lactose":0,"caffeine":0,"highfat":1,"fodmap":0},  # oil/fat only

    # Eggs / salads / sushi
    "omelette":              {"gluten":0,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},  # milk/cheese optional, onion (FODMAP) often
    "greek_salad":           {"gluten":0,"lactose":1,"caffeine":0,"highfat":0,"fodmap":1},  # onion (FODMAP), feta (lactose low/mod)
    "sushi":                 {"gluten":1,"lactose":0,"caffeine":0,"highfat":0,"fodmap":0},  # soy sauce often contains wheat (gluten)

    # Asian dishes (garlic/onion sauces → FODMAP; sometimes gluten via soy/noodles)
    "fried_rice":            {"gluten":1,"lactose":0,"caffeine":0,"highfat":1,"fodmap":1},
    "ramen":                 {"gluten":1,"lactose":0,"caffeine":0,"highfat":1,"fodmap":1},
    "pad_thai":              {"gluten":0,"lactose":0,"caffeine":0,"highfat":1,"fodmap":1},

    # Desserts (lactose + FODMAP sugars; often high fat)
    "ice_cream":             {"gluten":0,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},
    "cheesecake":            {"gluten":1,"lactose":1,"caffeine":0,"highfat":1,"fodmap":1},
    "tiramisu":              {"gluten":1,"lactose":1,"caffeine":1,"highfat":1,"fodmap":1},  # coffee → caffeine
}

def to_vector(food_class: str) -> List[int]:
    """Return [gluten,lactose,caffeine,highfat,fodmap] for given class name."""
    if food_class in RULES:
        r = RULES[food_class]
    else:
        # default: no triggers (unknown class)
        r = {k: 0 for k in TRIGGERS}
    return [r[t] for t in TRIGGERS]

def get_rules() -> Dict[str, Dict[str, int]]:
    """Expose raw rules (useful for exporting mapping CSV)."""
    return RULES

def map_class_to_triggers(food_class: str):
    """
    Compatibility helper for integration pipeline.
    Returns trigger vector [gluten, lactose, caffeine, highfat, fodmap]
    for a given food class.
    """
    return to_vector(food_class)