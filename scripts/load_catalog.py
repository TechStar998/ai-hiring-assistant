import json
from pathlib import Path

# Path to the catalog
catalog_path = Path("data") / "shl_product_catalog.json"

# Load JSON
with open(catalog_path, "r", encoding="utf-8") as file:
    catalog = json.load(file)

print("=" * 50)
print(f"Total assessments: {len(catalog)}")
print("=" * 50)

print("\nKeys available in one assessment:\n")
print(catalog[0].keys())

print("\nFirst assessment:\n")
print(catalog[0])