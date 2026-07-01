import json
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the catalog
catalog_path = Path("data") / "shl_product_catalog.json"

with open(catalog_path, "r", encoding="utf-8") as f:
    catalog = json.load(f)

documents = []

for assessment in catalog:
    text = f"""
    Name: {assessment.get("name", "")}
    Description: {assessment.get("description", "")}
    Job Levels: {", ".join(assessment.get("job_levels", []))}
    Categories: {", ".join(assessment.get("keys", []))}
    Languages: {", ".join(assessment.get("languages", []))}
    Remote Testing: {assessment.get("remote", "")}
    Adaptive: {assessment.get("adaptive", "")}
    """

    documents.append(text)

print(f"Created {len(documents)} documents.")

# Create embeddings
embeddings = model.encode(documents, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
Path("vectorstore").mkdir(exist_ok=True)
faiss.write_index(index, "vectorstore/shl_index.faiss")

# Save catalog metadata
with open("vectorstore/catalog.pkl", "wb") as f:
    pickle.dump(catalog, f)

print("Vector database created successfully!")