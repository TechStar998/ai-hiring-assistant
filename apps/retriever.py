from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SHLRetriever:
    

    def __init__(
        self,
        index_path: str = "vectorstore/shl_index.faiss",
        catalog_path: str = "vectorstore/catalog.pkl",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.catalog_path = Path(catalog_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog file not found: {self.catalog_path}"
            )

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print("Loading catalog metadata...")
        with open(self.catalog_path, "rb") as f:
            self.catalog = pickle.load(f)

        print(f"Retriever ready ({len(self.catalog)} assessments loaded).")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ):
        """
        Returns top-k matching assessments.

        Returns:
            List[dict]
        """

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        distances, indices = self.index.search(
            embedding,
            top_k,
        )

        results = []

        for score, idx in zip(distances[0], indices[0]):

            if idx == -1:
                continue

            assessment = dict(self.catalog[idx])

            assessment["similarity_score"] = float(score)

            results.append(assessment)

        return results


if __name__ == "__main__":

    retriever = SHLRetriever()

    query = "Hiring a Java backend developer with communication skills"

    results = retriever.search(query)

    print("\nTop Results\n")

    for i, item in enumerate(results, start=1):
        print("=" * 60)
        print(i)
        print(item.get("name"))
        print(item.get("similarity_score"))