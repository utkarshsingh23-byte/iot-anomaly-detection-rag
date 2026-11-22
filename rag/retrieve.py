
import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
STORE_DIR = BASE_DIR / "rag" / "rag_store"

INDEX_PATH = STORE_DIR / "kb.index"
CHUNKS_PATH = STORE_DIR / "chunks.json"

if not INDEX_PATH.exists():
    raise FileNotFoundError("Index not found. Run: python rag/build_index.py first")

# Load embedding model
print("Loading embedding model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
_index = faiss.read_index(str(INDEX_PATH))

# Load chunk metadata
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    _chunks = json.load(file)


def retrieve(query, k=3, attack_type=None):

    if not query.strip():
        return []

    if _index.ntotal == 0:
        return []

    # Generate query embedding
    query_embedding = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding, dtype="float32"
    )

    # Search FAISS
    search_count = min(k * 6, _index.ntotal)

    scores, ids = _index.search(
        query_embedding, search_count
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        # Ignore invalid FAISS IDs
        if idx < 0:
            continue

        chunk = _chunks[idx]

        # Filter by attack type
        if attack_type:
            if chunk["attack_type"].lower() not in (
                attack_type.lower(), "general"
            ):
                continue

        results.append({
            **chunk,
            "score": float(score)
        })

        if len(results) >= k:
            break

    return results


if __name__ == "__main__":

    test_queries = [
        (
            "DDoS high SYN count short inter-arrival time",
            "DDoS"
        ),
        (
            "Botnet command and control DNS beaconing",
            "Botnet"
        )
    ]

    for query, attack_type in test_queries:

        print("\nQUERY:", query)

        results = retrieve(
            query,
            k=3,
            attack_type=attack_type
        )

        for result in results:
            print(
                f"\nScore: {result['score']:.3f}"
                f"\nSource: {result['source']}"
                f"\nPage: {result['page']}"
                f"\nType: {result['attack_type']}"
            )

            preview = result["text"][:250].encode("ascii", "backslashreplace").decode("ascii")
            print("Text:", preview, "...")
