
import json
import sys
from pathlib import Path

import numpy as np
import faiss
import pymupdf
from sentence_transformers import SentenceTransformer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

MITRE_DIR = BASE_DIR / "data" / "rag_documents" / "mitre_attack"
CISA_DIR = BASE_DIR / "data" / "rag_documents" / "cisa"
CSK_DIR = BASE_DIR / "data" / "rag_documents" / "csk"

OUT_DIR = BASE_DIR / "rag" / "rag_store"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_WORDS = 150
OVERLAP = 30


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()

    if not words:
        return []

    step = size - overlap
    if step <= 0:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []

    for i in range(0, len(words), step):
        piece = words[i:i + size]

        if len(piece) >= 30:
            chunks.append(" ".join(piece))

        if i + size >= len(words):
            break

    return chunks


def tag_attack_type(filename):
    name = filename.lower()

    # Botnet-related MITRE techniques
    if "botnet" in name or "botenago" in name or name.startswith(("t1583_005", "t1584_005")):
        return "Botnet"

    # DDoS-related MITRE techniques
    if "ddos" in name or name.startswith(("t1498", "t1499")):
        return "DDoS"

    return "General"


def tag_csk_attack_type(text):
    return "Botnet" if "botnet" in text.lower() else "General"


records = []
chunks_per_file = {}

# -------------------------------
# 1. Read MITRE text documents
# -------------------------------
if MITRE_DIR.exists():

    for file in sorted(MITRE_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_words(text)
        chunks_per_file[file.relative_to(BASE_DIR).as_posix()] = len(chunks)

        for chunk in chunks:
            records.append({
                "text": chunk,
                "source": file.name,
                "page": None,
                "attack_type": tag_attack_type(file.name)
            })

# -------------------------------
# 2. Read CSK text documents
# -------------------------------
if CSK_DIR.exists():

    for file in sorted(CSK_DIR.glob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_words(text)
        chunks_per_file[file.relative_to(BASE_DIR).as_posix()] = len(chunks)
        attack_type = tag_csk_attack_type(text)

        for chunk in chunks:
            records.append({
                "text": chunk,
                "source": file.name,
                "page": None,
                "attack_type": attack_type
            })

# -------------------------------
# 3. Read CISA PDF documents
# -------------------------------
if CISA_DIR.exists():

    for file in sorted(CISA_DIR.glob("*.pdf")):

        file_chunk_count = 0
        with pymupdf.open(file) as pdf:

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                chunks = chunk_words(text)
                file_chunk_count += len(chunks)

                for chunk in chunks:
                    records.append({
                        "text": chunk,
                        "source": file.name,
                        "page": page_number,
                        "attack_type": tag_attack_type(file.name)
                    })
        chunks_per_file[file.relative_to(BASE_DIR).as_posix()] = file_chunk_count

# -------------------------------
# 4. Validate extracted chunks
# -------------------------------
if not records:
    raise ValueError(
        "No chunks generated. Check MITRE, CSK, and CISA document folders."
    )

print("\nChunks per file:")
for source, chunk_count in chunks_per_file.items():
    print(f"{source}: {chunk_count}")

zero_chunk_pdfs = [
    source
    for source, chunk_count in chunks_per_file.items()
    if source.lower().endswith(".pdf") and chunk_count == 0
]
print("\nPDFs with zero chunks:")
for source in zero_chunk_pdfs or ["None"]:
    print(source)

print("Total chunks generated:", len(records))

for record in records[:3]:
    print("\nSource:", record["source"])
    print("Attack type:", record["attack_type"])
    preview = record["text"][:200].encode("ascii", "backslashreplace").decode("ascii")
    print("Text:", preview)

# -------------------------------
# 4. Generate embeddings
# -------------------------------
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [record["text"] for record in records]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

embeddings = np.asarray(embeddings, dtype="float32")

# -------------------------------
# 5. Build FAISS index
# -------------------------------
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# -------------------------------
# 6. Save index and metadata
# -------------------------------
faiss.write_index(index, str(OUT_DIR / "kb.index"))

with open(OUT_DIR / "chunks.json", "w", encoding="utf-8") as file:
    json.dump(records, file, indent=2, ensure_ascii=False)

print("\nIndexing completed successfully!")
print("Total indexed chunks:", index.ntotal)
print("Embedding dimensions:", embeddings.shape[1])
print("Saved location:", OUT_DIR)
