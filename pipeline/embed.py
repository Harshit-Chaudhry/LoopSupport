"""
pipeline/embed.py

Builds (or rebuilds) the FAISS retrieval index from data/clean/tickets.csv
and data/clean/kb_docs.csv. Also exposes the Embedder class that
pipeline/rag.py will reuse to embed a live incoming query — same reasoning
as PIIAnonymizer: the embedding model should be loaded ONCE (at FastAPI
startup) and reused across every request, not reloaded per call.

Retrieval corpus = all rows in tickets.csv (already excludes holdout,
since holdout lives in its own separate holdout.csv — the file split
itself is what guarantees holdout tickets never enter the index) +
all rows in kb_docs.csv.

For tickets, we embed `customer_text` — retrieval is meant to find past
tickets whose CUSTOMER MESSAGE is similar to the incoming one. The full
record (customer_text + agent_response) is kept in metadata.json so
pipeline/rag.py has the actual resolved answer to build a prompt from,
not just the similarity match.

For kb_docs, we embed `content` (title is prepended for a bit of extra
signal), full doc kept in metadata for the same reason.

ENTERPRISE MIGRATION NOTE: EMBEDDING_MODEL is an env var for the same
reason DATABASE_URL is in api/db.py and the model config is in
pipeline/llm.py — swapping to a hosted embedding API in production means
changing this one value, not the calling code in rag.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "data" / "clean"
INDEX_DIR = REPO_ROOT / "data" / "faiss_index"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")  # cpu by default —
# reserves the 6GB VRAM budget for Phi-4-mini per the hardware plan; this
# model is small enough that CPU is fine at prototype scale.


class Embedder:
    """
    Wraps the sentence-transformer model. Load once, reuse everywhere —
    in api/dependencies.py this gets instantiated once at startup
    alongside the PIIAnonymizer and LLM singletons, then:
      - pipeline/embed.py uses it for batch corpus embedding
      - pipeline/rag.py uses the SAME instance to embed each live query
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = EMBEDDING_DEVICE):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns L2-normalized embeddings — normalized so FAISS
        IndexFlatIP (inner product) is equivalent to cosine similarity."""
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # guard against a zero vector on empty/degenerate text
        return vecs / norms


# --- Corpus construction ------------------------------------------------

def build_corpus(tickets_path: Path, kb_docs_path: Path):
    """
    Returns (texts, metadata) — parallel lists. texts[i] is what gets
    embedded, metadata[i] is what gets stored/returned on retrieval hit.
    """
    texts = []
    metadata = []

    if tickets_path.exists():
        df = pd.read_csv(tickets_path)
        for _, row in df.iterrows():
            customer_text = row.get("customer_text")
            if pd.isna(customer_text) or not str(customer_text).strip():
                continue  # nothing to embed for this row — skip rather than index a blank
            texts.append(str(customer_text))
            metadata.append({
                "type": "ticket",
                "id": row.get("ticket_id"),
                "customer_text": str(customer_text),
                "agent_response": None if pd.isna(row.get("agent_response")) else str(row.get("agent_response")),
                "product": None if pd.isna(row.get("product")) else str(row.get("product")),
                "issue_type": None if pd.isna(row.get("issue_type")) else str(row.get("issue_type")),
                "is_gold_example": bool(row.get("is_gold_example")) if not pd.isna(row.get("is_gold_example")) else False,
            })
    else:
        print(f"  ⚠ {tickets_path} not found — no tickets in corpus")

    if kb_docs_path.exists():
        df = pd.read_csv(kb_docs_path)
        for _, row in df.iterrows():
            content = row.get("content")
            if pd.isna(content) or not str(content).strip():
                continue
            title = "" if pd.isna(row.get("title")) else str(row.get("title"))
            embed_text = f"{title}. {content}" if title else str(content)
            texts.append(embed_text)
            metadata.append({
                "type": "kb_doc",
                "id": row.get("doc_id"),
                "title": title,
                "content": str(content),
                "product": None if pd.isna(row.get("product")) else str(row.get("product")),
                "doc_type": None if pd.isna(row.get("doc_type")) else str(row.get("doc_type")),
            })
    else:
        print(f"  ⚠ {kb_docs_path} not found — no KB docs in corpus")

    return texts, metadata


# --- Index build / save / load ------------------------------------------

def build_index(embedder: Embedder, texts: list[str]):
    if not texts:
        raise ValueError("Corpus is empty — nothing to index. Check data/clean/ contents.")
    vecs = embedder.embed(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs.astype(np.float32))
    return index


def save_index(index, metadata: list[dict], index_dir: Path = INDEX_DIR):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index(index_dir: Path = INDEX_DIR):
    """Used by pipeline/rag.py at query time — loads the prebuilt index,
    does NOT recompute embeddings."""
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No index found at {index_dir} — run `python3 pipeline/embed.py` first."
        )
    index = faiss.read_index(str(index_path))
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL} (device={EMBEDDING_DEVICE})")
    embedder = Embedder()

    texts, metadata = build_corpus(
        CLEAN_DIR / "tickets.csv",
        CLEAN_DIR / "kb_docs.csv",
    )
    print(f"Corpus size: {len(texts)} entries "
          f"({sum(1 for m in metadata if m['type']=='ticket')} tickets, "
          f"{sum(1 for m in metadata if m['type']=='kb_doc')} kb_docs)")

    index = build_index(embedder, texts)
    save_index(index, metadata)
    print(f"Index saved to {INDEX_DIR} ({index.ntotal} vectors, dim={index.d})")


if __name__ == "__main__":
    main()