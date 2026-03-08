# embeddings_store.py
# Verwaltet den ChromaDB Vektorstore für Rechnungs-Chunks.
# Wird von main.py (upsert_invoice_text) und retrieval.py (collection.query) genutzt.
# Das Embedding-Modell (_embedder) ist eine Singleton-Instanz – sowohl für Indexierung
# als auch für Abfragen, damit die Vektoren vergleichbar bleiben.

import hashlib
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, CHROMA_COLLECTION

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

_client = chromadb.Client(Settings(persist_directory=CHROMA_DIR, is_persistent=True))

collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=_embedder,
    metadata={"hnsw:space": "cosine"},
)

def get_collection():
    return collection

def split_into_chunks(text: str, max_chars: int = 900, overlap: int = 200) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def upsert_invoice_text(filename: str, full_text: str, base_fields: Dict[str, Any]) -> None:
    doc_hash = hashlib.sha256(full_text.encode()).hexdigest()

    # Hash-basierter Cache: PDFs werden nicht neu indexiert solange sich ihr Inhalt nicht ändert
    existing = collection.get(where={"doc_hash": doc_hash})
    if existing and existing.get("ids"):
        print(f"⚡ Cache-Hit: {filename} bereits indexiert, überspringe.")
        return

    chunks = split_into_chunks(full_text)
    ids = [f"{filename}::chunk::{i}" for i in range(len(chunks))]

    metadatas: List[Dict[str, Any]] = []
    for _ in chunks:
        md: Dict[str, Any] = {"source": filename, "doc_hash": doc_hash}
        for k in ("rechnungsnummer", "rechnungsdatum", "waehrung"):
            if base_fields.get(k):
                md[k] = str(base_fields[k])
        metadatas.append(md)

    embeddings = _embedder(chunks)

    # Erst löschen, dann neu hinzufügen – sonst bleiben veraltete Chunks für dieselbe Datei übrig
    try:
        collection.delete(where={"source": filename})
    except Exception:
        pass

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    print(f"✅ {filename}: {len(chunks)} Chunks mit Embeddings hinzugefügt.")
