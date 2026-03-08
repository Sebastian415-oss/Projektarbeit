# config.py
# Zentrale Konfiguration – alle Pfade und Modellnamen an einem Ort.
# Alle Werte sind per Umgebungsvariable überschreibbar (12-Factor-App-Stil).
# Wird von fast allen anderen Modulen importiert.

import os

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "invoices")

# OLLAMA_MODEL: das "schwere" Modell für die Extraktions-Pipeline (main.py)
# OLLAMA_MODEL_CHAT: kleineres Modell für den Chat-Endpunkt (api_server.py) – schnellere Antwortzeiten
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "qwen2.5:3b-instruct")

# Outputs
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
INVOICE_JSON_DIR = os.path.join(OUTPUT_DIR, "invoices")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "invoice_results.csv")

PRIORS_FILE = os.getenv("PRIORS_FILE", "priors.json")
