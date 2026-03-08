# api_server.py
# FastAPI-Server für den Chat-Endpunkt. Liest ausschließlich aus den von main.py
# erzeugten JSON-Dateien in INVOICE_JSON_DIR – keine direkte Pipeline-Logik hier.
# Endpunkte: GET /health, GET /invoices, GET /invoices/{id}, POST /chat
# Chat-Ablauf: Rechnung(en) identifizieren → RAG aus ChromaDB → LLM → deterministischer Fallback

import os, json, re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from main import INVOICE_JSON_DIR
from llm_client import ollama_is_up, ollama_chat
from config import OLLAMA_MODEL_CHAT

# ChromaDB beim Start laden – bei Fehler (z.B. noch keine PDFs verarbeitet) trotzdem hochfahren
try:
    from embeddings_store import get_collection
    _chroma_collection = get_collection()
    print("✅ ChromaDB vorgeladen")
except Exception as e:
    _chroma_collection = None
    print(f"⚠️ ChromaDB nicht verfügbar: {e}")

# -------------------------
# App
# -------------------------
app = FastAPI(title="Invoice Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später domain einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers: Files
# -------------------------
def _list_invoice_json_files() -> List[str]:
    if not os.path.isdir(INVOICE_JSON_DIR):
        return []
    return sorted([f for f in os.listdir(INVOICE_JSON_DIR) if f.lower().endswith(".json")])

def _invoice_id_from_filename(fn: str) -> str:
    return fn[:-5] if fn.lower().endswith(".json") else fn

def _load_invoice_json_by_id(invoice_id: str) -> Dict[str, Any]:
    path = os.path.join(INVOICE_JSON_DIR, invoice_id + ".json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Invoice not found: {invoice_id}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dedup_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _safe_str(x: Any) -> str:
    return str(x or "").strip()

# Reihenfolge wichtig: längere/spezifischere Strings zuerst
_COUNTRY_KEYWORDS = [
    ("niederlande", "Niederlande"), ("netherlands", "Niederlande"), ("holland", "Niederlande"),
    ("vereinigte staaten", "USA"), ("united states", "USA"),
    ("österreich", "Österreich"), ("austria", "Österreich"),
    ("schweiz", "Schweiz"), ("switzerland", "Schweiz"),
    ("frankreich", "Frankreich"), ("france", "Frankreich"),
    ("großbritannien", "Großbritannien"), ("united kingdom", "Großbritannien"),
    ("deutschland", "Deutschland"), ("germany", "Deutschland"),
    ("usa", "USA"),
]

def _detect_country(address: str) -> str:
    """Land aus Adresstext ableiten: explizite Ländernamen oder 5-stellige PLZ → Deutschland."""
    addr_lower = (address or "").lower()
    for keyword, country in _COUNTRY_KEYWORDS:
        if keyword in addr_lower:
            return country
    if re.search(r'\b\d{5}\b', address or ""):
        return "Deutschland"
    return "unbekannt"


# Erkennt Queries die sich auf alle oder mehrere Rechnungen beziehen
_ALL_INVOICES_RE = re.compile(
    r'\b('
    r'welche\s+rechnungen|'
    r'alle[nr]?\s+(rechnungen?|unternehmen?|firmen?|absender)|'
    r'gesamt(summe|betrag|übersicht|aller)|'
    r'summier[et]\s+alle|'
    r'rechnungen?\s+(vergleich|insgesamt|übersicht)|'
    r'aller\s+rechnungen?|'
    r'deutschen\s+unternehmen|'
    r'welche\s+firmen|'
    r'überblick\s+über\s+alle'
    r')\b',
    re.IGNORECASE,
)

def _is_all_invoices_query(message: str) -> bool:
    return bool(_ALL_INVOICES_RE.search(message or ""))

# -------------------------
# Models
# -------------------------
class ChatRequest(BaseModel):
    message: str
    invoiceIds: Optional[List[str]] = None
    use_llm: bool = True
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    matched_invoices: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

# -------------------------
# Deterministic helpers (Fallback)
# -------------------------
def _match_query_to_fields(msg: str) -> Dict[str, Any]:
    m = (msg or "").lower()
    return {
        "sum_brutto": bool(re.search(r"(summe|gesamt|insgesamt).*(brutto|total)", m)),
        "sum_netto": bool(re.search(r"(summe|gesamt|insgesamt).*(netto)", m)),
        "tax": bool(re.search(r"(mwst|ust|vat|steuer)", m)),
        "positions": bool(re.search(r"(position|posten|items|leistungen)", m)),
        "sender": bool(re.search(r"(absender|vendor|seller|wer.*gestellt)", m)),
        "buyer": bool(re.search(r"(empf[aä]nger|kunde|buyer|bill to)", m)),
        "date": bool(re.search(r"(datum|date|rechnungsdatum)", m)),
        "invoice_no": bool(re.search(r"(rechnungsnummer|invoice\s*(no|number))", m)),
        "all": bool(re.search(r"\b(alle|insgesamt|gesamt|summiere)\b", m)),
    }

def _compute_answer_from_invoice(final: Dict[str, Any], wants: Dict[str, Any]) -> str:
    parts = []
    if wants["invoice_no"]:
        parts.append(f"Rechnungsnummer: {final.get('rechnungsnummer')}")
    if wants["date"]:
        parts.append(f"Rechnungsdatum: {final.get('rechnungsdatum')}")
    if wants["sender"]:
        parts.append(f"Absender:\n{final.get('absender')}")
    if wants["buyer"]:
        parts.append(f"Empfänger:\n{final.get('empfaenger')}")
    if wants["sum_netto"]:
        parts.append(f"Netto: {final.get('netto')} {final.get('waehrung')}")
    if wants["tax"]:
        parts.append(f"Steuer: {final.get('steuer')} {final.get('waehrung')}")
    if wants["sum_brutto"]:
        parts.append(f"Brutto: {final.get('brutto')} {final.get('waehrung')}")
    if wants["positions"]:
        pos = final.get("positionen") or []
        if not pos:
            parts.append("Keine Positionen gefunden.")
        else:
            lines = [f"- {p.get('beschreibung')}: {p.get('zeilensumme')}" for p in pos[:15]]
            parts.append("Positionen:\n" + "\n".join(lines))

    if not parts:
        parts.append(
            f"Rechnung {final.get('rechnungsnummer')} vom {final.get('rechnungsdatum')} "
            f"({final.get('waehrung')}): Brutto {final.get('brutto')}, "
            f"Netto {final.get('netto')}, Steuer {final.get('steuer')}."
        )
    return "\n\n".join(parts)

# -------------------------
# LLM helpers
# -------------------------
def _summarize_invoice_for_llm(inv_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    final = data.get("final") or {}
    positionen = (final.get("positionen") or [])[:30]
    absender = final.get("absender") or ""
    return {
        "id": inv_id,
        "filename": data.get("filename"),
        "rechnungsnummer": final.get("rechnungsnummer"),
        "rechnungsdatum": final.get("rechnungsdatum"),
        "waehrung": final.get("waehrung"),
        "netto": final.get("netto"),
        "steuer": final.get("steuer"),
        "versand": final.get("versand"),
        "brutto": final.get("brutto"),
        "absender": absender,
        "absender_land": _detect_country(absender),
        "empfaenger": final.get("empfaenger"),
        "positionen_anzahl": len(positionen),
        "positionen": positionen,
        "validiert": final.get("validiert"),
    }

# Generische deutsche Stoppwörter und das Wort "Rechnung" selbst (erscheint in jedem
# Rechnungskontext) werden aus den Query-Tokens herausgefiltert, damit sie keinen
# fälschlichen Score-Bonus auf zufällig passende Rechnungen geben.
_STOP_WORDS = {
    "die", "der", "das", "ein", "eine", "einer", "eines", "einem", "den", "dem",
    "und", "oder", "aber", "hat", "ist", "sind", "von", "auf", "mit", "fur",
    "bei", "aus", "zur", "zum", "nach", "seit", "vor", "uber", "unter", "wie",
    "wer", "was", "wen", "wem", "welche", "welcher", "welches", "welchem", "welchen",
    "zeig", "zeige", "gibt", "bitte", "kann", "alle", "alles", "mir", "mich",
    "ich", "wir", "sie", "uns", "ihr", "bin", "war", "sich", "auch", "nicht",
    "mehr", "nur", "noch", "mal", "schon", "immer", "sehr", "ganz", "dann",
    "rechnung", "rechnungen",  # generisches Wort für Invoice – nicht spezifisch genug
}


def _select_invoices_by_message(files: List[str], message: str, limit: int = 6,
                                history: Optional[List[Dict[str, str]]] = None) -> List[str]:
    all_text = message or ""
    if history:
        for h in history[-4:]:
            all_text += " " + h.get("content", "")

    msg_l = all_text.lower()
    raw_tokens = [t for t in re.split(r"\W+", msg_l) if len(t) >= 3]

    # Rechnungsnummer-Patterns immer behalten (z.B. RE250004, BCE-0001)
    invoice_no_tokens = re.findall(r'\b(RE\d+|BCE\w+|\w+-\d+)\b', all_text, re.I)

    # Stoppwörter rausfiltern – nur bedeutungstragende Tokens behalten
    tokens = [t for t in raw_tokens if t not in _STOP_WORDS]
    tokens += [t.lower() for t in invoice_no_tokens]

    if not tokens:
        return [_invoice_id_from_filename(files[0])] if files else []

    scores: Dict[str, int] = {}
    for fn in files:
        inv_id = _invoice_id_from_filename(fn)
        data = _load_invoice_json_by_id(inv_id)
        final = data.get("final") or {}

        # Dateiname-Tokens: spezifischstes Signal → 5× Gewicht
        fn_tokens = {
            t.lower()
            for t in re.split(r"[\W_\d]+", data.get("filename", ""))
            if len(t) >= 3 and t.lower() not in _STOP_WORDS
        }

        pos_text = " ".join(
            _safe_str(p.get("beschreibung"))
            for p in (final.get("positionen") or [])
        )
        content_hay = _dedup_spaces(" ".join([
            _safe_str(final.get("rechnungsnummer")),
            _safe_str(final.get("rechnungsdatum")),
            _safe_str(final.get("absender")),
            _safe_str(final.get("empfaenger")),
            _safe_str(final.get("waehrung")),
            pos_text,
        ])).lower()

        score = 0
        for t in tokens:
            if t in fn_tokens:
                score += 5  # Dateiname-Match: sehr spezifisch
            elif t in content_hay:
                score += 1  # Inhalts-Match: weniger spezifisch

        if score > 0:
            scores[inv_id] = score

    if not scores:
        return [_invoice_id_from_filename(files[0])] if files else []

    best_score = max(scores.values())
    best = max(scores, key=lambda k: scores[k])

    # Eindeutiger oder klarer Gewinner (höchster Score allein oben)
    tied_best = [k for k, v in scores.items() if v == best_score]
    if len(tied_best) == 1:
        return [best]

    # Mehrere gleich starke Treffer: top 3 zurückgeben
    top = sorted(scores, key=lambda k: scores[k], reverse=True)[:3]
    return top

def _list_all_invoices_summary() -> str:
    lines = []
    for fname in _list_invoice_json_files():
        inv_id = fname.replace(".json", "")
        try:
            data = _load_invoice_json_by_id(inv_id)
            f = data.get("final") or {}
            absender = _safe_str(f.get("absender"))
            land = _detect_country(absender)
            lines.append(
                f"- {inv_id}: "
                f"Absender={absender.split(chr(10))[0]}, "
                f"Land={land}, "
                f"Empfaenger={_safe_str(f.get('empfaenger')).split(chr(10))[0]}, "
                f"Nr={f.get('rechnungsnummer', '?')}, "
                f"Datum={f.get('rechnungsdatum', '?')}, "
                f"Waehrung={f.get('waehrung', '?')}, "
                f"Netto={f.get('netto', '?')}, "
                f"Steuer={f.get('steuer', '?')}, "
                f"Brutto={f.get('brutto', '?')}"
            )
        except Exception:
            lines.append(f"- {inv_id}: (Fehler beim Laden)")
    return "\n".join(lines)


def _is_readable_chunk(text: str, min_ratio: float = 0.5) -> bool:
    """Garbled PDF-Chunks (zeichenweise Zeilenumbrüche) herausfiltern.
    Misst den Anteil normaler Wörter (>=3 Zeichen) an der Gesamtzeichenzahl."""
    words = re.findall(r'\b\w{3,}\b', text)
    if not words:
        return False
    word_chars = sum(len(w) for w in words)
    return (word_chars / max(len(text), 1)) >= min_ratio


def _retrieve_chat_context(query: str, invoice_ids: List[str]) -> List[str]:
    try:
        if _chroma_collection is None:
            return []
        results = _chroma_collection.query(
            query_texts=[query],
            n_results=5,
            where={"source": {"$in": invoice_ids}},
        )
        docs = results.get("documents") or []
        raw_chunks = docs[0] if docs else []
        # Garbled Chunks (zeichenweise aus PDF) herausfiltern
        chunks = [c[:500] for c in raw_chunks if _is_readable_chunk(c)]
        return chunks[:3]
    except Exception:
        return []


def _llm_answer(message: str, invoice_summaries: List[Dict[str, Any]], model: str,
                history: Optional[List[Dict[str, str]]] = None,
                rag_chunks: Optional[List[str]] = None) -> str:
    all_invoices = _list_all_invoices_summary()

    system = (
        "Du bist ein Rechnungs-Assistent. Antworte vollständig, klar und korrekt.\n"
        "WICHTIG:\n"
        "- Alle Daten die du brauchst stehen in `invoices[]` (strukturierte JSON-Daten) "
        "und in `all_invoices_overview` (Kurzübersicht aller Rechnungen).\n"
        "- Die Positionen einer Rechnung findest du unter invoices[].positionen – "
        "liste sie IMMER vollständig auf wenn danach gefragt wird.\n"
        "- Sage NIEMALS, dass Daten fehlen oder nicht vorhanden sind, wenn sie in "
        "invoices[].positionen, invoices[].absender, invoices[].empfaenger oder "
        "einem anderen Feld von invoices[] stehen.\n"
        "- Erfinde keine Werte. Wenn ein Feld wirklich null/leer ist: sag es kurz.\n"
        "- Wenn die Frage mehrere Rechnungen betrifft: gib eine strukturierte Übersicht.\n"
        "- Wenn der Nutzer eine bestimmte Rechnung meint: identifiziere sie über "
        "Rechnungsnummer, Datum, Absender oder Empfänger.\n"
        "- Antworte auf Deutsch.\n"
        f"\nAlle verfügbaren Rechnungen (Übersicht):\n{all_invoices}\n"
    )

    if rag_chunks:
        rag_text = "\n\n".join(f"[Auszug {i+1}]: {chunk}" for i, chunk in enumerate(rag_chunks))
        system += f"\nRelevante Textauszüge aus den Original-PDFs:\n{rag_text}\n"

    user_obj = {
        "question": message,
        "invoices": invoice_summaries,
        "output_rules": [
            "Keine erfundenen Werte.",
            "Wenn unsicher: Rückfrage stellen oder 'nicht ersichtlich' sagen.",
            "Wenn Summen gefragt: rechne nur über vorhandene Zahlen.",
        ],
    }

    recent_history = (history or [])[-6:]
    messages = [{"role": "system", "content": system}]
    for entry in recent_history:
        role = entry.get("role", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": entry.get("content", "")})
    messages.append({"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)})

    resp = ollama_chat(
        model=model,
        messages=messages,
        temperature=0.0,
        timeout_s=120,
        num_predict=800,
        retries=1,
    )
    return ((resp.get("message") or {}).get("content") or "").strip()


@app.get("/health")
def health():
    data = {
        "status": "ok",
        "ollama_up": ollama_is_up(),
        "invoice_json_dir": INVOICE_JSON_DIR,
        "count": len(_list_invoice_json_files()),
    }
    return JSONResponse(content=data, media_type="application/json; charset=utf-8")

@app.get("/invoices")
def list_invoices():
    out = []
    for fn in _list_invoice_json_files():
        inv_id = _invoice_id_from_filename(fn)
        data = _load_invoice_json_by_id(inv_id)
        final = data.get("final") or {}
        out.append({
            "id": inv_id,
            "filename": data.get("filename"),
            "rechnungsnummer": final.get("rechnungsnummer"),
            "rechnungsdatum": final.get("rechnungsdatum"),
            "absender": (final.get("absender") or "")[:120],
            "empfaenger": (final.get("empfaenger") or "")[:120],
            "waehrung": final.get("waehrung"),
            "netto": final.get("netto"),
            "steuer": final.get("steuer"),
            "brutto": final.get("brutto"),
            "validiert": final.get("validiert"),
        })
    return JSONResponse(content=out, media_type="application/json; charset=utf-8")

@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    data = _load_invoice_json_by_id(invoice_id)
    return JSONResponse(content=data, media_type="application/json; charset=utf-8")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    files = _list_invoice_json_files()
    if not files:
        raise HTTPException(status_code=400, detail="No processed invoices found. Run pipeline first.")

    # 1) invoiceIds aus Frontend oder per Suche bestimmen
    invoice_ids = (req.invoiceIds or [])
    if not invoice_ids:
        if _is_all_invoices_query(req.message):
            # Aggregationsabfragen: alle Rechnungen laden damit das LLM vollständig rechnen kann
            invoice_ids = [_invoice_id_from_filename(f) for f in files]
        else:
            invoice_ids = _select_invoices_by_message(files, req.message, limit=6, history=req.history)

    # 2) invoices laden + zusammenfassen
    invoice_summaries = []
    matched = []
    sources = []
    for inv_id in invoice_ids:
        data = _load_invoice_json_by_id(inv_id)
        invoice_summaries.append(_summarize_invoice_for_llm(inv_id, data))
        final = data.get("final") or {}
        matched.append({"id": inv_id, "rechnungsnummer": final.get("rechnungsnummer")})
        sources.append({"invoice_id": inv_id, "filename": data.get("filename")})

    # RAG: Semantische Suche in ChromaDB
    rag_chunks = _retrieve_chat_context(
        req.message,
        [inv["filename"] for inv in invoice_summaries if inv.get("filename")],
    )

    # 3) LLM wenn gewünscht + verfügbar
    model = OLLAMA_MODEL_CHAT
    if req.use_llm and ollama_is_up():
        answer = _llm_answer(req.message, invoice_summaries, model=model, history=req.history,
                             rag_chunks=rag_chunks)
        if answer:
            return ChatResponse(answer=answer, matched_invoices=matched, sources=sources)

    # Deterministischer Fallback: wenn Ollama nicht läuft oder eine leere Antwort kommt
    wants = _match_query_to_fields(req.message)
    # wenn "alle + summe brutto" gefragt -> aggregieren über ausgewählte
    if wants["all"] and wants["sum_brutto"]:
        total = 0.0
        cur = None
        cnt = 0
        for inv in invoice_summaries:
            b = inv.get("brutto")
            if b is not None:
                total += float(b)
                cur = cur or inv.get("waehrung")
                cnt += 1
        ans = f"Gesamt-Brutto über {cnt} Rechnungen: {round(total,2)} {cur or ''}".strip()
        return ChatResponse(answer=ans, matched_invoices=matched, sources=sources)

    # sonst: erste Rechnung
    first = invoice_summaries[0]
    ans = _compute_answer_from_invoice(first, wants)
    return ChatResponse(answer=ans, matched_invoices=matched[:1], sources=sources[:1])
