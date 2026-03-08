# retrieval.py
# Semantische Suche in ChromaDB: findet die relevantesten Chunks pro Rechnungsfeld
# (netto/steuer/versand/brutto) und liefert Beträge + Evidence-Texte zurück.
# retrieve_fix_fields() wird von main.py als zweite Extraktionsschicht nach dem
# Basis-Parser genutzt. retrieve_evidence_chunks() liefert Kontext für das LLM.

import re
from typing import Dict, List, Optional
from embeddings_store import collection
from parsing import _normalize_amount

FIELD_QUERIES = {
    "netto":   ["Nettobetrag", "Netto", "Zwischensumme (netto)", "Gesamt Netto", "Subtotal", "Zwischensumme netto"],
    "steuer":  ["Umsatzsteuer", "Umsatzsteuer 19", "MwSt", "Mehrwertsteuer", "USt", "VAT", "Tax amount"],
    "versand": ["Versandkosten", "Versand", "Lieferkosten", "Porto", "Shipping"],
    "brutto":  ["Gesamtbetrag", "Rechnungsbetrag", "Brutto", "Total", "Summe gesamt", "Gesamtsumme", "Total amount"],
}


AMOUNT_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2}))\s*(€|EUR|\$|USD)?", re.I)

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

@dataclass
class RetrievalHit:
    value: float
    dist: float
    matched_keyword: str
    evidence: str

def _compact_lines_around_keywords(doc: str, keywords: List[str], max_lines: int = 8, max_chars: int = 500) -> str:
    lines = doc.splitlines()
    idxs = []
    low = doc.lower()
    for kw in keywords:
        kwl = kw.lower()
        for i, line in enumerate(lines):
            if kwl in line.lower():
                idxs.append(i)
    if not idxs:
        return doc[:max_chars]

    center = idxs[0]
    start = max(0, center - max_lines//2)
    end = min(len(lines), start + max_lines)
    snippet = "\n".join(lines[start:end]).strip()
    return snippet[:max_chars]

def _extract_amounts_with_positions(s: str) -> List[Tuple[str, Optional[str], int]]:
    out: List[Tuple[str, Optional[str], int]] = []
    for m in AMOUNT_RE.finditer(s):
        out.append((m.group(1), m.group(2), m.start()))
    return out

def retrieve_fix_fields(
    filename: str,
    currency_hint: Optional[str] = None,
    top_k: int = 3,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Liefert:
      - netto/steuer/versand/brutto als float oder None
      - zusätzlich _meta[field] mit evidence + matched_keyword + dist
    """
    fixed: Dict[str, Any] = {
        "netto": None,
        "steuer": None,
        "versand": None,
        "brutto": None,
        "_meta": {},  # field -> {value, dist, keyword, evidence}
    }

    # 0.65 ist empirisch: bei Cosinus-Distanz bedeutet das noch "semantisch ähnlich".
    # Chunks mit dist > 0.65 sind zu weit weg und liefern meist falschen Kontext.
    DIST_THRESHOLD = 0.65  # cosine distance: kleiner = besser

    for field, queries in FIELD_QUERIES.items():
        best: Optional[RetrievalHit] = None

        if debug:
            print(f"\n[retrieve_fix_fields] Feld: {field}")

        for q in queries:
            qtext = f"{q} {currency_hint or ''}".strip()

            if debug:
                print(f"  🔎 Query-Text: '{qtext}'")

            try:
                res = collection.query(
                    query_texts=[qtext],
                    n_results=top_k,
                    where={"source": filename},
                )
            except Exception as e:
                if debug:
                    print(f"⚠️ Chroma-Query-Fehler: {e}")
                continue

            docs_list = res.get("documents", [[]])
            dists_list = res.get("distances", [[]])
            docs = docs_list[0] if docs_list else []
            dists = dists_list[0] if dists_list else [None] * len(docs)

            for doc, dist in zip(docs, dists):
                dist_val = float(dist) if dist is not None else 0.0
                if dist is not None and dist_val > DIST_THRESHOLD:
                    continue

                # evidence stark kürzen
                evidence = _compact_lines_around_keywords(doc, queries, max_lines=8, max_chars=500)

                # NUR Beträge NACH dem Keyword im evidence suchen (field-spezifisch)
                low = evidence.lower()
                for kw in queries:
                    kwl = kw.lower()
                    pos = low.find(kwl)
                    if pos == -1:
                        continue

                    window = evidence[pos : min(len(evidence), pos + 220)]
                    # Beträge finden
                    for raw, cur, p in _extract_amounts_with_positions(window):
                        # Prozentwerte ignorieren
                        around = window[max(0, p - 6):min(len(window), p + len(raw) + 6)]
                        if "%" in around:
                            continue

                        val = _normalize_amount((raw + (f" {cur}" if cur else "")).strip())
                        if val is None or val < 0:
                            continue

                        hit = RetrievalHit(
                            value=val,
                            dist=dist_val,
                            matched_keyword=kw,
                            evidence=evidence,
                        )

                        # Niedrigste Distanz gewinnt; 1e-6 Toleranz um Float-Gleichheit zu vermeiden
                        if best is None:
                            best = hit
                        else:
                            if hit.dist < best.dist - 1e-6:
                                best = hit

        if best:
            fixed[field] = best.value
            fixed["_meta"][field] = {
                "value": best.value,
                "dist": best.dist,
                "keyword": best.matched_keyword,
                "evidence": best.evidence,
            }

    return fixed


def retrieve_evidence_chunks(
    filename: str,
    currency_hint: Optional[str] = None,
    top_k: int = 6,
) -> List[str]:
    """
    Liefert wenige, aber sehr relevante Chunks als Evidence für das LLM.
    """
    queries = [
        f"Gesamtbetrag {currency_hint or ''}".strip(),
        f"Rechnungsbetrag {currency_hint or ''}".strip(),
        f"Gesamt Netto MwSt Versand {currency_hint or ''}".strip(),
        "Invoice total subtotal VAT tax shipping",
        "Rechnungsadresse Lieferadresse Absender Empfänger",
    ]

    chunks: List[str] = []
    seen = set()

    for q in queries:
        try:
            res = collection.query(query_texts=[q], n_results=top_k, where={"source": filename})
        except Exception:
            continue

        docs = (res.get("documents") or [[]])[0]
        for d in docs:
            dd = d.strip()
            if dd and dd not in seen:
                chunks.append(dd)
                seen.add(dd)

        if len(chunks) >= top_k:
            break

    return chunks[:top_k]
