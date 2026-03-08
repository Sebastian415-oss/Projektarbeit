# knowledge.py
# Bereitet strukturierte "Facts" pro Rechnung auf, die dem LLM als Kontext übergeben werden.
# Keine LLM-Aufrufe hier – nur Datenaufbereitung aus bereits geparstem Text.
# Wird von main.py aufgerufen, bevor llm_extract_invoice() gestartet wird.

from typing import Dict, Any, List, Optional
import re
from parsing import _normalize_amount

AMOUNT_LINE_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2}))\s*(€|EUR|\$|USD)?", re.I)


# Wenn genug beschriftete Beträge (hint != None) vorhanden sind, filtern wir
# alle unbeschrifteten raus – das reduziert Rauschen im LLM-Kontext deutlich.
def _filter_amount_facts(amount_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [f for f in amount_facts if f.get("hint")]
    if len(filtered) >= 5:
        return filtered[:30]
    return amount_facts[:20]


def build_invoice_facts(
    filename: str,
    full_text: str,
    base: Dict[str, Any],
    retrieval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Erstellt 'Facts' pro Rechnung:
    - lines (bereinigt)
    - amount_facts: alle Beträge + Zeile + Hinweislabel
    - party hints
    Diese Facts dienen als "Knowledge", das du später auch in Chroma speichern kannst.
    """
    lines = []
    for l in full_text.splitlines():
        l = l.replace("\x00", "").strip()
        if l:
            lines.append(l)

    amount_facts: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        matches = list(AMOUNT_LINE_RE.finditer(line))
        if not matches:
            continue

        hint = None
        if re.search(r"(Gesamtbetrag|Rechnungsbetrag|Total|Summe)", line, re.I):
            hint = "total"
        elif re.search(r"(Netto|Zwischensumme|Subtotal)", line, re.I):
            hint = "net"
        elif re.search(r"(MwSt|USt|VAT|Steuer|Umsatzsteuer)", line, re.I):
            hint = "tax"
        elif re.search(r"(Versand|Shipping|Porto|Lieferkosten)", line, re.I):
            hint = "shipping"

        for m in matches:
            raw = m.group(1)
            cur = m.group(2)
            val = _normalize_amount(raw + (f" {cur}" if cur else ""))
            if val is None:
                continue
            amount_facts.append({
                "line_index": i,
                "value": val,
                "currency": cur,
                "hint": hint,
                "line": line[:250],
            })

    facts = {
        "filename": filename,
        "currency": base.get("waehrung"),
        "invoice_no": base.get("rechnungsnummer"),
        "date": base.get("rechnungsdatum"),
        "parties": {
            "seller_hint": base.get("absender"),
            "buyer_hint": base.get("empfaenger"),
        },
        "amount_facts": _filter_amount_facts(amount_facts),
        "base_fields": {k: base.get(k) for k in ["netto", "steuer", "versand", "brutto"]},
        "retrieval_fields": {k: retrieval.get(k) for k in ["netto", "steuer", "versand", "brutto"]},
    }
    return facts
