# llm_extract.py
# Ziel: LLM soll NUR die Felder sauber extrahieren (v.a. Absender/Empfänger),
# ohne zu halluzinieren. Zusätzlich: deterministische Fallbacks + Party-Repair
# über Kandidaten aus `facts`, falls das LLM Müll liefert.

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from llm_client import ollama_chat
from utils import safe_float as _to_float_or_none


# ----------------------------
# JSON parsing helpers
# ----------------------------

def _extract_first_json_object(text: str) -> Optional[str]:
    """Findet das erste JSON-Objekt {...} in einem Text (robust gegen Vor-/Nachtext)."""
    if not text:
        return None

    s = text.strip()
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def _safe_json_load(s: str) -> Dict[str, Any]:
    """Versucht JSON zu laden; wenn das fehlschlägt, versucht es das erste {...} zu extrahieren."""
    s = (s or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    snippet = _extract_first_json_object(s)
    if not snippet:
        return {}

    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ----------------------------
# Evidence shrinking
# ----------------------------

def _shrink_evidence(
    evidence_chunks: List[str],
    max_total_chars: int = 6500,
    max_chunk_chars: int = 1400,
) -> List[str]:
    """
    Konservativ: Header/Fußzeile nicht zu hart kürzen (Parties / Nummer / Datum!).
    """
    out: List[str] = []
    total = 0
    for idx, c in enumerate(evidence_chunks or []):
        c = (c or "").strip()
        if not c:
            continue

        # 0..1: meistens Header/Fußzeile -> etwas mehr Platz geben
        limit = 2200 if idx in (0, 1) else max_chunk_chars
        c2 = c[:limit]

        if total + len(c2) > max_total_chars:
            break
        out.append(c2)
        total += len(c2)
    return out


# ----------------------------
# Normalization helpers
# ----------------------------


def _extract_invoice_number_from_text(blob: str) -> Optional[str]:
    if not blob:
        return None

    patterns = [
        r"(?:Rechnung\s*Nr\.?|Rechnung\s*No\.?|Rechnungsnr\.?|Rechnungsnummer|Invoice\s*(?:No\.?|Number))\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\-_/]+)",
        r"\bRechnung\s*Nr\.?\s*([A-Za-z0-9][A-Za-z0-9\-_/]+)\b",
    ]
    for p in patterns:
        m = re.search(p, blob, re.I)
        if m:
            val = m.group(1).strip()
            if len(val) >= 4 and val.lower() not in {"zahlungsziel", "rechnung", "invoice"}:
                return val
    return None


def _extract_invoice_date_from_text(blob: str) -> Optional[str]:
    """
    Minimaler Fallback: sucht nach 'Rechnungsdatum/Invoice date' und nimmt die nächste Datum-Entität.
    Gibt RAW zurück (Pipeline normalisiert ggf. später), aber meistens passt dd.mm.yyyy / yyyy-mm-dd.
    """
    if not blob:
        return None

    lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
    cue = re.compile(r"\b(Rechnungsdatum|Invoice\s*date)\b", re.I)
    due = re.compile(r"\b(Zahlungsziel|fällig|Faellig|Due\s*date|Payment\s*due)\b", re.I)
    date_re = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2}|[0-3]?\d[.\-/][0-1]?\d[.\-/]\d{2,4})\b")

    best: Optional[str] = None
    for i, ln in enumerate(lines[:120]):
        if due.search(ln):
            continue
        if cue.search(ln):
            m = date_re.search(ln)
            if m:
                return m.group(1)
            if i + 1 < len(lines):
                m2 = date_re.search(lines[i + 1])
                if m2 and not due.search(lines[i + 1]):
                    return m2.group(1)

    # fallback: irgendein Datum im oberen Bereich, aber keine Due-Date Zeilen
    for ln in lines[:120]:
        if due.search(ln):
            continue
        m = date_re.search(ln)
        if m:
            best = m.group(1)
            break

    return best


# ----------------------------
# Party cleaning + scoring
# ----------------------------

_AMOUNT_RE = re.compile(r"(?P<sign>[-+−])?\s*\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2})\s*(€|EUR|\$|USD)?", re.I)

_PARTY_FORBIDDEN = re.compile(
    r"\b("
    r"Rechnungsnummer|Invoice\s*(No|Number)|Rechnungsdatum|Invoice\s*date|"
    r"Zahlungsziel|fällig|Faellig|Due\s*date|Payment\s*due|"
    r"Zwischensumme|Gesamtbetrag|Rechnungsbetrag|Total\s*due|Amount\s*due|Grand\s*total|"
    r"Netto|Brutto|MwSt|USt|VAT|Summe|Total|"
    r"Beschreibung|Bezeichnung|Menge|Qty|Quantity|Einheit|Preis|Betrag|Position|Pos\.|"
    r"Zahlungsbeleg|Receipt|Payment\s*receipt"
    r")\b",
    re.I
)

_SELLER_CUES = re.compile(r"(USt-?Id|VAT|Tax\s*ID|Steuernummer|KVK|HRB|Handelsregister|IBAN|BIC|Bank|Reg\.)", re.I)
_BUYER_CUES = re.compile(r"(Rechnungsadresse|Bill\s*to|Invoice\s*to|Sold\s*to|Ship\s*to|Lieferadresse|Customer|Kunde|Rechnungsempfänger)", re.I)
_LEGAL_SUFFIX = re.compile(r"\b(GmbH|AG|UG|KG|GbR|Gbr|Ltd|LTD|Inc|INC|B\.V\.|BV|S\.à r\.l\.|Sarl)\b", re.I)

def _looks_addressish(line: str) -> bool:
    l = (line or "").strip()
    if len(l) < 3:
        return False
    if re.search(r"\b\d{5}\b", l):  # DE PLZ
        return True
    if re.search(r"\b(Deutschland|Germany|France|Niederlande|Netherlands|USA|United States|Vereinigte Staaten)\b", l, re.I):
        return True
    if re.search(r"\b(Straße|Str\.|Weg|Allee|Platz|Gasse|Street|St\.|Road|Rd\.|Avenue|Ave\.|Boulevard|Blvd\.|Lane|Ln\.)\b", l, re.I):
        return True
    if re.search(r"\b(www\.|https?://|\S+@\S+)\b", l, re.I):
        return True
    return False


def _clean_party_block(s: Optional[str], max_lines: int = 6) -> Optional[str]:
    """
    Macht aus einem Kandidaten einen sauberen Adressblock:
    - entfernt Tabellen-/Totals-/Meta-Zeilen
    - entfernt Betragszeilen
    - begrenzt auf max_lines
    """
    if not s or not isinstance(s, str):
        return None

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out: List[str] = []

    for ln in lines:
        if not ln:
            continue
        if len(ln) > 180:
            continue

        # harte Ausschlüsse
        if _PARTY_FORBIDDEN.search(ln):
            continue

        # Betragszeilen raus (außer die Zeile ist klar "Kontakt/Adresse")
        if _AMOUNT_RE.search(ln) and not _looks_addressish(ln):
            continue

        # "Rechnung" als Titelzeile raus
        if re.fullmatch(r"(Rechnung|Invoice|Zahlungsbeleg|Receipt)", ln, re.I):
            continue

        out.append(ln)
        if len(out) >= max_lines:
            break

    block = "\n".join(out).strip()
    return block if len(block) >= 10 else None


def _party_is_bad(block: Optional[str]) -> bool:
    if not block:
        return True
    b = block.strip()
    if len(b) < 10:
        return True
    # zu viel verbotene Tokens oder Beträge
    if _PARTY_FORBIDDEN.search(b):
        return True
    if _AMOUNT_RE.search(b):
        return True
    # muss wenigstens 2 Zeilen haben oder eine "starke" Zeile (Firma + Straße)
    lines = [ln for ln in b.splitlines() if ln.strip()]
    if len(lines) == 1 and not _looks_addressish(lines[0]) and not _LEGAL_SUFFIX.search(lines[0]):
        return True
    return False


def _score_party(block: Optional[str], role: str) -> float:
    """role: 'seller' oder 'buyer'."""
    if not block:
        return -1e9
    b = block.strip()
    if not b:
        return -1e9
    if _party_is_bad(b):
        return -1e8

    s = 0.0
    s += 3.0 * len(_SELLER_CUES.findall(b))
    s += 1.2 * len(_LEGAL_SUFFIX.findall(b))
    s += 0.6 * len(re.findall(r"\b(www\.|https?://|\S+@\S+)\b", b, re.I))
    s += 0.5 * sum(1 for ln in b.splitlines() if _looks_addressish(ln))

    if role == "seller":
        s -= 3.0 * len(_BUYER_CUES.findall(b))
    else:
        s += 3.0 * len(_BUYER_CUES.findall(b))
        s -= 1.5 * len(_SELLER_CUES.findall(b))

    # Bonus wenn Block "firma-like" beginnt
    first = b.splitlines()[0] if b.splitlines() else b
    if _LEGAL_SUFFIX.search(first) or re.search(r"\b(B\.V\.|GmbH|AG|Ltd|Inc|GbR)\b", first, re.I):
        s += 1.0

    return s


def _deep_get(obj: Any, path: List[str]) -> Any:
    cur = obj
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _collect_party_candidates_from_facts(facts: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Erwartete (optional) Felder, aber robust falls nicht vorhanden:
      facts.pipeline_context.base._party_candidates_seller
      facts.pipeline_context.base._party_candidates_buyer
      facts.pipeline_context.base._party_footer_seller
    Gibt (seller_candidates, buyer_candidates, footer_seller_candidates) zurück.
    """
    seller = _deep_get(facts, ["pipeline_context", "base", "_party_candidates_seller"])
    buyer = _deep_get(facts, ["pipeline_context", "base", "_party_candidates_buyer"])
    footer = _deep_get(facts, ["pipeline_context", "base", "_party_footer_seller"])

    def norm_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            out = []
            for it in x:
                if isinstance(it, str) and it.strip():
                    out.append(it.strip())
            return out
        return []

    return norm_list(seller), norm_list(buyer), norm_list(footer)


def _repair_parties_with_candidates(
    absender: Optional[str],
    empfaenger: Optional[str],
    facts: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Wenn LLM Parties schlecht/gleich/mischig sind:
    - nimm best-of Kandidaten (footer_seller ist stärkstes Signal)
    - verhindere absender==empfaenger
    """
    a = _clean_party_block(absender)
    e = _clean_party_block(empfaenger)

    seller_cands, buyer_cands, footer_seller_cands = _collect_party_candidates_from_facts(facts)

    candidates_seller = [_clean_party_block(x) for x in (footer_seller_cands + seller_cands + [a, e]) if _clean_party_block(x)]
    candidates_buyer = [_clean_party_block(x) for x in (buyer_cands + [e, a]) if _clean_party_block(x)]

    # best seller
    best_seller = None
    best_s_score = float("-inf")
    for c in candidates_seller:
        sc = _score_party(c, "seller")
        if sc > best_s_score:
            best_s_score = sc
            best_seller = c

    # best buyer (≠ seller)
    best_buyer = None
    best_b_score = float("-inf")
    for c in candidates_buyer:
        if best_seller and c.strip().lower() == best_seller.strip().lower():
            continue
        sc = _score_party(c, "buyer")
        if sc > best_b_score:
            best_b_score = sc
            best_buyer = c

    # Wenn LLM schon gut war, respektieren wir es
    if not _party_is_bad(a):
        best_seller = a
    if not _party_is_bad(e):
        best_buyer = e

    # Wenn Seller/Buyer weiterhin schlecht, setze None statt Müll
    if _party_is_bad(best_seller):
        best_seller = None
    if _party_is_bad(best_buyer):
        best_buyer = None

    # final safety: identisch -> buyer löschen
    if best_seller and best_buyer and best_seller.strip().lower() == best_buyer.strip().lower():
        best_buyer = None

    return best_seller, best_buyer


# ----------------------------
# Main LLM extraction
# ----------------------------

def llm_extract_invoice(
    evidence_chunks: List[str],
    facts: Dict[str, Any],
    currency_hint: Optional[str] = None,
    model: str = "qwen2.5:7b-instruct",
    debug: bool = False,
    timeout_s: int = 180,
    num_predict: int = 550,
    retries: int = 1,
) -> Dict[str, Any]:
    """
    LLM muss ausschließlich JSON liefern.
    Fokus: Parties (Absender/Empfänger) als saubere, getrennte Adressblöcke.
    """
    evidence_chunks = _shrink_evidence(evidence_chunks)

    schema = {
        "waehrung": "string|null",
        "rechnungsnummer": "string|null",
        "rechnungsdatum": "string|null",
        "absender": "string|null",
        "empfaenger": "string|null",
        "netto": "number|null",
        "steuer": "number|null",
        "versand": "number|null",
        "brutto": "number|null",
        "notes": "string|null",
    }

    system = (
        "Du bist ein extrem präziser Parser für Rechnungen.\n"
        "Du bekommst Textauszüge (evidence_chunks) und Facts als Kontext.\n"
        "WICHTIG: Du gibst AUSSCHLIESSLICH ein gültiges JSON-Objekt zurück.\n"
        "Kein Markdown. Kein Text davor oder danach."
    )

    # Regeln: kurz, hart, deutsch, mit Fokus auf Parties + Datum/Nummer + Steuerlogik
    rules: List[str] = [
        "Nutze NUR Informationen aus evidence_chunks (und Facts als Hinweis auf Kandidaten).",
        "Erfinde nichts. Wenn unsicher: null.",
        "Gib NUR diese Keys aus: waehrung, rechnungsnummer, rechnungsdatum, absender, empfaenger, netto, steuer, versand, brutto, notes.",
        "rechnungsnummer: EXAKT übernehmen (Keywords: Rechnung Nr, Rechnungsnr, Rechnungsnummer, Invoice No, Invoice Number).",
        "rechnungsdatum: NUR Rechnungsdatum/Invoice date. NICHT Zahlungsziel/Fälligkeitsdatum/Due date. NICHT Leistungszeitraum/Period.",
        "Beträge: netto/steuer/versand/brutto als Zahlen (Punkt oder Komma egal). Keine Währung im Zahlenfeld.",
        "Steuerregel: Wenn netto+steuer(+versand) vorhanden sind, sollen sie ungefähr brutto ergeben. Wenn nicht: notes kurz mit Problemhinweis füllen, aber NICHT raten.",
        "Parties: absender = Aussteller/Verkäufer; empfaenger = Kunde/Buyer.",
        "Parties: NUR Adressblock, max. 6 Zeilen. KEINE Rechnungsnummer, kein Datum, kein Zahlungsziel, keine Positionen, keine Totals, keine Beträge.",
        "NIEMALS absender und empfaenger mischen. Wenn du nicht sicher trennen kannst: setze das unsichere Feld auf null.",
        "Wenn in Facts Party-Kandidaten stehen (seller/buyer/footer_seller), nutze diese bevorzugt und wähle den saubersten Block.",
        "Fußzeile kann Seller enthalten (z.B. IBAN/VAT/HRB/Steuernummer). Das ist ein starkes Signal für absender.",
    ]

    user_payload = {
        "task": "Extrahiere Rechnungsfelder aus evidence_chunks. Ausgabe ist NUR JSON (kein Markdown).",
        "currency_hint": currency_hint,
        "schema": schema,
        "facts": facts,
        "evidence_chunks": evidence_chunks,
        "rules": rules,
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    out = ollama_chat(
        model=model,
        messages=messages,
        temperature=0.0,
        timeout_s=timeout_s,
        num_predict=num_predict,
        retries=retries,
    )

    content = (out.get("message") or {}).get("content", "") if isinstance(out, dict) else ""
    parsed = _safe_json_load(content)

    if debug:
        print("LLM raw:", content)
        print("LLM parsed:", parsed)

    # Nur die erlaubten Keys übernehmen
    clean: Dict[str, Any] = {}
    keys = ["waehrung", "rechnungsnummer", "rechnungsdatum", "absender", "empfaenger",
            "netto", "steuer", "versand", "brutto", "notes"]
    for k in keys:
        clean[k] = parsed.get(k, None) if isinstance(parsed, dict) else None

    # Numeric normalize
    for k in ["netto", "steuer", "versand", "brutto"]:
        clean[k] = _to_float_or_none(clean.get(k))

    # Strings trim
    for k in ["waehrung", "rechnungsnummer", "rechnungsdatum", "absender", "empfaenger", "notes"]:
        v = clean.get(k)
        if isinstance(v, str):
            v2 = v.strip()
            clean[k] = v2 if v2 else None

    # Currency fallback
    if not clean.get("waehrung") and currency_hint:
        clean["waehrung"] = currency_hint

    # Deterministischer Fallback: Rechnungsnummer / Datum aus Evidence, falls LLM null oder Müll
    blob = "\n".join(evidence_chunks)
    if not clean.get("rechnungsnummer"):
        inv = _extract_invoice_number_from_text(blob)
        if inv:
            clean["rechnungsnummer"] = inv

    if not clean.get("rechnungsdatum"):
        dt = _extract_invoice_date_from_text(blob)
        if dt:
            clean["rechnungsdatum"] = dt

    # Parties post-clean + Repair über Facts-Kandidaten (entscheidend!)
    clean["absender"] = _clean_party_block(clean.get("absender"))
    clean["empfaenger"] = _clean_party_block(clean.get("empfaenger"))

    a, e = _repair_parties_with_candidates(clean.get("absender"), clean.get("empfaenger"), facts or {})
    clean["absender"] = a
    clean["empfaenger"] = e

    # Final safety: Wenn Parties identisch -> empfaenger null
    if clean.get("absender") and clean.get("empfaenger"):
        if str(clean["absender"]).strip().lower() == str(clean["empfaenger"]).strip().lower():
            clean["empfaenger"] = None

    return clean
