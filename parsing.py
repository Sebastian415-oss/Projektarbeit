# parsing.py
# Deterministischer Invoice-Parser – kein LLM, nur Regex + Heuristiken.
# Public API: extract_invoice_data(text) -> Dict[str, Any]
# Wird von main.py als erster Schritt aufgerufen (vor Retrieval und LLM).
#
# Zwei wichtige Designentscheidungen:
#   Fix A: Datums-Strings (04.04.2025) erzeugen sonst false-positive Beträge (4.04)
#   Fix B: Versandkosten nur im Totals-Bereich mit klaren Labels – verhindert Positions-Konfusion
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple


# ----------------------------
# Regex / Basics
# ----------------------------

# Betrag mit optionalem Vorzeichen (inkl. Unicode-Minus "−") + optional Währung
# Beispiele: 1.234,56 | 1234,56 | 1 234,56 | -9,95 | −9,95 | (9,95)
AMOUNT_RE = re.compile(
    r"(?P<sign>[-+−])?\s*(?P<num>\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2}))\s*(?P<cur>€|EUR|\$|USD)?",
    re.I,
)

# Datum/Time (zum Herausfiltern bei Positionen & Kandidaten)
DATE_FULL_RE = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b")
DATE_RANGE_RE = re.compile(
    r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\s*[–-]\s*\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b"
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")

PAGE_MARKER_RE = re.compile(r"^---\s*Seite\s+\d+\s*---$", re.I)
PAGE_MARKER_RE2 = re.compile(r"^Seite\s+\d+(\s+von\s+\d+)?$", re.I)

# Monat (DE)
MONTHS_DE = {
    "januar": 1,
    "februar": 2,
    "märz": 3,
    "maerz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}


# ----------------------------
# Amount helpers
# ----------------------------

def _normalize_amount(raw: str) -> Optional[float]:
    """Wandelt Rohbetrag (z.B. '1.234,56 €', '-9,95', '(9,95)') in float um."""
    if not raw:
        return None

    s = raw.replace("\xa0", " ").strip()

    # Klammern = negativ
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # Unicode minus -> normal minus
    s = s.replace("−", "-")

    # Währung raus
    s = s.replace("€", "").replace("$", "").replace("EUR", "").replace("USD", "")
    s = s.replace(" ", "")

    # deutsches Komma -> Punkt
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-+]", "", s)

    # Entferne Tausenderpunkte: "1.234.56" -> "1234.56"
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        val = float(s)
        if neg:
            val = -abs(val)

        # Plausi
        if abs(val) > 10_000_000:
            return None
        return round(val, 2)
    except Exception:
        return None


def _detect_currency(text: str) -> Optional[str]:
    t = text.upper()
    if "€" in text or "EUR" in t:
        return "EUR"
    if "$" in text or "USD" in t:
        return "USD"
    return None


# ----------------------------
# Date extraction (ranking + normalization)
# ----------------------------

def _extract_numeric_date_from_text(s: str) -> Optional[Tuple[int, int, int]]:
    """
    Returns (yyyy, mm, dd) if parseable, else None.
    Assumes numeric formats:
      dd.mm.yyyy / dd-mm-yyyy / dd/mm/yyyy
      yyyy-mm-dd
    """
    s = s.strip()

    # yyyy-mm-dd
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return y, mo, d

    # dd.mm.yyyy / dd-mm-yyyy / dd/mm/yyyy
    m = re.search(r"\b([0-3]?\d)[.\-/]([0-1]?\d)[.\-/](\d{2,4})\b", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return y, mo, d

    return None


def _extract_monthname_date_from_text(s: str) -> Optional[Tuple[int, int, int]]:
    """
    4. März 2025 / 4. Maerz 2025 etc.
    """
    m = re.search(
        r"\b(\d{1,2})\.\s*(Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*(\d{4})\b",
        s,
        flags=re.I,
    )
    if not m:
        return None
    d = int(m.group(1))
    month_name = m.group(2).lower()
    month_name = month_name.replace("ä", "ae")
    y = int(m.group(3))
    mo = MONTHS_DE.get(month_name)
    if mo and 1 <= d <= 31:
        return y, mo, d
    return None


def _extract_any_date_raw(s: str) -> Optional[str]:
    """
    Extract first date-like string from s (raw, not normalized).
    """
    m = DATE_FULL_RE.search(s)
    if m:
        return m.group(0)
    # month-name
    m = re.search(
        r"\b\d{1,2}\.\s*(Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*\d{4}\b",
        s,
        flags=re.I,
    )
    if m:
        return m.group(0)
    # yyyy-mm-dd
    m = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", s)
    if m:
        return m.group(0)
    return None


def _normalize_date_to_iso(raw_date: str) -> str:
    """
    If parseable, returns YYYY-MM-DD, else returns raw_date unchanged.
    """
    if not raw_date:
        return raw_date

    tup = _extract_numeric_date_from_text(raw_date)
    if tup:
        y, mo, d = tup
        return f"{y:04d}-{mo:02d}-{d:02d}"

    tup = _extract_monthname_date_from_text(raw_date)
    if tup:
        y, mo, d = tup
        return f"{y:04d}-{mo:02d}-{d:02d}"

    return raw_date


def _extract_best_invoice_date(text: str) -> Optional[str]:
    """
    Picks the best invoice date by scoring line-level candidates:
    - Rechnungsdatum / Invoice date strongly preferred
    - Bezahlt am / Paid on okay (for receipts)
    - Generic Datum/Date weaker
    - Zahlungsziel / Fällig / Due date strongly penalized (avoid Lexoffice issue)
    """
    lines = _clean_lines(text)

    DUE_CUES = re.compile(r"\b(Zahlungsziel|fällig|Faellig|Due\s*date|Payment\s*due|Fälligkeit)\b", re.I)
    INV_DATE_CUES = re.compile(r"\b(Rechnungsdatum|Invoice\s*date)\b", re.I)
    PAID_CUES = re.compile(r"\b(bezahlt\s*am|paid\s*on)\b", re.I)
    GEN_DATE_CUES = re.compile(r"\b(Datum|Date)\b", re.I)
    PERIOD_CUES = re.compile(r"\b(Leistungszeitraum|Zeitraum|Billing\s*period|Period)\b", re.I)

    best_raw = None
    best_score = float("-inf")

    # 220 Zeilen reichen für fast alle PDFs; weiter unten stehen meist nur Zahlungsinfos
    for ln in lines[:220]:
        raw = _extract_any_date_raw(ln)
        if not raw:
            continue

        score = 0.0
        if INV_DATE_CUES.search(ln):
            score += 10.0
        elif PAID_CUES.search(ln):
            score += 6.0
        elif GEN_DATE_CUES.search(ln):
            score += 3.0
        else:
            score += 1.0

        # Starke Strafe: Zahlungsziel ist fast immer das Datum das fälschlicherweise
        # als Rechnungsdatum extrahiert wird (Lexoffice-Problem)
        if DUE_CUES.search(ln):
            score -= 12.0

        # Penalize "period"/range lines (often not invoice date)
        if PERIOD_CUES.search(ln) or DATE_RANGE_RE.search(ln):
            score -= 5.0

        # Small bonus if near invoice number keywords
        if re.search(r"\b(Rechnungsnummer|Invoice\s*(No|Number))\b", ln, re.I):
            score += 1.5

        if score > best_score:
            best_score = score
            best_raw = raw

    if best_raw:
        return _normalize_date_to_iso(best_raw)

    # fallback: first date in whole document
    raw_any = _extract_any_date_raw(text)
    return _normalize_date_to_iso(raw_any) if raw_any else None


# ----------------------------
# Text cleaning
# ----------------------------

def _clean_lines(text: str) -> List[str]:
    lines: List[str] = []
    for l in text.splitlines():
        l = l.replace("\x00", "").strip()
        if not l:
            continue
        # PDF Seitenmarker entfernen
        if PAGE_MARKER_RE.match(l) or PAGE_MARKER_RE2.match(l):
            continue
        # häufige Noise
        if l.lower() in {"page", "page 1", "page 2", "seite", "seite 1", "seite 2"}:
            continue
        # normalize dashes a bit
        l = l.replace("–", "-").replace("—", "-")
        lines.append(l)
    return lines


# ----------------------------
# Fix A: Date overlap checker
# ----------------------------

def _match_overlaps_any_date(ctx: str, start: int, end: int) -> bool:
    """
    Fix A:
    verhindert, dass aus Datums-Strings (z.B. 04.04.2025) ein Betrag 4.04 wird.
    Wir checken: liegt unser Amount-Match innerhalb oder überlappend eines DATE_FULL_RE Matches?
    """
    for dm in DATE_FULL_RE.finditer(ctx):
        ds, de = dm.start(), dm.end()
        if start < de and end > ds:
            return True
    return False


# ----------------------------
# Parties (Absender/Empfänger) heuristics
# ----------------------------

# ----------------------------
# Parties (Absender/Empfänger) – NEU (robuster + trennt Blöcke sauber)
# ----------------------------

# ----------------------------
# Parties (Absender/Empfänger) heuristics (NEU: Header+Footer Address-Blocks)
# ----------------------------

# ----------------------------
# Parties (Absender/Empfänger) – robust + universal (Header + Footer)
# ----------------------------

# ----------------------------
# Parties (Absender/Empfänger) – Footer-Seller erzwingen + Ähnlichkeitsvergleich
# ----------------------------

import re
from typing import List, Optional, Dict, Tuple, Any

# 1. HILFS-HEURISTIKEN
def _looks_like_addressish(line: str) -> bool:
    """Heuristik: sieht wie Adress-/Kontaktzeile aus."""
    if not line: return False
    l = line.strip()
    if len(l) < 3: return False
    
    # PLZ / Länder / Straßen / Rechtsformen
    patterns = [
        r"\b\d{5}\b", 
        r"\b(Deutschland|Germany|Niederlande|Netherlands|France|USA)\b",
        r"\b(Straße|Str\.|Weg|Allee|Platz|Gasse|Street|St\.|Road|Rd\.)\b",
        r"\b(GmbH|AG|UG|KG|GbR|Gbr|Ltd|B\.V\.|BV|S\.à r\.l\.)\b",
        r"\b(IBAN|BIC|USt-Id|VAT|Steuernummer|HRB|Amtsgericht|KVK)\b",
        r"\b(www\.|@)\b"
    ]
    return any(re.search(p, l, re.I) for p in patterns)

def _is_section_break(line: str) -> bool:
    """Stoppt die Adresssuche, wenn der Inhaltsteil der Rechnung beginnt."""
    if not line: return False
    l = line.strip()
    stop_words = r"(Artikel|Position|Pos\.|Leistung|Bezeichnung|Beschreibung|Menge|Qty|Quantity|Einheit|" \
                 r"Einzelpreis|Stückpreis|Preis|Betrag|Zwischensumme|Gesamtbetrag|Rechnungsbetrag|Total|Summe|Netto|Brutto|MwSt|USt|VAT)"
    return bool(re.search(stop_words, l, re.I)) or bool(re.match(r'^[_\-\.]{3,}$', l))

# 2. BEREINIGUNG & SCORING
def _clean_party_block(block: List[str]) -> List[str]:
    """Entfernt 'Rauschen' (Bankdaten, Rechnungs-Meta) aus dem Namensfeld."""
    if not block: return []
    cleaned = []
    noise = r"(Rechnungsnummer|Datum|Seite|Page|Kundennummer|Fällig|Zahlbar|IBAN|BIC|SWIFT|Bank|Konto|USt-ID|VAT-ID|Steuernummer|HRB|Amtsgericht)"
    for line in block:
        l = line.strip()
        # Behalte die Zeile nur, wenn sie kein reines Rauschen ist (außer es ist ein Firmenname/Straße dabei)
        if re.search(noise, l, re.I) and not re.search(r"\b(Str\.|Straße|Weg|Wokr|GmbH|GbR|B\.V\.)\b", l, re.I):
            continue
        if len(l) > 2:
            cleaned.append(l)
    return cleaned[:6] # Max 6 Zeilen für eine Adresse

def _score_party_block(block: List[str], role: str) -> float:
    """Bewertet, ob ein Block eher Absender (Seller) oder Empfänger (Buyer) ist."""
    text = "\n".join(block).lower()
    score = 0.0
    if role == "seller":
        if any(m in text for m in ["iban", "bic", "ust-id", "vat", "hrb", "www.", "@"]): score += 10.0
        if any(m in text for m in ["gmbh", "ag", "b.v.", "gbr"]): score += 2.0
    else: # buyer
        if any(m in text for m in ["rechnungsadresse", "bill to", "empfänger", "kunde"]): score += 12.0
        if any(m in text for m in ["iban", "bic", "hrb"]): score -= 5.0 # Käufer hat selten Bankdaten im Block
    return score

def _is_footer_seller_block(block_text: str) -> bool:
    """Checkt, ob ein Block die typischen Footer-Merkmale eines Sellers hat."""
    markers = ["iban", "bic", "ust-id", "vat", "steuernummer", "hrb", "amtsgericht", "kvk"]
    count = sum(1 for m in markers if m in block_text.lower())
    return count >= 2

# 3. BLOCK-SAMMLUNG & VERGLEICH
def _collect_address_blocks(lines: List[str]) -> List[List[str]]:
    """Sammelt Textzeilen zu logischen Adress-Blöcken."""
    blocks, current = [], []
    for line in lines:
        if _is_section_break(line):
            if current: blocks.append(current); current = []
            continue
        if _looks_like_addressish(line):
            current.append(line)
        elif current and len(current) < 5: # "Mitnahme-Logik" für Namen ohne Keyword
            current.append(line)
        elif current:
            blocks.append(current); current = []
    if current: blocks.append(current)
    return [b for b in blocks if len(b) >= 2]

def _blocks_similar(block_a: str, block_b: str) -> bool:
    """Vergleicht zwei Blöcke auf inhaltliche Gleichheit (Firmenname)."""
    if not block_a or not block_b: return False
    # Extrahiere signifikante Wörter (>= 4 Zeichen) aus den ersten zwei Zeilen
    def get_tokens(b): return set(re.findall(r'\w{4,}', " ".join(b.lower().splitlines()[:2])))
    toks_a, toks_b = get_tokens(block_a), get_tokens(block_b)
    if not toks_a or not toks_b: return False
    return len(toks_a.intersection(toks_b)) > 0

POSITION_NOISE = re.compile(
    r"(Pos\.|Bezeichnung|Menge|Einheit|Einzel|"
    r"Gesamt|Zwischensumme|Umsatzsteuer|Gesamtbetrag|"
    r"Stück|Implementierung|Erstellen)", re.I)


# 4. DIE HAUPTFUNKTION (Das "Gehirn")
def _guess_parties(text: str) -> Dict[str, Optional[str]]:
    all_lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not all_lines: return {"absender": None, "empfaenger": None}

    header_blocks = _collect_address_blocks(all_lines[:40])
    footer_blocks = _collect_address_blocks(all_lines[-25:])

    # 1. Absender (Seller) im Footer suchen
    absender_raw = None
    footer_cands = sorted([(b, _score_party_block(b, "seller") + (15.0 if _is_footer_seller_block("\n".join(b)) else 0)) 
                           for b in footer_blocks], key=lambda x: x[1], reverse=True)
    
    # Footer hat Vorrang als Absender-Quelle: echte Rechnungen haben dort fast immer
    # Impressum/IBAN/VAT des Ausstellers
    if footer_cands and footer_cands[0][1] > 5.0:
        absender_raw = footer_cands[0][0]
    else:
        # Fallback: Best Seller im Header
        header_seller = sorted([(b, _score_party_block(b, "seller")) for b in header_blocks], key=lambda x: x[1], reverse=True)
        if header_seller and header_seller[0][1] > 2.0:
            absender_raw = header_seller[0][0]

    # 2. Empfänger (Buyer) im Header suchen (darf nicht der Absender sein)
    empfaenger_raw = None
    potential_buyers = []
    for b in header_blocks:
        if POSITION_NOISE.search("\n".join(b)):
            continue
        if absender_raw and _blocks_similar("\n".join(b), "\n".join(absender_raw)):
            continue
        score = _score_party_block(b, "buyer")
        potential_buyers.append((b, score))
    
    potential_buyers.sort(key=lambda x: x[1], reverse=True)
    if potential_buyers:
        empfaenger_raw = potential_buyers[0][0]

    # 3. Finales Cleanup
    return {
        "absender": "\n".join(_clean_party_block(absender_raw)) if absender_raw else "Unbekannt",
        "empfaenger": "\n".join(_clean_party_block(empfaenger_raw)) if empfaenger_raw else "Unbekannt"
    }


# ----------------------------
# Robust invoice number
# ----------------------------

def _normalize_dashes(s: str) -> str:
    return (s or "").replace("–", "-").replace("—", "-").replace("-", "-").replace("\u2212", "-").replace("−", "-")


# Erkennt typische Rechnungsnummer-Muster ohne Keyword-Kontext
_INV_NUM_RE = re.compile(
    r"\b("
    r"RE\d{5,}"                       # RE250008, RE260010, RE250004
    r"|lx\d{8,}"                      # lx2025010084916
    r"|[A-Za-z]{2,5}\d{3,}-\d{3,}"   # BCE54405-0001
    r"|[A-Za-z0-9]{2,5}-\d{3,6}"     # 3OJ3-0002
    r")\b",
    re.I,
)


def _is_bad_invoice_number_token(token: str) -> bool:
    bad = {
        "zahlungsziel",
        "leistungszeitraum",
        "rechnungsdatum",
        "datum",
        "kundennummer",
        "invoice",
        "rechnungsnummer",
        "rechnungsnr",
        "rechnungs-nr",
    }
    t = (token or "").strip().lower()
    return t in bad or len(t) < 3


def extract_invoice_number(text: str) -> Optional[str]:
    text = _normalize_dashes(text)

    BLACKLIST = ["Kundennr", "Kundennummer", "Lieferdatum",
                 "Datum", "Zahlungsziel", "Bestellnummer"]
    _bl = {v.lower() for v in BLACKLIST}
    PREFERRED = re.compile(
        r"^(RE|INV|RG|REC|lx|BCE|3OJ)", re.I)

    def _score(tok: str) -> int:
        return 3 if PREFERRED.match(tok) else 0

    # Primary keyword regex
    m = re.search(
        r"(?:Rechnungs[-\s]?nr\.?|Rechnungsnummer|Invoice\s*No\.?|Invoice\s*Number)[:\s#]*([A-Za-z0-9][A-Za-z0-9\-_/]+)",
        text,
        flags=re.I,
    )
    if m:
        val = m.group(1).strip()
        if not _is_bad_invoice_number_token(val) and val.strip().lower() not in _bl:
            # Suffix join: BCE54405  - 0001
            tail = text[m.end(): m.end() + 24]
            m2 = re.search(r"^\s*-\s*(\d{3,})\b", tail)
            if m2 and "-" not in val:
                val = f"{val}-{m2.group(1)}"
            return val

    # Line-based fallback
    lines = _clean_lines(text)
    for i, line in enumerate(lines[:120]):
        if re.search(r"\b(Rechnungsnummer|Invoice\s*(No|Number))\b", line, flags=re.I):
            line_n = _normalize_dashes(line)
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]{3,}", line_n)
            tokens = [t for t in tokens if not _is_bad_invoice_number_token(t) and t.strip().lower() not in _bl]
            if tokens:
                return max(tokens, key=_score)

            # maybe in next line
            if i + 1 < len(lines):
                nxt = _normalize_dashes(lines[i + 1])
                tokens2 = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]{3,}", nxt)
                tokens2 = [t for t in tokens2 if not _is_bad_invoice_number_token(t) and t.strip().lower() not in _bl]
                if tokens2:
                    return max(tokens2, key=_score)

    # Pattern-based fallback: erkennt RE\d+, lx\d+, BCE\d+-\d+, 3OJ3-\d+ ohne Keyword
    for line in lines[:60]:
        m = _INV_NUM_RE.search(_normalize_dashes(line))
        if not m:
            continue
        val = m.group(1)
        if val.strip().lower() not in _bl and not _is_bad_invoice_number_token(val):
            return val

    return None


# ----------------------------
# Candidates-based amount extraction (Totals)
# ----------------------------

@dataclass
class AmountCandidate:
    field: str
    value: float
    raw: str
    line_idx: int
    label_pat: str
    label_weight: float
    has_currency: bool
    distance_chars: int
    evidence: str


def _iter_amount_matches(ctx: str):
    ctx = _normalize_dashes(ctx)
    for m in AMOUNT_RE.finditer(ctx):
        sign = m.group("sign") or ""
        num = m.group("num") or ""
        cur = m.group("cur")
        start, end = m.start(), m.end()

        # Fix A: Datumsüberlappung -> skip
        if _match_overlaps_any_date(ctx, start, end):
            continue

        raw = (sign + num).strip()
        yield raw, cur, start, end


def extract_amount_candidates_from_lines(
    lines: List[str],
    field: str,
    label_specs: List[Tuple[str, float]],
    start_idx: int,
    end_idx: int,
    window_lines: int = 2,
) -> List[AmountCandidate]:
    """
    Kandidaten NUR in einem Lines-Segment (Totals-Bereich).
    Fix A ist hier schon drin (Datum->Betrag wird ignoriert).
    """
    candidates: List[AmountCandidate] = []
    end_idx = min(end_idx, len(lines))

    for i in range(max(0, start_idx), end_idx):
        line = lines[i]
        for label_pat, w in label_specs:
            mlabel = re.search(label_pat, line, flags=re.I)
            if not mlabel:
                continue

            ctx_lines = lines[i : min(end_idx, i + 1 + window_lines)]
            ctx = "\n".join(ctx_lines)
            label_pos = mlabel.end()

            for raw, cur, start, end in _iter_amount_matches(ctx):
                # Prozentwerte raus (19,00%)
                after = ctx[end:end + 1]
                before = ctx[max(0, start - 1):start]
                if after == "%" or before == "%":
                    continue

                val = _normalize_amount((raw + (f" {cur}" if cur else "")).strip())
                if val is None:
                    continue

                dist = start - label_pos
                dist_penalty = 0
                if dist < 0:
                    dist_penalty = 9999
                    dist = abs(dist)

                candidates.append(
                    AmountCandidate(
                        field=field,
                        value=val,
                        raw=raw,
                        line_idx=i,
                        label_pat=label_pat,
                        label_weight=w,
                        has_currency=cur is not None,
                        distance_chars=max(0, dist) + dist_penalty,
                        evidence=ctx[:500],
                    )
                )

    return candidates


def pick_best_candidate(cands: List[AmountCandidate]) -> Optional[AmountCandidate]:
    if not cands:
        return None

    best: Optional[AmountCandidate] = None
    best_score = float("-inf")

    netto_vals = [c.value for c in cands if c.field == "netto"]
    best_netto_val = max(netto_vals) if netto_vals else None

    for c in cands:
        score = 0.0
        score += c.label_weight
        score += 2.0 if c.has_currency else 0.0

        if c.distance_chars <= 20:
            score += 2.0
        elif c.distance_chars <= 60:
            score += 1.0
        elif c.distance_chars > 180:
            score -= 1.0

        # riesige Werte eher selten (aber nicht unmöglich)
        if abs(c.value) > 200000:
            score -= 2.0

        # Negative Werte sind meist Rabatt/Abzug; fürs Totals-Feld meist unplausibel
        if c.field in {"brutto", "netto", "steuer", "versand"} and c.value < 0:
            score -= 3.0

        # Plausibilitätsprüfung Steuer
        if c.field == "steuer":
            if c.value > 50000:
                score -= 8.0
            if best_netto_val is not None and c.value > best_netto_val:
                score -= 10.0

        if score > best_score:
            best_score = score
            best = c

    return best


def _try_parse_netto_mwst_brutto_table(lines: List[str]) -> Dict[str, Optional[float]]:
    """
    Spezialfall (Finom-ähnlich):
    Header (evtl. über 2 Zeilen): 'Netto ... MwSt/VAT ... Brutto'
    Danach: Zeile mit netto / mwst / brutto
    """
    # Suche Header über 2-Zeilen-Fenster
    for i in range(min(len(lines), 200)):
        w = lines[i]
        w2 = (w + " " + lines[i + 1]) if i + 1 < len(lines) else w

        if (
            re.search(r"\bNetto\b", w2, flags=re.I)
            and re.search(r"\b(MwSt|VAT|USt)\b", w2, flags=re.I)
            and re.search(r"\bBrutto\b", w2, flags=re.I)
        ):
            for j in range(i + 1, min(len(lines), i + 6)):
                vals: List[float] = []
                for raw, cur, _, _ in _iter_amount_matches(lines[j]):
                    v = _normalize_amount((raw + (f" {cur}" if cur else "")).strip())
                    if v is not None:
                        vals.append(v)
                # oft genau 3 Werte (netto, steuer, brutto)
                if len(vals) >= 3:
                    return {"netto": vals[0], "steuer": vals[1], "brutto": vals[2]}
    return {"netto": None, "steuer": None, "brutto": None}


# ----------------------------
# Positionen parsing
# ----------------------------

# Erkennt nummerierte Positionszeilen: "1 Beschreibung 1 Stück 3.000,00 3.000,00"
POS_PATTERN = re.compile(
    r"^(\d{1,3})\s+"
    r"(.+?)\s+"
    r"(-?[\d.,]+)\s*"
    r"(Stück|Std|h|Monat|Pauschal|pcs|flat|month)?\s*"
    r"(-?[\d.,]+(?:\s*[$€£])?)\s+"
    r"(-?[\d.,]+(?:\s*[$€£])?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _find_items_block(lines: List[str]) -> Tuple[int, int]:
    """
    Findet den Bereich, der wie eine Positions-/Leistungstabelle aussieht.
    Erkennung über 2-Zeilen-Fenster (robuster gegen geteilte Header).
    Rückgabe: (start_idx, end_idx) als [start, end)

    FIX:
    - totals_stop NICHT mehr bei "Netto/MwSt/Brutto" stoppen, weil das bei vielen Layouts
      als Spaltenname oder sogar in Positionszeilen vorkommt (Lexoffice!).
    - Stattdessen nur bei "starken" Totals-Keywords stoppen.
    """
    header_pats = [
        r"\bPos\.?\b", r"\bPosition(en)?\b",
        r"\bBezeichnung\b", r"\bBeschreibung\b", r"\bLeistung\b", r"\bArtikel\b",
        r"\bMenge\b", r"\bQty\b", r"\bQuantity\b",
        r"\bEinheit\b", r"\bStück\b",
        r"\bEinzelpreis\b", r"\bStückpreis\b", r"\bPreis\b", r"\bUnit\b",
        r"\bBetrag\b", r"\bSumme\b", r"\bTotal\b",
    ]

    # ✅ Wichtig: Stop nur bei "echten" Totals-Zeilen, nicht bei "Netto/MwSt/Brutto" allgemein
    totals_stop = re.compile(
        r"^\s*(Zwischensumme|Gesamtbetrag|Rechnungsbetrag|Amount\s*due|Total\s*due|Grand\s*total|"
        r"Summe\s*gesamt|Gesamtsumme|Zu\s*zahlen|Payable)\b",
        re.I,
    )

    start = -1
    max_scan = min(len(lines), 260)

    for i in range(max_scan):
        w = lines[i]
        w2 = (w + " " + lines[i + 1]) if i + 1 < max_scan else w
        hits = sum(1 for p in header_pats if re.search(p, w2, re.I))
        if hits >= 2:
            start = i + 1
            break

    if start == -1:
        return -1, -1

    end = len(lines)
    for j in range(start, len(lines)):
        if totals_stop.search(lines[j]):
            end = j
            break

    # minimale Länge
    if end - start < 2:
        return -1, -1

    return start, end



def _line_has_money(line: str) -> bool:
    return bool(AMOUNT_RE.search(line))


def _looks_like_table_header(line: str) -> bool:
    """
    Header-Erkennung erweitern: Lexoffice & SaaS Tabellen haben manchmal andere Begriffe.
    """
    return bool(re.search(
        r"(Beschreibung|Bezeichnung|Leistung|Artikel|Menge|Qty|Quantity|Einheit|"
        r"\bEinzelpreis\b|\bStückpreis\b|\bPreis\b|Betrag|Summe|Total|Netto|Brutto|Steuer|MwSt|VAT|USt|"
        r"Position|Pos\.)",
        line,
        re.I
    ))


def _looks_like_date_line(line: str) -> bool:
    if DATE_RANGE_RE.search(line):
        return True
    if re.search(r"\bDatum\b", line, re.I) and DATE_FULL_RE.search(line):
        return True
    if DATE_FULL_RE.fullmatch(line.strip()):
        return True
    if DATE_FULL_RE.search(line) and TIME_RE.search(line) and ("-" in line or "–" in line):
        return True
    # month-name date lines
    if re.fullmatch(
        r"\s*\d{1,2}\.\s*(Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*\d{4}\s*",
        line,
        flags=re.I,
    ):
        return True
    return False


def _extract_amounts_from_line(line: str) -> List[float]:
    """
    Extrahiert Beträge aus einer Zeile, aber:
    - Filtert Datumsüberlappungen (Fix A)
    """
    vals: List[float] = []
    for m in AMOUNT_RE.finditer(_normalize_dashes(line)):
        sign = (m.group("sign") or "").replace("−", "-")
        num = m.group("num") or ""
        cur = m.group("cur")
        start, end = m.start(), m.end()

        if _match_overlaps_any_date(line, start, end):
            continue

        raw = (sign + num).strip()
        v = _normalize_amount((raw + (f" {cur}" if cur else "")).strip())
        if v is None:
            continue
        vals.append(v)
    return vals


def _strip_amounts_from_text(line: str) -> str:
    s = AMOUNT_RE.sub(" ", line)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _cleanup_item_description(desc: str) -> str:
    s = desc.strip()
    s = re.sub(r"^\s*\d+\s+", "", s)          # führende Positionsnummern
    s = re.sub(r"\s{2,}", " ", s).strip()
    if len(s) < 3:
        return ""
    return s


def parse_positions(text: str) -> Dict[str, Any]:
    """
    Extrahiert Positionen als Liste:
    [{"beschreibung": ..., "zeilensumme": ..., "raw_line": ...}, ...]
    plus positionen_anzahl & positionen_summe.

    Strategy 1: POS_PATTERN – nummerierte Positionen (Format A: Prozessia/WebWokr)
    Strategy 2: money-line fallback – Zeilen ohne führende Positionsnummer (Format B: Lexoffice, Format C: Instantly)

    Wenn _find_items_block keinen Header erkennt (start == -1), wird der gesamte
    Dokumentanfang (erste 200 Zeilen) durchsucht statt sofort leer zurückzugeben.
    """
    HEADER_NOISE = re.compile(
        r"(Einzel[\s\xa0]?€\s*|"
        r"Gesamt[\s\xa0]?€\s*|"
        r"^\s*\d+\s+(?=\w))",
        re.IGNORECASE,
    )

    lines = _clean_lines(text)

    start, end = _find_items_block(lines)
    if start == -1:
        # Kein erkennbarer Tabellen-Header (z.B. einfache Receipts ohne "Pos./Menge"-Spalte)
        # → gesamtes Dokument durchsuchen statt leer zurückzugeben
        start, end = 0, min(len(lines), 200)

    positionen: List[Dict[str, Any]] = []

    totals_like = re.compile(
        r"^\s*(Zwischensumme|Gesamtbetrag|Rechnungsbetrag|Amount\s*due|Total\s*due|Grand\s*total|"
        r"Summe\s*gesamt|Gesamtsumme|Zu\s*zahlen|Payable)\b",
        re.I,
    )

    # --- Strategy 1: POS_PATTERN (nummerierte Positionen) ---
    i = start
    while i < end:
        line = lines[i]

        if totals_like.search(line):
            break

        m = POS_PATTERN.match(line)
        if m:
            desc = m.group(2).strip()
            total_raw = m.group(6).strip()

            # Mehrzeilige Beschreibung: Folgezeilen ohne Positionsnummer und ohne Betrag
            j = i + 1
            while j < end:
                nxt = lines[j]
                if (totals_like.search(nxt)
                        or POS_PATTERN.match(nxt)
                        or _line_has_money(nxt)
                        or _looks_like_table_header(nxt)
                        or _looks_like_date_line(nxt)):
                    break
                desc = (desc + " " + nxt.strip()).strip()
                j += 1
            i = j

            total_norm = re.sub(r"[$€£]", "", total_raw).strip()
            total_val = _normalize_amount(total_norm)
            while HEADER_NOISE.search(desc):
                desc = HEADER_NOISE.sub("", desc).strip()
            desc = re.sub(
                r"\s+\d+\s*(Stück|Std|h|Monat|pcs|flat|month)\s*$",
                "", desc, flags=re.IGNORECASE).strip()
            if total_val is not None and len(desc) >= 2:
                positionen.append({
                    "beschreibung": desc,
                    "zeilensumme": float(total_val),
                    "raw_line": line,
                })
            continue

        i += 1

    # --- Strategy 2: money-line fallback (kein Positionsnummer-Präfix) ---
    if not positionen:
        pending_desc: Optional[str] = None
        pending_raw: Optional[str] = None

        # Problem 1 + 3: Zeilen die zur vorherigen Position gehören
        # ("pro Monat", "period: ...", "zzgl." etc.) nie als neue Position behandeln
        CONTINUATION_LINE = re.compile(
            r"^(pro\s+|per\s+|je\s+|ab\s+|bis\s+|"
            r"zzgl\.|inkl\.|incl\.|"
            r"period:?\s+|subscription\s+period|abrechnungszeitraum)",
            re.I,
        )

        for i in range(start, end):
            line = lines[i]

            if _looks_like_table_header(line):
                continue
            if _looks_like_date_line(line):
                continue
            if totals_like.search(line):
                break

            # Problem 1 + 3: Fortsetzungszeile → immer an pending_desc hängen,
            # nie als eigene Position – auch wenn sie Zahlen enthält ("pro Monat 19")
            if CONTINUATION_LINE.match(line.strip()):
                if pending_desc:
                    pending_desc = (pending_desc + " " + line.strip()).strip()
                    pending_raw = (pending_raw + " " + line.strip()).strip() if pending_raw else line.strip()
                continue

            if not _line_has_money(line):
                if pending_desc and len(line) <= 200:
                    pending_desc = (pending_desc + " " + line).strip()
                    pending_raw = (pending_raw + " " + line).strip() if pending_raw else line
                else:
                    if len(line) <= 160 and not totals_like.search(line):
                        pending_desc = line.strip()
                        pending_raw = line.strip()
                continue

            amounts = _extract_amounts_from_line(line)
            if not amounts:
                continue

            zeilensumme = amounts[-1]
            desc_raw = _strip_amounts_from_text(line)
            desc_raw = _cleanup_item_description(desc_raw)

            if pending_desc:
                desc = (pending_desc + " " + desc_raw).strip() if desc_raw else pending_desc
                raw_line = (pending_raw + " | " + line).strip() if pending_raw else line
                pending_desc = None
                pending_raw = None
            else:
                desc = desc_raw
                raw_line = line

            while HEADER_NOISE.search(desc):
                desc = HEADER_NOISE.sub("", desc).strip()
            desc = re.sub(
                r"\s+\d+\s*(Stück|Std|h|Monat|pcs|flat|month)\s*$",
                "", desc, flags=re.IGNORECASE).strip()
            # Problem 2: Trailing-Menge abschneiden ("Growth Plan 1" → "Growth Plan")
            desc = re.sub(r"\s+\d+\s*$", "", desc).strip()
            if not desc or len(desc) > 420:
                continue

            if re.search(r"\bDatum\b", desc, re.I) and DATE_FULL_RE.search(desc):
                continue

            positionen.append({
                "beschreibung": desc,
                "zeilensumme": float(zeilensumme),
                "raw_line": raw_line,
            })

    s = round(sum(float(p.get("zeilensumme") or 0.0) for p in positionen), 2)
    return {
        "positionen": positionen,
        "positionen_anzahl": len(positionen),
        "positionen_summe": s,
    }



# ----------------------------
# Totals / Validation helper
# ----------------------------

def _find_totals_region(lines: List[str], items_end: int) -> Tuple[int, int]:
    """
    Totals-Bereich:
    - Wenn items_end vorhanden: ab items_end bis Ende
    - sonst: letzte ~80 Zeilen
    """
    if items_end < 0:
        start = max(0, len(lines) - 80)
        return start, len(lines)

    start = min(items_end, len(lines))
    return start, len(lines)


def validate_invoice(fields: Dict[str, Any], tol: float = 0.2) -> bool:
    try:
        netto = fields.get("netto")
        steuer = fields.get("steuer")
        versand = fields.get("versand")
        brutto = fields.get("brutto")

        if brutto is None or netto is None:
            return False

        n = float(netto)
        s = 0.0 if steuer is None else float(steuer)
        v = 0.0 if versand is None else float(versand)
        b = float(brutto)

        return abs((n + s + v) - b) <= tol
    except Exception:
        return False


# ----------------------------
# Public API
# ----------------------------

def extract_invoice_data(text: str) -> Dict[str, Any]:
    """
    Ergebnis enthält:
    - waehrung, rechnungsnummer, rechnungsdatum, absender, empfaenger
    - positionen, positionen_anzahl, positionen_summe
    - netto, steuer, versand, brutto
    - validiert
    """
    lines = _clean_lines(text)

    data: Dict[str, Any] = {}
    data["waehrung"] = _detect_currency(text)

    # Rechnungsnummer
    data["rechnungsnummer"] = extract_invoice_number(text)

    # Rechnungsdatum (Ranking statt first-hit)
    data["rechnungsdatum"] = _extract_best_invoice_date(text)

    # Parties
    parties = _guess_parties(text)
    data["absender"] = parties.get("absender")
    data["empfaenger"] = parties.get("empfaenger")

    # Positionen
    pos = parse_positions(text)
    data.update(pos)

    # Items-Block Ende für Totals-Region bestimmen (damit Versand NICHT aus Positionen kommt)
    items_start, items_end = _find_items_block(lines)
    totals_start, totals_end = _find_totals_region(lines, items_end)

    # Spezialtable (Finom)
    table_vals = _try_parse_netto_mwst_brutto_table(lines)

    # Labels + Gewichte (Totals stark!)
    netto_labels = [
        (r"Gesamt\s*Netto", 6.0),
        (r"Zwischensumme\s*\(netto\)", 5.5),
        (r"Zwischensumme", 4.5),
        (r"Subtotal", 4.0),
        (r"\bNetto\b", 2.0),
        (r"\bNet\s*amount\b", 2.0),
    ]
    steuer_labels = [
        (r"\bMwSt\.?\b", 6.5),
        (r"Mehrwertsteuer", 6.5),
        (r"Umsatzsteuer", 6.5),
        (r"\bVAT\b", 6.0),
        (r"\bUSt\b", 5.5),
        (r"\bSteuer\b", 3.5),
        (r"\bTax\b", 3.0),
    ]

    # Fix B: Versand nur klare Versandkosten-Labels, vorzugsweise am Zeilenanfang (Totals-Bereich)
    versand_labels = [
        (r"^\s*Versandkosten\b", 6.5),
        (r"^\s*Versand\b", 4.5),
        (r"^\s*Lieferkosten\b", 4.8),
        (r"^\s*Porto\b", 4.2),
        (r"^\s*Shipping(\s+fee|\s+costs|\s+cost)?\b", 4.2),
        (r"^\s*Delivery(\s+fee|\s+cost)?\b", 4.0),
        (r"^\s*Handling(\s+fee|\s+cost)?\b", 3.5),
    ]

    brutto_labels = [
        (r"Gesamtbetrag", 7.5),
        (r"Rechnungsbetrag", 7.5),
        (r"Amount\s*due", 7.0),
        (r"Total\s*due", 7.0),
        (r"Grand\s*total", 7.0),
        (r"\bTotal\b", 6.5),
        (r"Summe\s*gesamt", 6.5),
        (r"\bBrutto\b", 4.5),
        (r"\bGesamt\b", 3.0),
    ]

    netto_cands = extract_amount_candidates_from_lines(lines, "netto", netto_labels, totals_start, totals_end, window_lines=2)
    steuer_cands = extract_amount_candidates_from_lines(lines, "steuer", steuer_labels, totals_start, totals_end, window_lines=1)
    versand_cands = extract_amount_candidates_from_lines(lines, "versand", versand_labels, totals_start, totals_end, window_lines=2)
    brutto_cands = extract_amount_candidates_from_lines(lines, "brutto", brutto_labels, totals_start, totals_end, window_lines=2)

    best_netto = pick_best_candidate(netto_cands)
    best_steuer = pick_best_candidate(steuer_cands)
    best_versand = pick_best_candidate(versand_cands)
    best_brutto = pick_best_candidate(brutto_cands)

    # Werte setzen (Table hat Priorität)
    data["netto"] = table_vals.get("netto") or (best_netto.value if best_netto else None)
    data["steuer"] = table_vals.get("steuer") or (best_steuer.value if best_steuer else None)
    data["versand"] = best_versand.value if best_versand else None
    data["brutto"] = table_vals.get("brutto") or (best_brutto.value if best_brutto else None)

    # Steuer-Plausibilitätskorrektur nach dem Setzen aller Werte
    if data.get("steuer") is not None and data.get("netto") is not None:
        if data["steuer"] > data["netto"]:
            data["steuer"] = None
    if data.get("steuer") is not None and data.get("brutto") is not None:
        if abs(data["steuer"] - data["brutto"]) < 0.01:
            data["steuer"] = None

    data["validiert"] = validate_invoice(data)
    return data
