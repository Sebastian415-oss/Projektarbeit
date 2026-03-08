# main.py
# Orchestriert die komplette Extraktions-Pipeline für alle PDFs in einem Ordner.
# Reihenfolge pro Rechnung: PDF-Text → Basis-Parser → Vektorstore → Retrieval →
#   Math-Hints → (optional) LLM Pass 1 → (optional) LLM Pass 2 → scoring → JSON/CSV.
# Außerdem: Party-Heuristiken, Layout-basierte Party-Extraktion, Priors-Lernen.
# Einstiegspunkt: run_pipeline() oder python main.py direkt.

import os
import json
import csv
import re
from typing import List, Dict, Any, Optional, Tuple

from config import OUTPUT_DIR, INVOICE_JSON_DIR, RESULTS_CSV
from pdf_io import extract_text_from_pdf, extract_layout_from_pdf
from parsing import extract_invoice_data
from embeddings_store import upsert_invoice_text
from retrieval import retrieve_fix_fields, retrieve_evidence_chunks
from knowledge import build_invoice_facts
from llm_extract import llm_extract_invoice
from scoring import merge_results
from priors import load_priors, learn_global_patterns, save_priors


# ----------------------------
# Helpers
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INVOICE_JSON_DIR, exist_ok=True)


def write_results_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = [
        "filename",
        "waehrung",
        "rechnungsnummer",
        "rechnungsdatum",
        "absender",
        "empfaenger",
        "netto",
        "steuer",
        "versand",
        "brutto",
        "validiert",
        "_source",
        "_error",
        "_valid_constraints",
        "_bonus",
    ]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n🧾 Ergebnisse nach {RESULTS_CSV} geschrieben.")


def _ensure_fixed_shape(d: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    d = d or {}
    return {
        "netto": d.get("netto"),
        "steuer": d.get("steuer"),
        "versand": d.get("versand"),
        "brutto": d.get("brutto"),
    }


def _float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip())
    except Exception:
        return None


def _get_prior(priors: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not priors:
        return None
    try:
        v = priors.get(key)
        return float(v) if v is not None else None
    except Exception:
        return None


def _compute_math_error(n: Optional[float], s: Optional[float], v: Optional[float], b: Optional[float]) -> Optional[float]:
    if b is None:
        return None
    vv = v or 0.0
    if n is None or s is None:
        return None
    return abs((n + s + vv) - b)


def _derive_math_hints(
    base: Dict[str, Any],
    retrieval: Dict[str, Optional[float]],
    priors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    # Versucht fehlende Beträge mathematisch herzuleiten, bevor das LLM bemüht wird.
    # Wenn z.B. netto und brutto bekannt sind, ist steuer = brutto - netto.
    # Das spart einen LLM-Aufruf bei vielen einfachen Rechnungen.
    tvat = _get_prior(priors, "typical_vat")

    # Start mit "best known" Inputs (base prior, dann retrieval)
    n = retrieval.get("netto") if retrieval.get("netto") is not None else base.get("netto")
    s = retrieval.get("steuer") if retrieval.get("steuer") is not None else base.get("steuer")
    b = retrieval.get("brutto") if retrieval.get("brutto") is not None else base.get("brutto")
    v = retrieval.get("versand") if retrieval.get("versand") is not None else base.get("versand")
    v = v or 0.0

    n = _float_or_none(n)
    s = _float_or_none(s)
    b = _float_or_none(b)
    v = _float_or_none(v) or 0.0

    out: Dict[str, Optional[float]] = {"netto": None, "steuer": None, "versand": None, "brutto": None}

    # brutto aus netto+steuer
    if b is None and n is not None and s is not None:
        out["brutto"] = round(n + s + v, 2)

    # steuer aus brutto-netto
    if s is None and b is not None and n is not None:
        cand = b - n - v
        if cand >= -0.01 and cand <= b + 0.01:
            out["steuer"] = round(max(0.0, cand), 2)

    # netto aus brutto-steuer
    if n is None and b is not None and s is not None:
        cand = b - s - v
        if cand >= 0.0:
            out["netto"] = round(cand, 2)

    # VAT-prior Verbesserung:
    # Wenn netto & brutto da und typical_vat vorhanden -> steuer = netto*tvat, wenn das brutto dadurch passt
    if tvat is not None and n is not None and b is not None:
        exp_s = round(n * tvat, 2)
        exp_b = round(n + exp_s + v, 2)
        if abs(exp_b - b) < 0.25:
            out["steuer"] = exp_s
            out["brutto"] = b  # bleibt

    # Versand hint nur, wenn Base/Retrieval nichts hat
    if (base.get("versand") is None and retrieval.get("versand") is None) and _get_prior(priors, "typical_shipping") is not None:
        # Shipping ist optional → wir setzen es NICHT hart, nur wenn es eindeutig in Dokumenten fehlt, lassen wir es None.
        out["versand"] = None

    return out


def _merge_retrieval_with_math(retrieval: Dict[str, Optional[float]], math_hint: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    # Priorität: echtes retrieval > math-hint
    return {
        "netto": retrieval.get("netto") if retrieval.get("netto") is not None else math_hint.get("netto"),
        "steuer": retrieval.get("steuer") if retrieval.get("steuer") is not None else math_hint.get("steuer"),
        "versand": retrieval.get("versand") if retrieval.get("versand") is not None else math_hint.get("versand"),
        "brutto": retrieval.get("brutto") if retrieval.get("brutto") is not None else math_hint.get("brutto"),
    }



def _apply_llm_meta_repairs(final: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
    # merge_results in scoring.py optimiert nur Zahlenfelder.
    # Textfelder (Parties, Datum, Nummer) müssen hier separat aus dem LLM übernommen werden.
    out = dict(final)
    for k in ["waehrung", "rechnungsnummer", "rechnungsdatum", "absender", "empfaenger"]:
        v = (llm or {}).get(k)
        if isinstance(v, str):
            v = v.strip()
            if v:
                out[k] = v
    return out

def _postprocess_parties_swap_if_needed(final: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sicherheitsnetz: Wenn Absender/Empfänger offensichtlich vertauscht sind,
    swap wir sie. (rein heuristisch)
    """
    out = dict(final)

    absender = str(out.get("absender") or "").strip()
    empfaenger = str(out.get("empfaenger") or "").strip()

    if not absender or not empfaenger:
        return out

    seller_cues = re.compile(r"(USt|USt-Id|VAT|KVK|HRB|Handelsregister|IBAN|BIC|Bank|Tax|Reg\.)", re.I)
    buyer_cues  = re.compile(r"(Rechnungsadresse|Bill\s*to|Invoice\s*to|Sold\s*to|Ship\s*to|Lieferadresse)", re.I)

    abs_seller = len(seller_cues.findall(absender))
    emp_seller = len(seller_cues.findall(empfaenger))

    abs_buyer = len(buyer_cues.findall(absender))
    emp_buyer = len(buyer_cues.findall(empfaenger))

    # Swap-Regel: Empfänger hat deutlich mehr seller-signale,
    # und Absender hat eher buyer-signale
    if (emp_seller - abs_seller) >= 2 and (abs_buyer - emp_buyer) >= 1:
        out["absender"], out["empfaenger"] = out.get("empfaenger"), out.get("absender")

    return out


import re
from typing import Any, Optional, List

def _clean_party_block(x: Any, max_lines: int = 10) -> Optional[str]:
    """
    Macht aus rohem Party-Text einen sauberen Block:
    - Page marker raus
    - Duplicate Lines raus
    - Tabellen-/Receipt-Noise raus (Beschreibung/Menge/Total/Bezahlt am etc.)
    - zu lange Zeilen raus
    """
    if not isinstance(x, str):
        return None

    t = x.replace("\x00", "").strip()
    if not t:
        return None

    # Page marker entfernen
    t = re.sub(r"^\s*---\s*Seite\s+\d+\s*---\s*$", "", t, flags=re.I | re.M).strip()

    # Zeilen normalisieren
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None

    # harte Noise-Zeilen filtern (typische Receipt-/Tabellenzeilen)
    noise_re = re.compile(
        r"(Zahlungsbeleg|Payment\s*receipt|Bezahlt\s*am|Paid\s*on|Zahlungsmethode|"
        r"Beschreibung|Menge|Qty|Quantity|Preis|Unit|Betrag|Zwischensumme|Gesamtbetrag|Rechnungsbetrag|"
        r"Total\s*due|Amount\s*due|Grand\s*total|Fällig|Faellig|Due\s*date|Payment\s*due|"
        r"Leistungszeitraum|Billing\s*period|period:)",
        re.I,
    )

    # Dedupe (case-insensitive)
    seen = set()
    cleaned: List[str] = []
    for ln in lines:
        if len(ln) > 180:
            continue
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)

        # einzelne reine Dokumenttyp-Zeilen weg
        if ln.strip().lower() in {"rechnung", "invoice", "zahlungsbeleg", "receipt"}:
            continue

        # Noise raus (aber nicht blind alles killen -> nur wenn Zeile sehr "tabellig" wirkt)
        if noise_re.search(ln) and not re.search(r"\b\d{5}\b", ln) and not re.search(r"\b(GmbH|GbR|B\.V\.|Ltd|Inc)\b", ln, re.I):
            continue

        cleaned.append(ln)
        if len(cleaned) >= max_lines:
            break

    out = "\n".join(cleaned).strip()
    if len(out) < 10:
        return None
    return out
from typing import List, Optional

def _split_header_into_party_candidates(header_lines: Optional[str]) -> List[str]:
    """
    Zerlegt header_lines in mehrere plausible Kandidaten-Blöcke,
    damit header nicht als Mischblock in die Party-Auswahl rutscht.
    """
    if not header_lines or not isinstance(header_lines, str):
        return []

    h = _clean_party_block(header_lines, max_lines=25)
    if not h:
        return []

    lines = [ln.strip() for ln in h.splitlines() if ln.strip()]
    if not lines:
        return []

    buyer_cues = re.compile(r"(Rechnungsadresse|Bill\s*to|Invoice\s*to|Sold\s*to|Ship\s*to|Lieferadresse|Customer|Kunde)", re.I)

    blocks: List[List[str]] = []
    cur: List[str] = []

    for ln in lines:
        # wenn buyer cue auftaucht -> neuen Block starten (typischer Wechsel im Header)
        if buyer_cues.search(ln) and cur:
            blocks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)

    if cur:
        blocks.append(cur)

    out: List[str] = []
    for b in blocks:
        bb = "\n".join(b).strip()
        bb = _clean_party_block(bb, max_lines=12)
        if bb:
            out.append(bb)

    return out


def _remove_overlap_lines(a: str, b: str) -> str:
    """
    Entfernt aus a alle Zeilen, die auch in b vorkommen (case-insensitive).
    Hilft gegen Mischblöcke: Seller enthält Buyer-Teil.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return a

    bset = set(ln.strip().lower() for ln in b.splitlines() if ln.strip())
    kept = [ln for ln in a.splitlines() if ln.strip().lower() not in bset]
    out = "\n".join(kept).strip()

    # falls wir dadurch zu viel zerstören: original behalten
    if len(out) < 10:
        return a
    return out


from typing import Literal

def _score_party_block(block: str, mode: Literal["seller", "buyer"]) -> float:
    """
    Scoring für Party-Blöcke.
    - seller: steuer/iban/kvk/ust-id/website/email + firma suffix = gut
    - buyer: PLZ + Land + (optional) dein eigener Name/Firma = gut
    - Receipt-/Tabellen-Noise = schlecht
    """
    if not isinstance(block, str):
        return -1e9

    b = block.strip()
    if not b or len(b) < 10:
        return -1e9

    seller_cues = re.compile(r"(USt-?Id|VAT|Tax\s*ID|Steuernummer|KVK|HRB|IBAN|BIC|Bank|Handelsregister|Reg\.)", re.I)
    buyer_cues  = re.compile(r"(Rechnungsadresse|Bill\s*to|Invoice\s*to|Sold\s*to|Ship\s*to|Lieferadresse|Customer|Kunde)", re.I)

    legal_suffix = re.compile(r"\b(GmbH|AG|UG|KG|GbR|Gbr|Ltd|Inc|B\.V\.|BV)\b", re.I)

    noise_re = re.compile(
        r"(Zahlungsbeleg|Payment\s*receipt|Bezahlt\s*am|Paid\s*on|Zahlungsmethode|"
        r"Beschreibung|Menge|Qty|Quantity|Preis|Unit|Betrag|Zwischensumme|Gesamtbetrag|Rechnungsbetrag|"
        r"Total\s*due|Amount\s*due|Grand\s*total|Fällig|Faellig|Due\s*date|Payment\s*due|"
        r"Leistungszeitraum|Billing\s*period|period:)",
        re.I,
    )

    score = 0.0

    sc_seller = len(seller_cues.findall(b))
    sc_buyer  = len(buyer_cues.findall(b))
    sc_legal  = len(legal_suffix.findall(b))
    sc_noise  = len(noise_re.findall(b))

    has_email_or_web = bool(re.search(r"\b(www\.|https?://|\S+@\S+)\b", b, re.I))
    has_plz = bool(re.search(r"\b\d{5}\b", b))
    has_country = bool(re.search(r"\b(Deutschland|Germany|Niederlande|Netherlands|Vereinigte Staaten|United States)\b", b, re.I))

    if mode == "seller":
        score += 4.5 * sc_seller
        score += 1.8 * sc_legal
        if has_email_or_web:
            score += 1.5
        score -= 3.5 * sc_buyer
        score -= 0.9 * sc_noise
        if has_plz:
            score += 0.4  # kann auch seller haben
    else:
        score += 4.0 * sc_buyer
        if has_plz:
            score += 2.0
        if has_country:
            score += 1.0
        # buyer sollte keine seller-IDs tragen
        score -= 3.8 * sc_seller
        score -= 0.5 * sc_noise
        # Websites/Emails sind eher Seller (außer bei echter Kundenmail, aber selten im Addressblock)
        if has_email_or_web:
            score -= 0.3

    # Extra: WebWokr als Buyer-Boost (passt zu deinen Daten, verhindert Instantly-Swap)
    if re.search(r"\bwebwokr\b", b, re.I):
        if mode == "buyer":
            score += 2.2
        else:
            score -= 1.2

    # "Instantly" als Seller-Boost (Receipt-typisch)
    if re.search(r"\binstantly\b", b, re.I):
        if mode == "seller":
            score += 1.4
        else:
            score -= 0.8

    return score



from typing import Dict, Any, Optional, List, Tuple

def _llm_party_is_plausible(block: Any) -> bool:
    if not isinstance(block, str):
        return False
    b = block.strip()
    if len(b) < 10:
        return False
    noise = re.compile(r"(Kundennr|Lieferdatum|Rechnungsnr|Zahlungsziel|Bezahlt am)", re.I)
    if noise.search(b):
        return False
    return True


def fix_parties_best_of(
    base: Dict[str, Any],
    llm_out: Dict[str, Any],
    final: Dict[str, Any],
    header_lines: Optional[str] = None,
) -> Dict[str, Any]:
    # Sammelt alle Party-Kandidaten aus Base/LLM/Header und wählt per Scoring
    # den besten Seller und den besten (vom Seller verschiedenen) Buyer.
    # Header wird bewusst gesplittet bevor er als Kandidat rein geht –
    # sonst landet oft der komplette Adress-Header als ein einziger Mischblock.
    out = dict(final)

    # LLM-Parties direkt übernehmen wenn beide plausibel und nicht identisch
    llm_abs = llm_out.get("absender") if isinstance(llm_out, dict) else None
    llm_emp = llm_out.get("empfaenger") if isinstance(llm_out, dict) else None
    if (
        _llm_party_is_plausible(llm_abs)
        and _llm_party_is_plausible(llm_emp)
        and llm_abs.strip().lower() != llm_emp.strip().lower()
    ):
        out["absender"] = llm_abs
        out["empfaenger"] = llm_emp
        return out

    candidates_raw: List[Any] = []
    for src in (final, llm_out, base):
        if isinstance(src, dict):
            candidates_raw.append(src.get("absender"))
            candidates_raw.append(src.get("empfaenger"))

    # Header: NICHT als 1 Block hinzufügen -> splitten!
    header_candidates = _split_header_into_party_candidates(header_lines)

    # Clean + dedup
    candidates: List[str] = []
    seen = set()

    for c in candidates_raw:
        cc = _clean_party_block(c)
        if cc:
            key = cc.strip().lower()
            if key not in seen:
                seen.add(key)
                candidates.append(cc)

    for hc in header_candidates:
        key = hc.strip().lower()
        if key not in seen:
            seen.add(key)
            candidates.append(hc)

    if not candidates:
        return out

    # Seller best
    scored_seller: List[Tuple[str, float]] = sorted(
        [(c, _score_party_block(c, "seller")) for c in candidates],
        key=lambda x: x[1],
        reverse=True,
    )
    best_seller = scored_seller[0][0] if scored_seller else None

    # Buyer best (muss != seller sein)
    scored_buyer: List[Tuple[str, float]] = sorted(
        [(c, _score_party_block(c, "buyer")) for c in candidates],
        key=lambda x: x[1],
        reverse=True,
    )
    best_buyer = None
    for c, _sc in scored_buyer:
        if best_seller and c.strip().lower() == best_seller.strip().lower():
            continue
        best_buyer = c
        break

    # Fallbacks
    if not best_seller:
        best_seller = _clean_party_block(base.get("absender")) or _clean_party_block(final.get("absender"))
    if not best_buyer:
        best_buyer = _clean_party_block(base.get("empfaenger")) or _clean_party_block(final.get("empfaenger"))

    if best_seller:
        out["absender"] = best_seller
    if best_buyer:
        out["empfaenger"] = best_buyer

    # Mischblock cleanup: seller enthält buyer -> raus
    a = out.get("absender")
    e = out.get("empfaenger")
    if isinstance(a, str) and isinstance(e, str) and a.strip() and e.strip():
        out["absender"] = _remove_overlap_lines(a, e)
        out["empfaenger"] = _remove_overlap_lines(e, out["absender"])

    # Final safety: wenn identisch -> empfaenger leeren (oder fallback auf base empfaenger)
    a2 = str(out.get("absender") or "").strip()
    e2 = str(out.get("empfaenger") or "").strip()
    if a2 and e2 and a2.lower() == e2.lower():
        fallback_buyer = _clean_party_block(base.get("empfaenger"))
        out["empfaenger"] = fallback_buyer if fallback_buyer and fallback_buyer.lower() != a2.lower() else None

    return out




# Entscheidet ob ein LLM-Aufruf lohnt. LLM ist teuer (Zeit), also nur wenn:
# Zahlen nicht validieren, Parties kaputt sind, Rechnungsnummer fehlt etc.
def should_call_llm(base: Dict[str, Any], retrieval: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    seller_cues = re.compile(r"(USt|USt-Id|VAT|Tax\s*ID|Steuernummer|KVK|HRB|IBAN|BIC|Bank|Handelsregister|Reg\.)", re.I)
    buyer_cues  = re.compile(r"(Rechnungsadresse|Bill\s*to|Invoice\s*to|Sold\s*to|Ship\s*to|Lieferadresse|Customer|Kunde|Rechnungsempfänger)", re.I)
    due_cues    = re.compile(r"(Zahlungsziel|fällig|Faellig|Due\s*date|Payment\s*due|Fälligkeit)", re.I)

    # NEW: Meta-Müll in Party-Blöcken (sehr häufig bei Receipt/Layouts)
    meta_noise = re.compile(
        r"(Rechnungsnr|Rechnungsnummer|Invoice\s*(No|Number)|Kundennr|Customer\s*ID|"
        r"Bezahlt\s*am|Paid\s*on|Zahlungsmethode|Payment\s*method|Visa|Mastercard|"
        r"Fällig|Zahlungsziel|Due\s*date|Payment\s*due)",
        re.I,
    )

    def looks_like_noise_party(s: Any) -> bool:
        if not isinstance(s, str):
            return True
        t = s.strip()
        if not t:
            return True
        tl = t.lower()
        if tl in {"zahlungsbeleg", "invoice", "rechnung", "receipt", "payment receipt"}:
            return True
        if len(tl) < 10:
            return True
        if "seite" in tl and "---" in tl:
            return True
        return False

    def party_contains_meta_noise(s: Any) -> bool:
        return isinstance(s, str) and bool(meta_noise.search(s))

    def parties_too_similar(a: Any, b: Any) -> bool:
        """Lexoffice-Killer + Mischblock-Detektor."""
        if not isinstance(a, str) or not isinstance(b, str):
            return False
        aa = a.strip()
        bb = b.strip()
        if not aa or not bb:
            return True
        if aa == bb:
            return True
        # containment in beide Richtungen (große Blöcke)
        if len(aa) > 40 and len(bb) > 40 and (aa in bb or bb in aa):
            return True
        return False

    def is_mixed_party_block(s: Any) -> bool:
        if not isinstance(s, str):
            return False
        t = s.strip()
        if not t:
            return False
        sc = len(seller_cues.findall(t))
        bc = len(buyer_cues.findall(t))
        if sc >= 1 and bc >= 1:
            return True
        # viele Seller-Cues + sieht nach kompletter Adresse aus + mail/url -> oft Mischblock
        if sc >= 2 and (("\n" in t and t.count("\n") >= 4) or re.search(r"\b\d{5}\b", t)):
            if re.search(r"\b(www\.|https?://|\S+@\S+)\b", t, re.I):
                return True
        return False

    def date_is_suspicious(d: Any) -> bool:
        if not isinstance(d, str):
            return True
        t = d.strip()
        if not t:
            return True
        if due_cues.search(t):
            return True
        return False

    # ------------------------
    # 1) Wenn numerisch validiert: Meta kann trotzdem kaputt sein → LLM für Meta-Repair
    # ------------------------
    if candidate.get("validiert") is True and candidate.get("_valid_constraints", True) is True:
        if not candidate.get("rechnungsdatum") or date_is_suspicious(candidate.get("rechnungsdatum")):
            return True
        if not candidate.get("rechnungsnummer"):
            return True

        absender = candidate.get("absender")
        empfaenger = candidate.get("empfaenger")

        # missing/noisy party
        if looks_like_noise_party(absender) or looks_like_noise_party(empfaenger):
            return True

        # NEW: Parties identisch/zu ähnlich -> Lexoffice, Mischblöcke
        if parties_too_similar(absender, empfaenger):
            return True

        # NEW: Party enthält Meta-Müll -> Instantly/Schaufler/WebWokr
        if party_contains_meta_noise(absender) or party_contains_meta_noise(empfaenger):
            return True

        # Mischblock-Trigger
        if is_mixed_party_block(absender) or is_mixed_party_block(empfaenger):
            return True

        # Optional: Positionen fehlen komplett, aber Rechnung wirkt klassisch (keine Receipt)
        if (candidate.get("positionen_anzahl") in (None, 0)) and (base.get("positionen_anzahl") in (None, 0)):
            if (candidate.get("waehrung") == "EUR") and (candidate.get("steuer") is not None):
                return True

        # Zahlen + Meta gut → kein LLM
        if candidate.get("netto") is not None and candidate.get("brutto") is not None:
            return False

    # ------------------------
    # 2) harte Gründe
    # ------------------------
    if candidate.get("_valid_constraints") is False:
        return True
    if candidate.get("validiert") is False:
        return True

    if candidate.get("netto") is None or candidate.get("brutto") is None:
        return True

    if base.get("validiert") is False:
        return True

    # Klassiker: steuer == netto (häufiger Parse-Bug)
    b_netto = base.get("netto")
    b_steuer = base.get("steuer")
    if b_netto is not None and b_steuer is not None:
        try:
            if abs(float(b_netto) - float(b_steuer)) < 0.01:
                return True
        except Exception:
            pass

    r_any = any(retrieval.get(k) is not None for k in ("netto", "steuer", "brutto", "versand"))
    if r_any and candidate.get("validiert") is not True:
        return True

    return False






# Pass 2 wird nur gestartet wenn Pass 1 die Situation nicht ausreichend verbessert hat.
def _needs_second_pass(final: Dict[str, Any]) -> bool:
    if final.get("_valid_constraints") is False:
        return True
    if final.get("validiert") is not True:
        return True
    if final.get("netto") is None or final.get("brutto") is None:
        return True

    # Meta-Killer: fehlende Rechnungsnummer
    if not final.get("rechnungsnummer"):
        return True

    absender = final.get("absender")
    empfaenger = final.get("empfaenger")

    # Parties identisch/zu ähnlich -> Lexoffice + Mischblock
    def too_similar(a: Any, b: Any) -> bool:
        if not isinstance(a, str) or not isinstance(b, str):
            return False
        aa = a.strip()
        bb = b.strip()
        if not aa or not bb:
            return True
        if aa == bb:
            return True
        if len(aa) > 40 and len(bb) > 40 and (aa in bb or bb in aa):
            return True
        return False

    meta_noise = re.compile(
        r"(Rechnungsnr|Rechnungsnummer|Invoice\s*(No|Number)|Kundennr|Customer\s*ID|"
        r"Bezahlt\s*am|Paid\s*on|Zahlungsmethode|Payment\s*method|Visa|Mastercard|"
        r"Fällig|Zahlungsziel|Due\s*date|Payment\s*due)",
        re.I,
    )

    def has_meta_noise(s: Any) -> bool:
        return isinstance(s, str) and bool(meta_noise.search(s))

    if too_similar(absender, empfaenger):
        return True
    if has_meta_noise(absender) or has_meta_noise(empfaenger):
        return True

    return False



def _safe_merge_results(
    base: Dict[str, Any],
    retrieval: Dict[str, Any],
    llm: Dict[str, Any],
    priors: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        return merge_results(base=base, retrieval=retrieval, llm=llm, priors=priors, debug=debug)  # type: ignore
    except TypeError:
        return merge_results(base=base, retrieval=retrieval, llm=llm, debug=debug)


# ----------------------------
# Layout-based party extraction
# ----------------------------

def _extract_parties_from_layout(layout: dict) -> dict:
    seller_cues = re.compile(
        r"(IBAN|BIC|USt|Steuernummer|HRB|VAT|KVK|@|www\.)", re.I)
    noise_cues = re.compile(
        r"(Kundennr|Rechnungsnr|Zahlungsziel|Lieferdatum|"
        r"Zwischensumme|Gesamtbetrag|Pos\.|Bezeichnung|"
        r"Menge|Einheit|Betrag)", re.I)
    legal_cues = re.compile(
        r"(GmbH|AG|UG|KG|GbR|Ltd|Inc|B\.V\.|\d{5})", re.I)

    # Dient als Empfänger-Label UND als Cutoff-Grenze im Absender-Block
    cutoff = re.compile(
        r"(Rechnungsempf.nger|Bill\s*to|"
        r"Rechnungsadresse|Ship\s*to|"
        r"Zahlungsempf.nger)", re.I)

    # Filtert Metadaten-Zeilen aus dem Empfänger-Block (WebWokr-Problem)
    EMPF_NOISE = re.compile(
        r"^(\d{3,6}$|"
        r"\d{2}\.\d{2}\.\d{4}$|"
        r"Anzahlungsrechnung|"
        r"Rechnung$|"
        r"Lieferdatum|"
        r"Kundennr|"
        r"RE\d{6}$)",
        re.I)

    absender = None
    empfaenger = None

    # Fix 2: Label-Erkennung für empfaenger – nutzt cutoff statt der alten
    # empfaenger_label, erfasst damit auch "Rechnungsadresse" und "Zahlungsempfänger"
    for block in (layout.get("header_blocks", []) +
                  layout.get("left_blocks", []) +
                  layout.get("right_blocks", [])):
        if cutoff.search(block):
            lines = block.split("\n")
            for i, line in enumerate(lines):
                if cutoff.search(line):
                    empfaenger = "\n".join(
                        lines[i+1:i+5]).strip()
                    break
            if empfaenger:
                break

    for block in layout.get("footer_blocks", []):
        if seller_cues.search(block) and len(block) > 20:
            if not noise_cues.search(block):
                absender = block[:300]
                break

    if not absender:
        for block in layout.get("right_blocks", []):
            if seller_cues.search(block) and len(block) > 20:
                absender = block[:300]
                break

    if not empfaenger:
        for block in layout.get("left_blocks", []):
            if legal_cues.search(block) and len(block) > 10:
                if not seller_cues.search(block):
                    if not noise_cues.search(block):
                        empfaenger = block[:300]
                        break

    # Fix 1: Cutoff-Bereinigung – entfernt Empfänger-Abschnitte die fälschlicherweise
    # im Absender-Block landen (Instantly-Problem: "Rechnungsempfänger\nSebastian Spuhler")
    result = {"absender": absender, "empfaenger": empfaenger}
    for field in ["absender", "empfaenger"]:
        val = result.get(field, "")
        if val:
            m = cutoff.search(val)
            if m:
                result[field] = val[:m.start()].strip()

    # EMPF_NOISE-Bereinigung – entfernt Metadaten-Zeilen aus empfaenger
    # (WebWokr-Problem: Kundennummern, Datumsangaben, "Anzahlungsrechnung", RE\d{6} etc.)
    if result.get("empfaenger"):
        lines = result["empfaenger"].split("\n")
        clean = [l for l in lines
                 if l.strip()
                 and not EMPF_NOISE.match(l.strip())]
        if clean:
            result["empfaenger"] = "\n".join(clean).strip()

    return result


# ----------------------------
# LLM Party / Position Fix
# ----------------------------

def _llm_fix_parties_and_positions(
    text: str,
    base: dict,
    model: str = None,
    timeout_s: int = 45,
) -> dict:
    from llm_client import ollama_chat
    from config import OLLAMA_MODEL

    if model is None:
        model = OLLAMA_MODEL

    system = """Du bist ein präziser Rechnungs-Parser.
Extrahiere aus dem Rechnungstext die fehlenden Felder.
Antworte NUR mit validem JSON, kein Text davor/danach."""

    user = f"""Rechnungstext (Auszug):
{text[:3000]}

Bekannte Werte:
- absender: {base.get('absender', 'unbekannt')}
- empfaenger: {base.get('empfaenger', 'unbekannt')}
- positionen: {base.get('positionen', [])}

Aufgabe: Korrigiere absender und empfaenger falls \
sie offensichtlich falsch sind (Metadaten, Tabellen,
Nummern statt Firmennamen).
Extrahiere positionen falls die Liste leer ist.

Regeln:
- absender = die ausstellende Firma mit Adresse
- empfaenger = die empfangende Firma/Person mit Adresse
- positionen = Liste von Leistungen mit Beträgen
- Gib NUR Felder zurück die du korrigierst/ergänzt

Wichtige Regeln für positionen:
- Steuersätze (19, 7, 0) sind KEINE Positionen
- 'pro Monat', 'per month' gehören zur vorherigen Position als Beschreibungszusatz
- Jede Position muss eine sinnvolle Beschreibung haben (min. 3 Wörter)
- Kombiniere Zeilen die zusammen eine Position beschreiben (z.B. Leistungsname + Zeitraum + Betrag)

Antworte mit JSON:
{{
  "absender": "...",
  "empfaenger": "...",
  "positionen": [
    {{
      "beschreibung": "...",
      "menge": 1,
      "einheit": "Stück",
      "einzelpreis": 0.0,
      "zeilensumme": 0.0
    }}
  ]
}}"""

    try:
        result = ollama_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
            num_predict=600,
            timeout_s=timeout_s,
            temperature=0.0,
        )
        content = result["message"]["content"]
        import json as _json
        json_match = re.search(r'\{.*\}', content, re.S)
        if json_match:
            data = _json.loads(json_match.group())
            return data
    except Exception as e:
        print(f"⚠️ LLM Party/Position Fix Fehler: {e}")
    return {}


# ----------------------------
# Main pipeline
# ----------------------------

def run_pipeline(
    pdf_folder: str = "pdfs",
    use_llm: bool = True,
    debug: bool = False,
    retrieval_top_k: int = 4,
    llm_evidence_top_k: int = 6,
) -> None:
    ensure_dirs()

    priors_global = load_priors()
    if priors_global:
        print(f"📥 Geladene globale Muster (priors.json): {priors_global}")

    if not os.path.isdir(pdf_folder):
        print(f"⚠️ Ordner nicht gefunden: {pdf_folder}")
        return

    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("⚠️ Keine PDFs im Ordner gefunden.")
        return

    finals: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    total = len(pdf_files)

    # -------------------------
    # Lokale Helpers
    # -------------------------
    def _party_bad(a: Any, b: Any) -> bool:
        aa = str(a or "").strip()
        bb = str(b or "").strip()

        if not aa or not bb:
            return True
        if aa.lower() == bb.lower():
            return True
        if len(aa) > 40 and len(bb) > 40 and (aa in bb or bb in aa):
            return True

        bad_cues = r"(Zahlungsbeleg|Bezahlt\s*am|Zahlungsmethode|Kundennr|Rechnungsnr|Fällig|Zahlungsziel|Amount\s*due|Total\s*due|Zu\s*zahlen)"
        if re.search(bad_cues, aa, re.I) or re.search(bad_cues, bb, re.I):
            return True

        # Wenn einer der Blöcke eher "Positions-/Totals-Text" ist
        table_cues = r"(Beschreibung|Menge|Qty|Einheit|Preis|Betrag|Zwischensumme|Gesamtbetrag|Rechnungsbetrag|Grand\s*total|Summe\s*gesamt|Gesamtsumme)"
        if re.search(table_cues, aa, re.I) or re.search(table_cues, bb, re.I):
            return True

        return False

    def _dedup_chunks(chunks: List[str], limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for c in chunks or []:
            cc = (c or "").strip()
            if not cc:
                continue
            if cc in seen:
                continue
            seen.add(cc)
            out.append(cc)
            if len(out) >= limit:
                break
        return out

    def _prepend_if_present(evidence: List[str], chunk: str) -> List[str]:
        ch = (chunk or "").strip()
        if not ch:
            return evidence
        return [ch] + (evidence or [])

    # -------------------------
    # Main loop
    # -------------------------
    for idx, pdf_file in enumerate(pdf_files, start=1):
        file_path = os.path.join(pdf_folder, pdf_file)
        print(f"\n[{idx}/{total}] 📄 Verarbeite Datei: {pdf_file}\n")

        # 1) PDF -> Text
        try:
            text = extract_text_from_pdf(file_path)
        except Exception as e:
            print(f"❌ Fehler beim PDF-Text-Extract: {e}")
            continue

        if not text.strip():
            print("❌ Kein Text erkannt (möglicherweise Scan).")
            continue

        if debug:
            print("📑 Textauszug:\n")
            print(text[:900], "\n")

        try:
            layout = extract_layout_from_pdf(file_path)
        except Exception:
            layout = {}

        # Header (erste 30 Zeilen) und Footer (letzte 35) separat aufbewahren:
        # werden dem LLM explizit als party_hints übergeben und bei der Evidence prependet
        header_lines = "\n".join(text.splitlines()[:30]).strip()
        footer_lines = "\n".join(text.splitlines()[-35:]).strip()

        # 2) Base Parsing
        base: Dict[str, Any] = extract_invoice_data(text)

        layout_parties = _extract_parties_from_layout(layout)
        if layout_parties.get("absender"):
            base["absender"] = layout_parties["absender"]
        if layout_parties.get("empfaenger"):
            base["empfaenger"] = layout_parties["empfaenger"]

        cutoff = re.compile(
            r"(Rechnungsempf.nger|Bill\s*to|"
            r"Rechnungsadresse|Ship\s*to|"
            r"Zahlungsempf.nger)", re.I)
        for key in ["absender", "empfaenger"]:
            val = base.get(key) or ""
            m = cutoff.search(val)
            if m:
                base[key] = val[:m.start()].strip()

        # LLM als Korrektur-Schicht für Parties und Positionen
        needs_party_fix = (
            not base.get("empfaenger")
            or len(base.get("empfaenger", "")) < 5
            or re.search(r"^\d{3,6}$", (base.get("empfaenger") or "").split("\n")[0])
        )
        needs_position_fix = (
            len(base.get("positionen", [])) == 0
        ) or any(
            len(p.get("beschreibung", "").strip()) <= 2
            or p.get("beschreibung", "").strip().isdigit()
            for p in base.get("positionen", [])
        )

        if needs_party_fix or needs_position_fix:
            print("🤖 LLM korrigiert Parties/Positionen...")
            llm_fixes = _llm_fix_parties_and_positions(text, base)

            if needs_party_fix:
                if llm_fixes.get("absender"):
                    base["absender"] = llm_fixes["absender"]
                if llm_fixes.get("empfaenger"):
                    base["empfaenger"] = llm_fixes["empfaenger"]

            for key in ["absender", "empfaenger"]:
                val = base.get(key, "") or ""
                if re.search(r"\d{5}", val) and ", " in val:
                    base[key] = val.replace(", ", "\n")

            if needs_position_fix:
                if llm_fixes.get("positionen"):
                    base["positionen"] = llm_fixes["positionen"]

        print("📊 Basisdaten:", base)

        # 3) Upsert in Vektor-DB
        try:
            upsert_invoice_text(pdf_file, text, base)
        except Exception as e:
            print(f"⚠️ Upsert/Chroma Problem bei {pdf_file}: {e}")

        # 4) Retrieval Fix
        try:
            fixed_raw = retrieve_fix_fields(
                pdf_file,
                base.get("waehrung"),
                top_k=retrieval_top_k,
                debug=debug,
            )
        except Exception as e:
            print(f"⚠️ Retrieval Fehler bei {pdf_file}: {e}")
            fixed_raw = {}

        fixed = _ensure_fixed_shape(fixed_raw)
        print("🧠 Retrieval:", fixed)

        # 4b) Deterministic Math-Repair Hints
        math_hint = _derive_math_hints(base=base, retrieval=fixed, priors=priors_global)
        retrieval_plus = _merge_retrieval_with_math(fixed, math_hint)

        if debug:
            print("🧮 Math-hint:", math_hint)
            print("🧠 Retrieval+Math:", retrieval_plus)

        # 5) Facts (für LLM Kontext)
        try:
            facts = build_invoice_facts(pdf_file, text, base, retrieval_plus)
        except Exception as e:
            print(f"⚠️ Facts Fehler bei {pdf_file}: {e}")
            facts = {}

        # ✅ EXTRA: LLM bekommt explizit Header+Footer + klare Party-Regeln
        facts["party_hints"] = {
            "header_lines": header_lines,
            "footer_lines": footer_lines,
            "rules": [
                "absender(empfaenger) müssen echte Adress-/Firmenblöcke sein, keine Tabellen/Positionen/Totals.",
                "Bevorzuge Absender-Block mit Firmennamen + Rechtsform + Adresse + (optional) VAT/IBAN.",
                "Empfänger-Block ist i.d.R. Rechnungsadresse (Bill to / Rechnungsempfänger) oder der zweite Adressblock.",
                "Wenn Footer klar Absenderdaten enthält (IBAN/VAT/Impressum), dann ist das sehr likely Absender.",
            ],
        }

        facts["pipeline_context"] = {
            "base": {k: base.get(k) for k in [
                "waehrung", "rechnungsnummer", "rechnungsdatum", "absender", "empfaenger",
                "netto", "steuer", "versand", "brutto", "validiert"
            ]},
            "retrieval": retrieval_plus,
            "priors": priors_global,
        }

        # 6) Candidate ohne LLM
        llm_out: Dict[str, Any] = {}
        candidate = _safe_merge_results(
            base=base,
            retrieval=retrieval_plus,
            llm=llm_out,
            priors=priors_global,
            debug=debug,
        )

        # 7) Default: candidate ist final, wenn kein LLM nötig
        final = candidate

        # -------------------------
        # 7) LLM Pass 1
        # -------------------------
        if use_llm and should_call_llm(base=base, retrieval=retrieval_plus, candidate=candidate):
            print("🧩 LLM-Fallback wird genutzt (Pass 1).")

            # Evidence 1: Standard Retrieval
            try:
                evidence = retrieve_evidence_chunks(
                    pdf_file,
                    base.get("waehrung"),
                    top_k=llm_evidence_top_k,
                )
            except Exception as e:
                print(f"⚠️ Evidence Retrieval Fehler bei {pdf_file}: {e}")
                evidence = []

            # ✅ Header + Footer immer ganz vorne (Parties/Invoice-No/Date stehen dort)
            evidence = _prepend_if_present(evidence, footer_lines)
            evidence = _prepend_if_present(evidence, header_lines)

            # Wenn Parties verdächtig: mehr Evidence zulassen
            if _party_bad(candidate.get("absender"), candidate.get("empfaenger")):
                try:
                    extra_party = retrieve_evidence_chunks(
                        pdf_file,
                        base.get("waehrung"),
                        top_k=max(llm_evidence_top_k, 12),
                    )
                except Exception:
                    extra_party = []
                evidence = _dedup_chunks((evidence or []) + (extra_party or []), limit=max(llm_evidence_top_k, 12))
            else:
                evidence = _dedup_chunks(evidence or [], limit=llm_evidence_top_k)

            cand_err = _compute_math_error(
                candidate.get("netto"),
                candidate.get("steuer"),
                candidate.get("versand"),
                candidate.get("brutto"),
            )

            # ✅ Repair Request präziser für Parties
            facts["repair_request"] = {
                "goal": (
                    "Fix ALL fields. Ensure amounts consistent: netto + steuer + versand ≈ brutto. "
                    "Parties MUST be two distinct address/company blocks: absender=seller, empfaenger=buyer. "
                    "DO NOT include totals/table/positions text inside parties."
                ),
                "current_candidate": {k: candidate.get(k) for k in [
                    "waehrung", "rechnungsnummer", "rechnungsdatum", "absender", "empfaenger",
                    "netto", "steuer", "versand", "brutto", "validiert", "_error", "_source"
                ]},
                "current_math_error": cand_err,
                "party_bad": _party_bad(candidate.get("absender"), candidate.get("empfaenger")),
            }

            try:
                llm_out = llm_extract_invoice(
                    evidence_chunks=evidence,
                    facts=facts,
                    currency_hint=base.get("waehrung"),
                    debug=debug,
                    timeout_s=180,
                    num_predict=550,
                    retries=1,
                )
            except Exception as e:
                print(f"⚠️ LLM Fehler/Timeout bei {pdf_file}: {e}")
                llm_out = {}

            # Merge + Meta-Repair + Party-Fix + optional Swap
            final = _safe_merge_results(
                base=base,
                retrieval=retrieval_plus,
                llm=llm_out,
                priors=priors_global,
                debug=debug,
            )
            final = _apply_llm_meta_repairs(final, llm_out)
            final = fix_parties_best_of(
                base=base,
                llm_out=llm_out,
                final=final,
                header_lines=header_lines,
            )
            final = _postprocess_parties_swap_if_needed(final)

            # -------------------------
            # 7b) LLM Pass 2 (Forced Repair)
            # -------------------------
            if _needs_second_pass(final):
                print("🧩 LLM-Fallback wird genutzt (Pass 2 - forced repair).")

                try:
                    evidence2 = retrieve_evidence_chunks(
                        pdf_file,
                        base.get("waehrung"),
                        top_k=max(llm_evidence_top_k, 14),
                    )
                except Exception:
                    evidence2 = evidence

                evidence2 = _prepend_if_present(evidence2, footer_lines)
                evidence2 = _prepend_if_present(evidence2, header_lines)

                # Pass2: wenn Parties weiterhin verdächtig -> noch mehr Evidence zulassen
                limit2 = max(llm_evidence_top_k, 14) if _party_bad(final.get("absender"), final.get("empfaenger")) else max(llm_evidence_top_k, 10)
                evidence2 = _dedup_chunks(evidence2 or [], limit=limit2)

                err2 = _compute_math_error(
                    final.get("netto"),
                    final.get("steuer"),
                    final.get("versand"),
                    final.get("brutto"),
                )

                facts["repair_request"] = {
                    "goal": (
                        "Force repair. Output correct values from evidence so that math holds. "
                        "Parties MUST be correct and clearly separated address blocks. "
                        "Use header/footer to decide seller vs buyer."
                    ),
                    "after_pass1_final": {k: final.get(k) for k in [
                        "waehrung", "rechnungsnummer", "rechnungsdatum", "absender", "empfaenger",
                        "netto", "steuer", "versand", "brutto", "validiert", "_error", "_source"
                    ]},
                    "math_error_after_pass1": err2,
                    "party_bad_after_pass1": _party_bad(final.get("absender"), final.get("empfaenger")),
                }

                try:
                    llm_out2 = llm_extract_invoice(
                        evidence_chunks=evidence2,
                        facts=facts,
                        currency_hint=base.get("waehrung"),
                        debug=debug,
                        timeout_s=240,
                        num_predict=750,
                        retries=2,
                    )
                except Exception as e:
                    print(f"⚠️ LLM Pass2 Fehler/Timeout bei {pdf_file}: {e}")
                    llm_out2 = {}

                final2 = _safe_merge_results(
                    base=base,
                    retrieval=retrieval_plus,
                    llm=llm_out2,
                    priors=priors_global,
                    debug=debug,
                )
                final2 = _apply_llm_meta_repairs(final2, llm_out2)
                final2 = fix_parties_best_of(
                    base=base,
                    llm_out=llm_out2,
                    final=final2,
                    header_lines=header_lines,
                )
                final2 = _postprocess_parties_swap_if_needed(final2)

                        # Pass2 übernehmen wenn besser validiert, oder wenn Pass1 sowieso nicht validiert war
                if (final2.get("validiert") is True) or (final.get("validiert") is not True):
                    final = final2
                    llm_out = llm_out2

        print("✅ Final:", final)

        # 8) Debug JSON pro Rechnung
        out_json = {
            "filename": pdf_file,
            "base": base,
            "retrieval": fixed,
            "math_hint": math_hint,
            "retrieval_plus": retrieval_plus,
            "facts": facts,
            "llm": llm_out,
            "candidate": candidate,
            "final": final,
        }
        try:
            with open(os.path.join(INVOICE_JSON_DIR, pdf_file + ".json"), "w", encoding="utf-8") as f:
                json.dump(out_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Konnte Debug-JSON nicht schreiben: {e}")

        finals.append(final)

        # 9) CSV Row
        csv_rows.append({
            "filename": pdf_file,
            "waehrung": final.get("waehrung"),
            "rechnungsnummer": final.get("rechnungsnummer"),
            "rechnungsdatum": final.get("rechnungsdatum"),
            "absender": final.get("absender"),
            "empfaenger": final.get("empfaenger"),
            "netto": final.get("netto"),
            "steuer": final.get("steuer"),
            "versand": final.get("versand"),
            "brutto": final.get("brutto"),
            "validiert": final.get("validiert"),
            "_source": final.get("_source"),
            "_error": final.get("_error"),
            "_valid_constraints": final.get("_valid_constraints"),
            "_bonus": final.get("_bonus"),
        })

    # 10) 🌍 Priors neu lernen (nur validierte Finals)
    valid_finals = [f for f in finals if f and f.get("validiert") is True]
    if valid_finals:
        priors_new = learn_global_patterns(valid_finals)
        print("\n🌍 Gelernte globale Muster (nur validierte Finals):")
        print(priors_new)
        save_priors(priors_new)
        print("💾 Priors in priors.json gespeichert.")
    else:
        print("\nℹ️ Keine validierten Finals → Priors werden nicht überschrieben.")

    write_results_csv(csv_rows)



if __name__ == "__main__":
    run_pipeline(pdf_folder="pdfs", use_llm=True, debug=False)
