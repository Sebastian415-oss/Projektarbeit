from __future__ import annotations

# scoring.py
# Kernlogik für Validierung und Varianten-Selektion.
# merge_results() ist der zentrale Entscheidungspunkt: es baut aus Base/Retrieval/LLM
# mehrere Kandidaten, berechnet für jeden einen Fehler-Score und wählt den besten.
# Wird ausschließlich von main.py genutzt.

import re
from typing import Dict, Any, Optional, List, Tuple

from utils import safe_float as _safe_float


# ----------------------------
# Validation + Constraints
# ----------------------------

def validate_invoice(fields: Dict[str, Any], tol: float = 0.2) -> bool:
    """
    Ziel: "am Ende muss es stimmen"
    - brutto MUSS existieren
    - netto MUSS existieren
    - steuer darf fehlen (Receipt/Payment) -> wird als 0 behandelt
    - versand darf fehlen -> 0
    """
    try:
        brutto = _safe_float(fields.get("brutto"))
        netto = _safe_float(fields.get("netto"))
        steuer = _safe_float(fields.get("steuer"))
        versand = _safe_float(fields.get("versand"))

        if brutto is None or netto is None:
            return False

        s = 0.0 if steuer is None else float(steuer)
        v = 0.0 if versand is None else float(versand)

        return abs((float(netto) + s + v) - float(brutto)) <= float(tol)
    except Exception:
        return False


def _check_hard_constraints(fields: Dict[str, Any]) -> bool:
    """
    Harte Plausibilitätsregeln, die quasi nie verletzt sein dürfen.
    """
    n = _safe_float(fields.get("netto"))
    s = _safe_float(fields.get("steuer"))
    v = _safe_float(fields.get("versand"))
    b = _safe_float(fields.get("brutto"))

    if b is not None and n is not None and b + 1e-6 < n:
        return False
    if b is not None and s is not None and s > b + 1e-6:
        return False
    if b is not None and v is not None and v > b + 1e-6:
        return False

    if s is not None and s < -0.01:
        return False

    # Beträge über 1 Mio. oder stark negativ sind für Rechnungen unrealistisch
    for val in (n, s, v, b):
        if val is not None and (val < -0.01 or val > 1_000_000):
            return False

    return True


def _derive_brutto_if_missing(fields: Dict[str, Any]) -> Dict[str, Any]:
    newf = dict(fields)
    if newf.get("brutto") is None:
        n = _safe_float(newf.get("netto"))
        s = _safe_float(newf.get("steuer"))
        v = _safe_float(newf.get("versand"))
        if n is not None:
            s0 = 0.0 if s is None else float(s)
            v0 = 0.0 if v is None else float(v)
            newf["brutto"] = round(float(n) + s0 + v0, 2)
            newf["_brutto_source"] = "derived"
    return newf


# ----------------------------
# Meta quality heuristics (FINAL-relevant!)
# ----------------------------

_SELLER_CUES = re.compile(r"(USt|USt-Id|VAT|Tax\s*ID|Steuernummer|KVK|HRB|IBAN|BIC|Bank|Handelsregister|Reg\.)", re.I)
_BUYER_CUES  = re.compile(r"(Rechnungsadresse|Bill\s*to|Invoice\s*to|Sold\s*to|Ship\s*to|Lieferadresse|Customer|Kunde|Rechnungsempfänger)", re.I)

def _looks_like_noise_party(s: Any) -> bool:
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

def _is_mixed_party_block(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    sc = len(_SELLER_CUES.findall(t))
    bc = len(_BUYER_CUES.findall(t))
    if sc >= 1 and bc >= 1:
        return True
    if sc >= 2 and (("\n" in t and t.count("\n") >= 4) or re.search(r"\b\d{5}\b", t)):
        if re.search(r"\b(www\.|https?://|\S+@\S+)\b", t, re.I):
            return True
    return False

def _party_similarity_penalty(a: Any, b: Any) -> float:
    # Very cheap: if both strings exist and are highly overlapping -> penalty.
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    aa = a.strip()
    bb = b.strip()
    if not aa or not bb:
        return 0.0
    # If one contains the other and both are long, they are likely the same block duplicated.
    if len(aa) > 40 and len(bb) > 40:
        if aa in bb or bb in aa:
            return 2.5
    return 0.0


# ----------------------------
# Error / Ranking
# ----------------------------

def _compute_validation_error(
    fields: Dict[str, Any],
    priors: Optional[Dict[str, float]] = None,
) -> float:
    """
    Immer numerisch (kleiner = besser).
    Bausteine:
    1) Mathefehler
    2) Missing-Penalties (Zahlen + Meta!)
    3) Priors-Regularisierung
    4) Harte Unplausibilitäten -> sehr großer Fehler
    """
    n = _safe_float(fields.get("netto"))
    s = _safe_float(fields.get("steuer"))
    v = _safe_float(fields.get("versand"))
    b = _safe_float(fields.get("brutto"))

    v0 = 0.0 if v is None else float(v)

    # harte Unplausi sofort bestrafen
    if b is not None and n is not None and b + 1e-6 < n:
        return 10_000.0 + abs(n - b)
    if b is not None and s is not None and s > b + 1e-6:
        return 10_000.0 + abs(s - b)
    if b is not None and v is not None and v > b + 1e-6:
        return 10_000.0 + abs(v - b)

    miss = 0.0
    # Amount missing penalties
    if b is None:
        miss += 20.0
    if n is None:
        miss += 18.0
    if s is None:
        miss += 4.0
    if fields.get("versand") is None:
        miss += 0.8

    # ✅ Meta penalties (FINAL correctness)
    if not fields.get("rechnungsnummer"):
        miss += 6.0
    if not fields.get("rechnungsdatum"):
        miss += 3.0

    absender = fields.get("absender")
    empfaenger = fields.get("empfaenger")
    if _looks_like_noise_party(absender):
        miss += 5.0
    if _looks_like_noise_party(empfaenger):
        miss += 5.0

    # Mischblock: strongly discouraged
    if _is_mixed_party_block(absender):
        miss += 4.0
    if _is_mixed_party_block(empfaenger):
        miss += 4.0

    miss += _party_similarity_penalty(absender, empfaenger)

    # Fallback-Fehler 8.0 wenn keine Beträge vorhanden – bewusst hoch damit diese
    # Variante nie "gewinnt" wenn eine bessere Option existiert
    base_error = 8.0
    if b is not None and n is not None:
        s0 = 0.0 if s is None else float(s)
        base_error = abs((float(n) + s0 + v0) - float(b))

    reg = 0.0
    if priors:
        tvat = priors.get("typical_vat")
        if tvat is not None and n is not None and s is not None and n > 0:
            expected_s = float(n) * float(tvat)
            reg += 0.25 * abs(expected_s - float(s))

        tship = priors.get("typical_shipping")
        if tship is not None and fields.get("versand") is not None and v is not None:
            reg += 0.10 * abs(float(v) - float(tship))

    if base_error < 0.02:
        base_error *= 0.5

    return base_error + miss + reg


# ----------------------------
# LLM helper
# ----------------------------

def _flatten_llm_amount(llm: Dict[str, Any], key: str) -> Tuple[Optional[float], float]:
    # Normalisiert LLM-Antworten: entweder direkte Zahl oder {"value": ..., "confidence": ...}
    obj = (llm or {}).get(key)
    if isinstance(obj, (int, float, str)):
        # Kein Confidence-Wert vom LLM → 0.75 als moderater Default
        return _safe_float(obj), 0.75
    if isinstance(obj, dict):
        val = _safe_float(obj.get("value"))
        conf = _safe_float(obj.get("confidence"))
        c = 0.0 if conf is None else float(conf)
        c = max(0.0, min(1.0, c))
        return val, c
    return None, 0.0


# ----------------------------
# Merge: base + retrieval + llm
# ----------------------------

def merge_results(
    base: Dict[str, Any],
    retrieval: Dict[str, Any],
    llm: Optional[Dict[str, Any]] = None,
    priors: Optional[Dict[str, float]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Baut Varianten und wählt:
    1) beste validierte (min error - bonus)
    2) sonst beste unvalidierte + _needs_llm=True
    """
    llm = llm or {}

    llm_vals: Dict[str, Optional[float]] = {}
    llm_conf: Dict[str, float] = {}
    for k in ["netto", "steuer", "versand", "brutto"]:
        v, c = _flatten_llm_amount(llm, k)
        llm_vals[k] = v
        llm_conf[k] = c

    llm_has_any_amount = any(llm_vals.get(k) is not None for k in ["netto", "steuer", "versand", "brutto"])
    llm_has_any_meta = any((llm or {}).get(k) is not None for k in ["rechnungsnummer", "rechnungsdatum", "absender", "empfaenger", "waehrung"])
    llm_has_any = llm_has_any_amount or llm_has_any_meta

    variants: List[Dict[str, Any]] = []

    def add_variant(source: str, overrides: Dict[str, Any], bonus: float = 0.0) -> None:
        v = dict(base)

        for meta_k in ["rechnungsnummer", "rechnungsdatum", "absender", "empfaenger", "waehrung"]:
            if overrides.get(meta_k) is not None:
                v[meta_k] = overrides[meta_k]

        for amt_k in ["netto", "steuer", "versand", "brutto"]:
            if overrides.get(amt_k) is not None:
                v[amt_k] = overrides[amt_k]

        v["_source"] = source
        v = _derive_brutto_if_missing(v)
        v["_valid_constraints"] = _check_hard_constraints(v)
        v["_error"] = _compute_validation_error(v, priors=priors)
        v["_bonus"] = float(bonus)
        v["validiert"] = validate_invoice(v)
        variants.append(v)

    # base
    add_variant("base", {}, bonus=0.0)

    # Einzelfelder und Kombinationen aus Retrieval/LLM als separate Varianten testen –
    # manchmal ist nur ein Feld vom Retrieval korrekt, die anderen nicht.
    for f in ["netto", "steuer", "versand", "brutto"]:
        if retrieval.get(f) is not None:
            add_variant(f"base+retr_{f}", {f: retrieval.get(f)}, bonus=0.05)

    combos = [("netto", "steuer"), ("netto", "brutto"), ("steuer", "brutto")]
    for a, b in combos:
        if retrieval.get(a) is not None and retrieval.get(b) is not None:
            add_variant(f"base+retr_{a}_{b}", {a: retrieval[a], b: retrieval[b]}, bonus=0.08)

    # llm single-field (only if llm provides anything)
    if llm_has_any:
        for f in ["netto", "steuer", "versand", "brutto"]:
            if llm_vals.get(f) is not None:
                add_variant(f"base+llm_{f}", {f: llm_vals[f]}, bonus=0.14 * llm_conf.get(f, 0.0))

        for a, b in combos:
            if llm_vals.get(a) is not None and llm_vals.get(b) is not None:
                add_variant(
                    f"base+llm_{a}_{b}",
                    {a: llm_vals[a], b: llm_vals[b]},
                    bonus=0.14 * (llm_conf.get(a, 0.0) + llm_conf.get(b, 0.0)) / 2.0,
                )

        full_overrides = dict(llm_vals)
        for meta_k in ["rechnungsnummer", "rechnungsdatum", "absender", "empfaenger", "waehrung"]:
            if llm.get(meta_k) is not None:
                full_overrides[meta_k] = llm.get(meta_k)

        full_bonus = 0.14 * sum(llm_conf.get(k, 0.0) for k in ["netto", "steuer", "versand", "brutto"]) / 4.0
        add_variant("base+llm_full", full_overrides, bonus=full_bonus)

    # mix: prefer retrieval else llm (only if it changes something)
    mix = {}
    for k in ["netto", "steuer", "versand", "brutto"]:
        mix[k] = retrieval.get(k) if retrieval.get(k) is not None else llm_vals.get(k)

    mix_changes = any(
        (mix.get(k) is not None and mix.get(k) != base.get(k))
        for k in ["netto", "steuer", "versand", "brutto"]
    )
    if mix_changes:
        add_variant("base+mix_all", mix, bonus=0.10)

    def score(v: Dict[str, Any]) -> float:
        return float(v.get("_error", 1e9)) - float(v.get("_bonus", 0.0))

    # Erst nur mathematisch validierte Varianten betrachten; falls keine existiert,
    # die "am wenigsten schlechte" nehmen und _needs_llm setzen
    valid_pool = [v for v in variants if v.get("_valid_constraints", True)]
    validated = [v for v in valid_pool if v.get("validiert") is True]

    if validated:
        best = min(validated, key=score)
        best["_needs_llm"] = False
    else:
        best = min(valid_pool, key=score) if valid_pool else variants[0]
        best["_needs_llm"] = True

    if debug:
        print("\n[merge_results] Kandidaten:")
        for v in variants:
            print(
                f" - {str(v.get('_source')):18s} "
                f"valid={v.get('validiert')} "
                f"err={float(v.get('_error', 0.0)):.3f} "
                f"bonus={float(v.get('_bonus', 0.0)):.3f} "
                f"ok={v.get('_valid_constraints')}"
            )
        print("➡️ gewählt:", best.get("_source"), "validiert=", best.get("validiert"), "needs_llm=", best.get("_needs_llm"))

    return best
