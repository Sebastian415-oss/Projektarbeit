# priors.py
# Lernt globale Muster aus verarbeiteten Rechnungen (MwSt-Satz, Versandkosten)
# und speichert sie in priors.json. Diese werden in scoring.py als Regularisierung genutzt.
# Wird von main.py am Ende jedes Pipeline-Laufs aufgerufen.

from typing import Dict, Any, List, Optional
import json
import os
from config import PRIORS_FILE

def _median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    return xs[len(xs)//2]

def learn_global_patterns(invoices: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    invoices: liste von final dicts.
    Lernt NUR aus validierten Fällen + plausiblen VAT ratios.
    """
    vat_ratios: List[float] = []
    shipping_values: List[float] = []

    for inv in invoices:
        if not inv.get("validiert"):
            continue

        netto = inv.get("netto")
        steuer = inv.get("steuer")
        versand = inv.get("versand")

        if netto is not None and steuer is not None and netto > 0:
            ratio = steuer / netto
            # 0–35 % ist plausibler MwSt-Bereich; drüber sind es vermutlich Parsing-Fehler
            if 0.0 <= ratio <= 0.35:
                vat_ratios.append(ratio)

        if versand is not None and versand >= 0:
            shipping_values.append(float(versand))

    priors: Dict[str, float] = {}
    vat = _median(vat_ratios)
    ship = _median(shipping_values)

    if vat is not None:
        priors["typical_vat"] = round(vat, 4)
    if ship is not None:
        priors["typical_shipping"] = round(ship, 2)

    return priors

def save_priors(priors: Dict[str, float]) -> None:
    with open(PRIORS_FILE, "w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)

def load_priors() -> Dict[str, float]:
    if not os.path.exists(PRIORS_FILE):
        return {}
    try:
        with open(PRIORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
