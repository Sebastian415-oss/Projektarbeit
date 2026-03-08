# utils.py
# Projektweite Hilfsfunktionen. Aktuell nur safe_float.
# Importiert von scoring.py und llm_extract.py.

import re
from typing import Any, Optional


def safe_float(x: Any) -> Optional[float]:
    """
    Konvertiert einen beliebigen Wert robust in float oder None.
    - Entfernt Währungssymbole (€, $, EUR, USD) und Leerzeichen
    - Behandelt Klammern als negativ: (1.23) -> -1.23
    - Behandelt Em-Dash (−) als Minus
    - Entfernt Tausenderpunkte: 1.234.56 -> 1234.56
    - Plausicheck: Beträge > 10.000.000 -> None
    - Rundet auf 2 Dezimalstellen
    """
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return round(float(x), 2)
        s = str(x).strip()
        if not s:
            return None

        s = s.replace("€", "").replace("$", "").replace("EUR", "").replace("USD", "")
        s = s.replace("\xa0", " ").replace(" ", "")

        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1].strip()

        s = s.replace("−", "-")
        s = s.replace(",", ".")
        s = re.sub(r"[^0-9.\-+]", "", s)

        parts = s.split(".")
        if len(parts) > 2:
            s = "".join(parts[:-1]) + "." + parts[-1]

        val = float(s)
        if neg:
            val = -abs(val)

        if abs(val) > 10_000_000:
            return None

        return round(val, 2)
    except Exception:
        return None
