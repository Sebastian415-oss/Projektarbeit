# validation.py – Weiterleitungsdatei für Rückwärtskompatibilität.
# Alle Funktionen wurden nach priors.py und scoring.py migriert (bessere Implementierungen).
# Diese Datei bleibt als Shim, damit eventuelle externe Imports weiter funktionieren.

from priors import load_priors, save_priors, learn_global_patterns
from scoring import (
    validate_invoice,
    _compute_validation_error,
    _check_hard_constraints,
    _derive_brutto_if_missing,
    merge_results,
)

__all__ = [
    "load_priors",
    "save_priors",
    "learn_global_patterns",
    "validate_invoice",
    "_compute_validation_error",
    "_check_hard_constraints",
    "_derive_brutto_if_missing",
    "merge_results",
]
