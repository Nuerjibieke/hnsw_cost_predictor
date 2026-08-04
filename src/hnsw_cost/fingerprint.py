"""Build compact fingerprints for reusable experiment outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint(values: Any) -> str:
    """Hash a JSON-compatible value with stable key ordering."""
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
