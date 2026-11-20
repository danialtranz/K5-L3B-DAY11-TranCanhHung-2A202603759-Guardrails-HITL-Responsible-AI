"""
Deterministic secret / leak shield for Blue Team (demo hardening).

Catches plaintext and common obfuscations (spacing, ZWSP, reverse, ROT13,
base64, hex) without calling an LLM. Fail-closed on match.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata

from core.config import DEMO_SECRETS

ZERO_WIDTH = "\u200b\u200c\u200d\ufeff\u2060"

_B64_TOKEN = re.compile(r"[A-Za-z0-9+/_-]{8,}={0,2}")
_HEX_TOKEN = re.compile(r"(?:[0-9a-fA-F]{2}[\s:-]?){6,}")

# Hard lab secrets + common aliases
_DEFAULT_NEEDLES = (
    "admin123",
    "skvinbanksecret2024",
    "sk-vinbank-secret-2024",
    "dbvinbankinternal",
    "db.vinbank.internal",
)

_SECRET_REGEX = (
    r"\badmin123\b",
    r"sk-[a-z0-9-]{8,}",
    r"db\.vinbank\.internal(?::\d+)?",
    r"(?:password|mật\s*khẩu)\s*(?:[:=]|is)\s*\S+",
    r"api[_\s-]?key\s*(?:[:=]|is)\s*\S+",
)


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return normalized.translate(str.maketrans("", "", ZERO_WIDTH))


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(text).casefold())


def _needles() -> list[str]:
    out: list[str] = []
    for n in list(DEMO_SECRETS) + list(_DEFAULT_NEEDLES):
        s = (n or "").strip()
        if s and s not in out:
            out.append(s)
    return out


def _decoded_views(text: str) -> list[str]:
    base = normalize(text or "")
    views = [base, base[::-1], codecs.decode(base, "rot_13")]
    for token in _B64_TOKEN.findall(base):
        padded = token.replace("-", "+").replace("_", "/")
        padded += "=" * (-len(padded) % 4)
        try:
            decoded = base64.b64decode(padded).decode("utf-8", errors="ignore")
        except (binascii.Error, ValueError):
            continue
        if decoded:
            views.extend([decoded, decoded[::-1]])
    for token in _HEX_TOKEN.findall(base):
        digits = re.sub(r"[^0-9a-fA-F]", "", token)
        digits = digits[: len(digits) - (len(digits) % 2)]
        try:
            decoded = bytes.fromhex(digits).decode("utf-8", errors="ignore")
        except ValueError:
            continue
        if decoded:
            views.extend([decoded, decoded[::-1]])
    return views


def mentions_secret(text: str) -> bool:
    """True if text contains (or lightly obfuscates) a VinBank lab secret."""
    needles = [squash(n) for n in _needles() if squash(n)]
    for view in _decoded_views(text):
        sq = squash(view)
        if any(n in sq for n in needles):
            return True
        if any(re.search(p, normalize(view), re.IGNORECASE) for p in _SECRET_REGEX):
            return True
    return False


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Redact known secret patterns; also blank whole reply if squash-leak remains."""
    issues: list[str] = []
    redacted = text or ""
    for pattern in _SECRET_REGEX:
        if re.search(pattern, redacted, re.IGNORECASE):
            issues.append(f"pattern:{pattern[:24]}")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
    # Explicit substrings
    for n in _needles():
        if n and n.lower() in redacted.lower():
            issues.append(f"substring:{n[:20]}")
            redacted = re.sub(re.escape(n), "[REDACTED]", redacted, flags=re.IGNORECASE)
    if mentions_secret(redacted):
        issues.append("obfuscated_secret")
        redacted = (
            "Response blocked by secret_shield: protected lab credentials detected."
        )
    return redacted, issues
