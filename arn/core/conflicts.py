"""
Lightweight contradiction and claim extraction helpers for ARN.

This is intentionally small and offline-friendly. It is not a full NLI model;
it catches high-value agent-memory conflicts such as:
  - Mohamed prefers Python
  - Mohamed prefers JavaScript now
and lets ARN keep both facts with timestamps while warning the agent that the
newer/older facts disagree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

_SUBJECT = r"(?P<subject>user|mohamed|agent|developer|researcher|manager|system|[A-Z][a-zA-Z0-9_-]{1,40})"
_OBJECT = r"(?P<object>[A-Za-z0-9_+#.\-/ ]{2,80})"

_PATTERNS: list[tuple[str, str]] = [
    ("preference", rf"\b{_SUBJECT}\s+(?:currently\s+|now\s+)?(?:prefers?|likes?|loves?)\s+{_OBJECT}"),
    ("preference", rf"\b{_SUBJECT}\s+(?:switched?|changed?|moved?)\s+(?:from\s+[A-Za-z0-9_+#.\-/ ]+?\s+)?to\s+{_OBJECT}"),
    ("preference", rf"\b{_SUBJECT}'s\s+favorite\s+(?:language|tool|framework|editor|stack)?\s*(?:is|=)\s+{_OBJECT}"),
    ("coding_language", rf"\b{_SUBJECT}\s+(?:codes?|programs?|develops?|scripts?)\s+(?:in|with|using)\s+{_OBJECT}"),
    ("uses", rf"\b{_SUBJECT}\s+(?:uses?|runs?|works\s+with)\s+{_OBJECT}"),
    ("identity", rf"\b{_SUBJECT}\s+(?:is|are)\s+{_OBJECT}"),
    ("location", rf"\b{_SUBJECT}\s+(?:lives?|is\s+based|works?)\s+(?:in|at|from)\s+{_OBJECT}"),
    ("role",     rf"\b{_SUBJECT}\s+(?:works?\s+as|is\s+a|is\s+an)\s+{_OBJECT}"),
    ("goal",     rf"\b{_SUBJECT}\s+(?:wants?\s+to|is\s+trying\s+to|plans?\s+to|needs?\s+to)\s+{_OBJECT}"),
    ("owns",     rf"\b{_SUBJECT}\s+(?:has|owns?)\s+(?:a\s+|an\s+|the\s+)?{_OBJECT}"),
]

# Port/version patterns use a different extractor (no subject group)
_PORT_PATTERN = re.compile(r"(?:port|running\s+on|listening\s+on|on\s+port)\s+([0-9]{2,5})", re.I)
_VERSION_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_\-\.]+)\s+v(?:ersion\s+)?([0-9][0-9A-Za-z\.\-]+)", re.I)

# Keywords that signal a fact is changing
_CHANGE_MARKERS = re.compile(
    r"\b(?:now|switched?\s+to|changed?\s+to|moved?\s+to|updated?\s+to|no\s+longer|instead\s+of|replaced?\s+with)\b",
    re.I
)

_STOP_OBJECT_TRAILERS = re.compile(r"\b(?:because|but|and|when|while|for|with|on|at|from|to)\b.*$", re.I)
_TRAILING_TEMPORAL = re.compile(r"\b(?:now|currently|right now|these days|today|at present)\s*$", re.I)
_BAD_SUBJECTS = {
    "no", "longer", "now", "currently", "previously", "formerly",
    "before", "after", "then", "that", "this", "there", "here",
}

@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str
    object: str

    def key(self) -> tuple[str, str]:
        return (self.subject.lower(), self.relation.lower())

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_object(value: str) -> str:
    value = _STOP_OBJECT_TRAILERS.sub("", value)
    value = _TRAILING_TEMPORAL.sub("", value)
    value = value.strip(" .,!?:;\"'()[]{}")
    return re.sub(r"\s+", " ", value).strip()


def _valid_subject(value: str) -> bool:
    return value.strip().lower() not in _BAD_SUBJECTS


def extract_claim(text: str) -> Optional[Claim]:
    """Extract a simple durable claim from a memory string, if one is obvious."""
    if not text or len(text.strip()) < 4:
        return None
    for relation, pattern in _PATTERNS:
        for match in re.finditer(pattern, text.strip(), flags=re.I):
            subject = match.group("subject").strip().lower()
            if not _valid_subject(subject):
                continue
            obj = _clean_object(match.group("object"))
            if not obj:
                continue
            return Claim(subject=subject, relation=relation, object=obj.lower())
    return None

def fact_key(claim: Claim) -> str:
    """
    Stable string key for a (subject, relation) pair.
    Used to link episodes that represent the same evolving fact.
    e.g. "user::preference" — regardless of what the object currently is.
    """
    return f"{claim.subject.lower()}::{claim.relation.lower()}"


def find_conflicts(new_text: str, episodes: Iterable[dict]) -> list[dict]:
    """
    Compare a new claim against existing episodic memories.

    A conflict means the subject and relation match but the object changed.
    Both memories are preserved; callers attach this as metadata so the agent
    can reason about timestamps and ask for clarification when needed.
    """
    new_claim = extract_claim(new_text)
    if not new_claim:
        return []
    conflicts: list[dict] = []
    for ep in episodes:
        old_claim = extract_claim(ep.get("content", ""))
        if not old_claim:
            continue
        if old_claim.key() != new_claim.key():
            continue
        if old_claim.object == new_claim.object:
            continue
        conflicts.append({
            "kind": "claim_conflict",
            "new_claim": new_claim.to_dict(),
            "old_claim": old_claim.to_dict(),
            "old_episode_id": ep.get("id"),
            "old_content": ep.get("content"),
            "old_time_context": (ep.get("context") or {}).get("time_context", "current"),
            "old_created_at": ep.get("created_at"),
        })
    return conflicts


def is_change_statement(text: str) -> bool:
    """Return True if the text signals that a fact is changing/updating."""
    return bool(_CHANGE_MARKERS.search(text))


def auto_extract(text: str) -> list[Claim]:
    """
    Extract ALL detectable claims from text, not just the first.
    Used for auto-tagging memories at store time.
    Returns a list (often empty, sometimes multiple claims).
    """
    if not text or len(text.strip()) < 4:
        return []
    found: list[Claim] = []
    seen_keys: set[tuple[str, str]] = set()

    for relation, pattern in _PATTERNS:
        for match in re.finditer(pattern, text.strip(), flags=re.I):
            try:
                subject = match.group("subject").strip().lower()
                if not _valid_subject(subject):
                    continue
                obj = _clean_object(match.group("object"))
                if not obj:
                    continue
                claim = Claim(subject=subject, relation=relation, object=obj.lower())
                if claim.key() not in seen_keys:
                    found.append(claim)
                    seen_keys.add(claim.key())
            except IndexError:
                continue

    # Port numbers — subject is "server" (generic)
    for m in _PORT_PATTERN.finditer(text):
        claim = Claim(subject="server", relation="port", object=m.group(1))
        if claim.key() not in seen_keys:
            found.append(claim)
            seen_keys.add(claim.key())

    # Software versions
    for m in _VERSION_PATTERN.finditer(text):
        subj = m.group(1).lower().strip()
        obj  = m.group(2).lower().strip()
        claim = Claim(subject=subj, relation="version", object=obj)
        if claim.key() not in seen_keys:
            found.append(claim)
            seen_keys.add(claim.key())

    return found
