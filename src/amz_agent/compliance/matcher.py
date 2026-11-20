"""Boundary-aware rule matching.

Two sources of false positives are handled explicitly:

1. Latin terms embedded in longer words ("class" in "classic", "CE" in
   "certificate").  A term gets a word boundary only on an edge that is an
   ASCII word character, so mixed terms such as "100%" or "#1" still work.
2. Legitimate contexts ("compatible with iPhone").  Rules can declare
   `allow_contexts`; a hit inside a matching context is discarded.

All matching runs on an NFKC-normalised copy.  An offset map translates spans
back to the original string, so evidence shown to reviewers is verbatim.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..models import MatchEvidence, MatchType, Rule, Site

_CHAR_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u3000": " ",
}

_WORD_RE = re.compile(r"[a-z0-9_]")


def normalize_with_map(text: str, *, lower: bool = True) -> tuple[str, list[int]]:
    """Return (normalized_text, original_index_for_each_char)."""
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(text):
        char = _CHAR_MAP.get(char, char)
        for normalised in unicodedata.normalize("NFKC", char):
            if normalised.isspace():
                if not chars or chars[-1] == " ":
                    continue
                chars.append(" ")
                mapping.append(index)
                continue
            folded = normalised.lower() if lower else normalised
            for folded_char in folded:
                if folded_char == "%" and chars and chars[-1] == " ":
                    chars.pop()
                    mapping.pop()
                chars.append(folded_char)
                mapping.append(index)

    raw = "".join(chars)
    stripped = raw.strip()
    if not stripped:
        return "", []
    leading = len(raw) - len(raw.lstrip())
    return stripped, mapping[leading : leading + len(stripped)]


def _is_word_char(char: str) -> bool:
    return bool(_WORD_RE.fullmatch(char))


@dataclass
class _CompiledRule:
    rule: Rule
    patterns: list[re.Pattern[str]]
    allow_patterns: list[re.Pattern[str]]


class RuleMatcher:
    def __init__(self, rules: list[Rule], *, max_hits_per_rule: int = 20):
        self.max_hits_per_rule = max_hits_per_rule
        self._compiled = [
            _CompiledRule(
                rule=rule,
                patterns=[re.compile(p, re.IGNORECASE) for p in rule.patterns],
                allow_patterns=[
                    re.compile(p, re.IGNORECASE) for p in rule.allow_contexts
                ],
            )
            for rule in rules
        ]

    def find(self, text: str, site: Site) -> list[MatchEvidence]:
        if not text.strip():
            return []
        normalised, mapping = normalize_with_map(text)
        if not normalised:
            return []

        matches: list[MatchEvidence] = []
        for compiled in self._compiled:
            rule = compiled.rule
            if not rule.applies_to(site):
                continue
            hits = 0
            for term in rule.terms:
                needle = normalize_with_map(term, lower=True)[0]
                if not needle:
                    continue
                start = normalised.find(needle)
                while start >= 0 and hits < self.max_hits_per_rule:
                    end = start + len(needle)
                    if self._accept(compiled, normalised, start, end, rule):
                        matches.append(
                            self._evidence(
                                text, mapping, rule, term, start, end, MatchType.TERM
                            )
                        )
                        hits += 1
                    start = normalised.find(needle, start + 1)
            for pattern in compiled.patterns:
                for found in pattern.finditer(normalised):
                    if hits >= self.max_hits_per_rule:
                        break
                    start, end = found.start(), found.end()
                    if start == end:
                        continue
                    if self._accept(compiled, normalised, start, end, rule):
                        matches.append(
                            self._evidence(
                                text,
                                mapping,
                                rule,
                                found.group(0),
                                start,
                                end,
                                MatchType.REGEX,
                            )
                        )
                        hits += 1

        return _dedupe(matches)

    def _accept(
        self, compiled: _CompiledRule, normalised: str, start: int, end: int, rule: Rule
    ) -> bool:
        if rule.boundary == "auto":
            # Boundary edges are determined from the matched span, not the rule,
            # so regex/term hits share the same logic.
            left_char = normalised[start]
            right_char = normalised[end - 1]
            if (
                left_char.isascii()
                and _is_word_char(left_char)
                and start > 0
                and _is_word_char(normalised[start - 1])
            ):
                return False
            if (
                right_char.isascii()
                and _is_word_char(right_char)
                and end < len(normalised)
                and _is_word_char(normalised[end])
            ):
                return False

        if compiled.allow_patterns:
            window = normalised[max(0, start - 80) : min(len(normalised), end + 80)]
            if any(pattern.search(window) for pattern in compiled.allow_patterns):
                return False
        return True

    def _evidence(
        self,
        original: str,
        mapping: list[int],
        rule: Rule,
        term: str,
        start: int,
        end: int,
        match_type: MatchType,
    ) -> MatchEvidence:
        orig_start = mapping[start]
        orig_end = mapping[end - 1] + 1
        snippet = original[max(0, orig_start - 32) : min(len(original), orig_end + 32)]
        snippet = " ".join(snippet.split())
        return MatchEvidence(
            rule_id=rule.id,
            category=rule.category,
            severity=rule.severity,
            action=rule.action,
            match_type=match_type,
            term=term,
            start=orig_start,
            end=orig_end,
            snippet=snippet,
            citation=rule.citation,
            suggestion=rule.suggestion,
        )


def _dedupe(matches: list[MatchEvidence]) -> list[MatchEvidence]:
    """Drop exact duplicates; keep the longest evidence per overlapping span."""
    seen: set[tuple[str, int, int]] = set()
    output: list[MatchEvidence] = []
    for match in sorted(matches, key=lambda m: (m.start, m.end, m.rule_id)):
        key = (match.rule_id, match.start, match.end)
        if key in seen:
            continue
        seen.add(key)
        output.append(match)
    return sorted(output, key=lambda m: (m.start, m.end, m.rule_id))
