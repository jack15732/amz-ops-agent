from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..models import Rule, Site


class RuleConfigError(ValueError):
    """Raised when a compliance YAML file is malformed."""


@dataclass
class RuleSet:
    rules: list[Rule] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def for_site(self, site: Site) -> list[Rule]:
        return [rule for rule in self.rules if rule.applies_to(site)]

    def get(self, rule_id: str) -> Rule | None:
        return next((rule for rule in self.rules if rule.id == rule_id), None)

    @property
    def version_fingerprint(self) -> str:
        payload = "|".join(
            f"{site}:{self.versions.get(site, '?')}" for site in sorted(self.versions)
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def version_for(self, site: Site) -> str:
        return f"{site.value}:{self.versions.get(site.value, 'unknown')}"


def _compile_check(rule: Rule, source: Path) -> None:
    for pattern in rule.patterns:
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:  # pragma: no cover - defensive
            raise RuleConfigError(f"{source}: rule {rule.id} bad regex {pattern!r}: {exc}") from exc


def load_rules(rule_dir: str | Path) -> RuleSet:
    path = Path(rule_dir)
    if not path.exists():
        raise RuleConfigError(f"rule dir not found: {path}")

    rules: list[Rule] = []
    versions: dict[str, str] = {}
    sources: dict[str, str] = {}
    seen: dict[str, str] = {}

    for file in sorted(path.glob("*.yaml")):
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise RuleConfigError(f"{file}: invalid YAML: {exc}") from exc

        meta = raw.get("meta", {})
        site_name = str(meta.get("site", "")).upper()
        version = str(meta.get("version", "v1"))
        if site_name not in {s.value for s in Site}:
            raise RuleConfigError(f"{file}: meta.site must be one of US/EU/JP, got {site_name!r}")
        site = Site(site_name)
        versions[site.value] = version
        sources[site.value] = str(meta.get("source", file.name))

        entries = raw.get("rules") or []
        if not isinstance(entries, list) or not entries:
            raise RuleConfigError(f"{file}: `rules` must be a non-empty list")

        for entry in entries:
            entry = dict(entry)
            # Site comes from file metadata unless a rule explicitly overrides it.
            entry.setdefault("sites", [site.value])
            entry.setdefault("version", version)
            try:
                rule = Rule.model_validate(entry)
            except ValidationError as exc:
                raise RuleConfigError(f"{file}: invalid rule {entry.get('id')}: {exc}") from exc
            if rule.id in seen:
                raise RuleConfigError(f"duplicated rule id {rule.id} in {file} and {seen[rule.id]}")
            seen[rule.id] = str(file)
            _compile_check(rule, file)
            rules.append(rule)

    if not rules:
        raise RuleConfigError(f"no rules loaded from {path}")

    return RuleSet(rules=rules, versions=versions, sources=sources)
