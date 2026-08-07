"""Configuration: validated against a schema, hashed by what actually matters.

A configuration's `content_hash` covers **only** the settings flagged
`affects_ranking`. Two configurations that differ in `display.k` are the same
computation and hash identically; two that differ in `ranking.b` are different
computations and cannot. That makes the hash a usable identity for "which
ranking produced this result" - which is what a benchmark, a cache key, or a
reproduction report actually needs.

Neural settings are deliberately **not** ranking-affecting. Provider choice
changes only the advisory tail, which merge appends below D and can never
reorder. This module therefore restates the architecture's central guarantee
in a second, independently checkable place: if changing `neural.provider` ever
altered a configuration's content hash, either merge or this schema would be
wrong, and `tests/test_config.py` checks the claim against real query output
rather than trusting the flag.

Ported from `config_manager.py:314-397` in the recovered project - genuinely
good code, the best in that repository. Fixed on port:

  * `_diff_dicts` iterated dict insertion order, so the *order* of the
    returned differences was not reproducible. Now sorted.
  * `diff` returned `{'error': ...}` for a missing config; a caller who did
    not inspect the dict would treat "not found" as "no differences". Raises.
  * `set()` accepted any key and any value. Now validated against the schema,
    so a typo is an error instead of a silently ignored setting.

Stdlib only.
"""

import copy
import json
from pathlib import Path
from typing import Any

from ..hashing import canonical_json, sha256_value

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "spec" / "config_schema.json"


class ConfigError(Exception):
    """A configuration was invalid. Never returned as a value - always raised."""


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)["settings"]


SCHEMA = load_schema()

# Python types accepted for each schema type. bool is checked before int
# because `isinstance(True, int)` is True in Python - without the ordering,
# `display.explain = True` would satisfy an "integer" field and a genuine
# type error would pass validation.
_TYPES: dict[str, tuple] = {
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "string": (str,),
}


def defaults() -> dict:
    """A fresh settings dict from the schema. Deep-copied per call."""
    return {name: copy.deepcopy(spec["default"]) for name, spec in SCHEMA.items()}


def ranking_keys() -> list[str]:
    """Settings that can influence ranking, sorted."""
    return sorted(k for k, s in SCHEMA.items() if s.get("affects_ranking"))


def display_keys() -> list[str]:
    return sorted(k for k, s in SCHEMA.items() if not s.get("affects_ranking"))


def validate_one(key: str, value: Any) -> None:
    """Validate a single setting, raising on anything the schema forbids."""
    spec = SCHEMA.get(key)
    if spec is None:
        raise ConfigError(
            f"unknown setting {key!r}; known settings are {sorted(SCHEMA)}"
        )

    expected = spec["type"]
    if expected != "boolean" and isinstance(value, bool):
        raise ConfigError(
            f"{key}: expected {expected}, got bool. Python treats bool as a "
            "subtype of int, so this is checked explicitly rather than left "
            "to isinstance."
        )
    if not isinstance(value, _TYPES[expected]):
        raise ConfigError(
            f"{key}: expected {expected}, got {type(value).__name__}"
        )

    if "enum" in spec and value not in spec["enum"]:
        raise ConfigError(f"{key}: {value!r} not in {spec['enum']}")
    if "minimum" in spec and value < spec["minimum"]:
        raise ConfigError(f"{key}: {value} below minimum {spec['minimum']}")
    if "maximum" in spec and value > spec["maximum"]:
        raise ConfigError(f"{key}: {value} above maximum {spec['maximum']}")


class Config:
    """A validated settings bundle with a ranking-scoped content hash."""

    def __init__(self, settings: dict | None = None):
        self._settings = defaults()
        for key, value in (settings or {}).items():
            self.set(key, value)

    def get(self, key: str) -> Any:
        if key not in self._settings:
            raise ConfigError(f"unknown setting {key!r}")
        return self._settings[key]

    def set(self, key: str, value: Any) -> None:
        validate_one(key, value)
        self._settings[key] = value

    def as_dict(self) -> dict:
        return copy.deepcopy(self._settings)

    def ranking_settings(self) -> dict:
        """Only the settings that can change a result's order."""
        return {k: self._settings[k] for k in ranking_keys()}

    def content_hash(self) -> str:
        """Identity of the *computation*, not of the presentation.

        Deliberately excludes display and neural settings. Neural is excluded
        on the same architectural grounds as display: it cannot influence
        ranking, so a configuration that differs only in provider produces the
        same authoritative order and should be recognisably the same thing.
        """
        return sha256_value({
            "schema_version": 1,
            "ranking": self.ranking_settings(),
        })

    def action_kwargs(self) -> dict:
        """Ranking settings keyed by the action parameter they bind to.

        The binding is read from the schema rather than restated here, so a
        renamed action parameter surfaces as a schema mismatch in the tests
        rather than as a silently ignored setting.
        """
        return {
            SCHEMA[key]["action_input"]: self._settings[key]
            for key in ranking_keys()
        }

    def diff(self, other: "Config") -> list[dict]:
        return diff_dicts(self.as_dict(), other.as_dict())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Config)
            and canonical_json(self._settings) == canonical_json(other._settings)
        )


def diff_dicts(first: dict, second: dict, path: str = "") -> list[dict]:
    """Recursive structural diff, ported and made order-stable.

    Keys are iterated in sorted order and the result is sorted by path, so the
    same pair of configurations always produces the same list. The original
    iterated insertion order, which made the diff's *content* correct but its
    *order* dependent on how each dict happened to be built.
    """
    differences: list[dict] = []

    for key in sorted(set(first) | set(second)):
        current = f"{path}.{key}" if path else key
        in_first, in_second = key in first, key in second

        if in_first and not in_second:
            differences.append({
                "path": current, "in_first": True, "in_second": False,
                "value1": first[key], "value2": None,
            })
        elif in_second and not in_first:
            differences.append({
                "path": current, "in_first": False, "in_second": True,
                "value1": None, "value2": second[key],
            })
        elif isinstance(first[key], dict) and isinstance(second[key], dict):
            differences.extend(diff_dicts(first[key], second[key], current))
        elif first[key] != second[key]:
            differences.append({
                "path": current, "in_first": True, "in_second": True,
                "value1": first[key], "value2": second[key],
            })

    return sorted(differences, key=lambda d: d["path"])


class Snapshot:
    """A configuration bound to the index it was validated against.

    A configuration alone does not identify a result: the same settings over a
    different index produce different output. A snapshot records both hashes,
    so "reproduce this" names a complete computation rather than half of one.
    """

    def __init__(self, *, name: str, config: Config, manifest_hash: str):
        if not manifest_hash:
            raise ConfigError(
                "a snapshot must bind to an index manifest hash; a "
                "configuration alone does not identify a result"
            )
        self.name = name
        self.config = config
        self.manifest_hash = manifest_hash

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "settings": self.config.as_dict(),
            "config_hash": self.config.content_hash(),
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        snapshot = cls(
            name=data["name"],
            config=Config(data["settings"]),
            manifest_hash=data["manifest_hash"],
        )
        stored = data.get("config_hash")
        if stored and stored != snapshot.config.content_hash():
            raise ConfigError(
                f"snapshot {data['name']!r}: stored config_hash does not match "
                "its settings; the file has been altered"
            )
        return snapshot

    def identity(self) -> str:
        """One hash naming the whole computation: settings plus index."""
        return sha256_value({
            "config": self.config.content_hash(),
            "manifest": self.manifest_hash,
        })
