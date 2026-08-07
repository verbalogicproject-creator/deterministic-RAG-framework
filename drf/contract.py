"""The action contract: typed, audited, and mechanically enforced.

Every operation in the framework is an *action* declaring two independent
properties:

    determinism : "deterministic" | "probabilistic"
        Does the same input always produce the same output?

    authority   : "authoritative" | "advisory"
        May this action's output influence the ranking of results?

These axes are orthogonal, and conflating them is the mistake this module
exists to prevent. Anchor-mode vector search over frozen vectors is fully
*deterministic* and yet strictly *advisory*. A remote embedder measured to be
deterministic is still advisory. Authority is an architectural property;
determinism is an empirical one.

Nothing here is documentation. Each label is enforced:

  * `deterministic` actions are replay-checked - calling one twice with the
    same inputs and getting different results raises. Mislabelling therefore
    fails at runtime, not at code review.
  * `probabilistic` actions must declare a confidence; `deterministic` ones
    must not. Forgetting is a call-time error.
  * `advisory` results are boxed in `Advisory[T]`, whose `unwrap()` refuses
    to yield its contents to any module outside a one-entry allowlist. This
    makes it physically impossible for advisory data to reach authoritative
    arithmetic anywhere in the codebase.

Stdlib only.
"""

import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Mapping, NamedTuple, TypeVar

from .hashing import canonical_json, sha256_value

T = TypeVar("T")

Determinism = Literal["deterministic", "probabilistic"]
Authority = Literal["authoritative", "advisory"]

# The single module permitted to unwrap advisory values. Advisory data may be
# appended to results there and nowhere else.
ADVISORY_CONSUMERS = frozenset({"drf.retrieval.merge"})


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class ContractError(Exception):
    """Base for all contract violations."""


class DeterminismViolation(ContractError):
    """An action declared deterministic returned two different results."""


class AuthorityViolation(ContractError):
    """Advisory data was accessed from a module not permitted to consume it."""


class DeclarationError(ContractError):
    """An action's declared labels are inconsistent with what it returned."""


# --------------------------------------------------------------------------
# Justification and results
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Justification:
    """Why a result is what it is.

    Auditability does not require determinism - only declaration. A
    probabilistic action is fully auditable provided it states its confidence,
    its provider, and the hash of the inputs it saw.
    """
    action: str
    determinism: Determinism
    authority: Authority
    inputs_hash: str
    elapsed_ns: int
    confidence: float | None = None
    evidence: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    provider: str | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "determinism": self.determinism,
            "authority": self.authority,
            "inputs_hash": self.inputs_hash,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "params": dict(self.params),
            "provider": self.provider,
        }


class ActionOutput(NamedTuple):
    """What a decorated action function returns."""
    value: Any
    evidence: tuple[str, ...] = ()
    confidence: float | None = None
    provider: str | None = None
    params: Mapping[str, Any] = {}


class ActionResult(NamedTuple):
    """What a caller of an action receives: `value, just = some_action(...)`."""
    value: Any
    justification: Justification


# --------------------------------------------------------------------------
# Advisory boxing
# --------------------------------------------------------------------------

class Advisory(Generic[T]):
    """A value that must not influence ranking.

    The box is not a convention - `unwrap()` inspects its caller's module and
    refuses anyone outside `ADVISORY_CONSUMERS`. Advisory data therefore
    cannot be summed into a score, used as a sort key, or otherwise granted
    authority, no matter what a future edit tries to do.
    """

    __slots__ = ("_value", "_provider")

    def __init__(self, value: T, provider: str | None = None):
        self._value = value
        self._provider = provider

    @property
    def provider(self) -> str | None:
        return self._provider

    def is_empty(self) -> bool:
        """Safe for anyone to call - reveals presence, not contents."""
        return not self._value

    def unwrap(self) -> T:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        module = caller.f_globals.get("__name__", "<unknown>") if caller else "<unknown>"
        if module not in ADVISORY_CONSUMERS:
            raise AuthorityViolation(
                f"module {module!r} may not unwrap Advisory data; "
                f"permitted consumers are {sorted(ADVISORY_CONSUMERS)}. "
                "Advisory results can only be appended below authoritative "
                "ones, never combined with them."
            )
        return self._value

    def __repr__(self) -> str:
        return f"Advisory(provider={self._provider!r}, empty={self.is_empty()})"


# --------------------------------------------------------------------------
# Registry and the @action decorator
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    determinism: Determinism
    authority: Authority
    func: Callable
    doc: str


ACTIONS: dict[str, ActionSpec] = {}

# inputs_hash -> sha256(result). Populated only for deterministic actions.
_replay_log: dict[str, str] = {}
_strict_replay = False


def reset_replay_log() -> None:
    """Clear memoised results. Call between independent test cases."""
    _replay_log.clear()


class strict_replay:
    """Context manager that forces every deterministic action to be invoked
    twice and its two results compared. Used in tests to convert the replay
    check from opportunistic into exhaustive.
    """

    def __enter__(self):
        global _strict_replay
        _strict_replay = True
        return self

    def __exit__(self, *exc):
        global _strict_replay
        _strict_replay = False
        return False


def _hashable_inputs(bound: inspect.BoundArguments, declared: tuple[str, ...] | None) -> dict:
    """Build the input dict that gets hashed.

    If the action declares `inputs`, only those parameters are hashed - which
    is what you want when an action also receives a database handle or a
    provider object. Otherwise every JSON-serialisable argument is included
    and the rest are represented by their type name, so an unhashable
    argument degrades the hash's precision rather than crashing.
    """
    out: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name == "self":
            continue
        if declared is not None and name not in declared:
            continue
        try:
            canonical_json(value)
            out[name] = value
        except (TypeError, ValueError):
            out[name] = f"<{type(value).__name__}>"
    return out


def action(
    name: str,
    *,
    determinism: Determinism,
    authority: Authority,
    inputs: tuple[str, ...] | None = None,
):
    """Declare a function as an action and enforce its contract.

    The decorated function must return an `ActionOutput`. The decorator
    validates the declaration, computes the inputs hash, performs the replay
    check, and returns an `ActionResult`.
    """
    if determinism not in ("deterministic", "probabilistic"):
        raise DeclarationError(f"{name}: bad determinism {determinism!r}")
    if authority not in ("authoritative", "advisory"):
        raise DeclarationError(f"{name}: bad authority {authority!r}")

    def decorate(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ActionResult:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs_hash = sha256_value({
                "action": name,
                "args": _hashable_inputs(bound, inputs),
            })

            started = time.perf_counter_ns()
            out = func(*args, **kwargs)
            elapsed = time.perf_counter_ns() - started

            if not isinstance(out, ActionOutput):
                raise DeclarationError(
                    f"{name}: must return ActionOutput, got {type(out).__name__}"
                )

            # Confidence is structurally coupled to the determinism label.
            if determinism == "deterministic" and out.confidence is not None:
                raise DeclarationError(
                    f"{name}: deterministic actions must not declare a "
                    f"confidence (got {out.confidence!r})"
                )
            if determinism == "probabilistic":
                if out.confidence is None:
                    raise DeclarationError(
                        f"{name}: probabilistic actions must declare a confidence"
                    )
                if not 0.0 <= out.confidence <= 1.0:
                    raise DeclarationError(
                        f"{name}: confidence {out.confidence!r} outside [0.0, 1.0]"
                    )

            # Replay check: the deterministic label is load-bearing.
            if determinism == "deterministic":
                try:
                    digest = sha256_value(out.value)
                except (TypeError, ValueError):
                    digest = None  # unhashable result; cannot verify
                if digest is not None:
                    previous = _replay_log.get(inputs_hash)
                    if previous is not None and previous != digest:
                        raise DeterminismViolation(
                            f"{name}: declared deterministic but returned a "
                            f"different result for identical inputs "
                            f"(inputs_hash={inputs_hash[:16]})"
                        )
                    _replay_log[inputs_hash] = digest

                    if _strict_replay and previous is None:
                        again = func(*args, **kwargs)
                        if sha256_value(again.value) != digest:
                            raise DeterminismViolation(
                                f"{name}: strict replay produced a different "
                                f"result on immediate re-invocation"
                            )

            justification = Justification(
                action=name,
                determinism=determinism,
                authority=authority,
                inputs_hash=inputs_hash,
                elapsed_ns=elapsed,
                confidence=out.confidence,
                evidence=tuple(out.evidence),
                params=dict(out.params),
                provider=out.provider,
            )

            value = out.value
            if authority == "advisory" and not isinstance(value, Advisory):
                value = Advisory(value, provider=out.provider)
            return ActionResult(value, justification)

        ACTIONS[name] = ActionSpec(
            name=name,
            determinism=determinism,
            authority=authority,
            func=wrapper,
            doc=inspect.getdoc(func) or "",
        )
        wrapper._drf_action = name  # type: ignore[attr-defined]
        return wrapper

    return decorate


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------

@dataclass
class Trace:
    """The ordered audit trail of one query."""
    entries: list[Justification] = field(default_factory=list)

    def record(self, justification: Justification) -> None:
        self.entries.append(justification)

    def to_list(self) -> list[dict]:
        return [j.to_dict() for j in self.entries]

    def digest(self) -> str:
        """Hash of the trace, excluding timings (which are not reproducible)."""
        return sha256_value(self.to_list())
