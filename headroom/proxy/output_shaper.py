"""Output token shaping for proxied Anthropic and OpenAI Responses requests.

Headroom's transforms compress what goes INTO the model. This module is the
first request-side lever on what comes OUT of it. The proxy never generates
output tokens, so every lever here works by reshaping the request:

1. Verbosity steering — a deterministic instruction block appended to the
   TAIL of the system prompt (after any ``cache_control`` breakpoint, so the
   provider prefix cache is preserved). Five levels, from "no ceremony" to
   full caveman.

2. Effort routing — agentic loops are mostly mechanical continuations (the
   last message is a clean tool_result: a file read, a passing test). Thinking
   bills as output tokens, and harnesses like Claude Code pin
   ``output_config.effort`` at ``xhigh`` for every turn. On turns classified
   as mechanical we lower an explicitly-present effort; on errors or new user
   asks we leave it alone. A request carrying a ``cache_control`` breakpoint
   pins the effort instead of lowering it (measured 2026-07-19: flipping
   ``output_config.effort`` busts the Anthropic prompt cache).

Safety rules (each prevents a concrete failure mode):
- Never INJECT ``output_config.effort`` where the client didn't send it —
  models without effort support 400 on it. Lowering an existing value is
  always valid.
- Never toggle ``thinking.type`` — disabling thinking while history carries
  thinking blocks 400s on some models, and the toggle busts the messages
  cache tier.
- Steering text is byte-stable per level and applied idempotently, so
  repeated requests keep an identical prefix.

Turn classification is purely structural (block types, roles, ``is_error``
flags) — no content regexes or keyword patterns.

The same two levers exist for the OpenAI Responses format (Codex et al.):
:func:`classify_responses_turn` reads the ``input`` item list,
:func:`apply_responses_verbosity_steering` appends the byte-stable steering
block to the tail of the ``instructions`` string, and
:func:`route_responses_effort` lowers an explicitly-present
``reasoning.effort`` on mechanical continuations. :func:`shape_responses_request`
is the Responses-format counterpart of :func:`shape_request`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from headroom.proxy import runtime_env
from headroom.proxy.output_effort_policy import (
    EFFORT_RANK as _EFFORT_RANK,
)
from headroom.proxy.output_effort_policy import (
    can_create_openai_text_verbosity,
    lower_effort_value,
    lower_text_verbosity_value,
)
from headroom.proxy.output_steering import (
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    replace_or_append_steering_block,
    steering_text,
)
from headroom.proxy.output_turn_policy import (
    TurnKind,
    classify_openai_responses_input,
    classify_turn,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OutputShaperSettings",
    "ShapeResult",
    "TurnKind",
    "apply_openai_responses_verbosity_steering",
    "apply_verbosity_steering",
    "classify_openai_responses_input",
    "classify_turn",
    "request_uses_prompt_caching",
    "resolve_verbosity_level",
    "route_effort",
    "route_openai_reasoning_effort",
    "route_openai_text_verbosity",
    "shape_openai_responses_request",
    "shape_request",
    "steering_text",
]

_replace_or_append_steering_block = replace_or_append_steering_block


@dataclass(frozen=True)
class OutputShaperSettings:
    """Runtime settings, resolved once per request from the environment.

    Env-driven (like HEADROOM_INTERCEPT_ENABLED) so the proxy picks it up
    without config plumbing through the server. Off by default.
    """

    enabled: bool = False
    verbosity_level: int = 2
    effort_router_enabled: bool = True
    mechanical_effort: str = "low"

    @classmethod
    def from_env(cls) -> OutputShaperSettings:
        enabled = runtime_env.getenv("HEADROOM_OUTPUT_SHAPER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            level = int(runtime_env.getenv("HEADROOM_VERBOSITY_LEVEL", "2"))
        except ValueError:
            level = 2
        level = max(0, min(4, level))
        router = runtime_env.getenv("HEADROOM_EFFORT_ROUTER", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        mech = runtime_env.getenv("HEADROOM_MECHANICAL_EFFORT", "low")
        if mech not in _EFFORT_RANK:
            mech = "low"
        return cls(
            enabled=enabled,
            verbosity_level=level,
            effort_router_enabled=router,
            mechanical_effort=mech,
        )


def resolve_verbosity_level(settings: OutputShaperSettings) -> tuple[int, str]:
    """Resolve the live verbosity level and its source.

    Precedence:
      1. ``HEADROOM_VERBOSITY_LEVEL`` set explicitly → manual override.
      2. AIMD controller state (when ``HEADROOM_VERBOSITY_AUTOTUNE`` is on).
      3. Learned ``verbosity.json`` from ``learn --verbosity``.
      4. The settings default.

    Returns ``(level, source)``. Kept separate from :func:`shape_request` so the
    body-mutating core stays a pure function of an explicit level.
    """
    if runtime_env.getenv("HEADROOM_VERBOSITY_LEVEL"):
        return settings.verbosity_level, "env"

    try:
        from ..paths import workspace_dir

        ws = workspace_dir()
    except Exception:
        return settings.verbosity_level, "default"

    autotune = runtime_env.getenv("HEADROOM_VERBOSITY_AUTOTUNE", "").lower() in ("1", "true", "yes")
    if autotune:
        ctrl_path = ws / "verbosity_controller.json"
        if ctrl_path.exists():
            try:
                import json as _json

                level = int(
                    _json.loads(ctrl_path.read_text()).get("level", settings.verbosity_level)
                )
                return max(0, min(4, level)), "controller"
            except (OSError, ValueError):
                pass

    prof_path = ws / "verbosity.json"
    if prof_path.exists():
        try:
            import json as _json

            level = int(_json.loads(prof_path.read_text()).get("verbosity_level", -1))
            if 0 <= level <= 4:
                return level, "learned"
        except (OSError, ValueError):
            pass

    return settings.verbosity_level, "default"


@dataclass
class ShapeResult:
    """What the shaper did to a request body."""

    changed: bool = False
    labels: list[str] | None = None
    effort_decision: str | None = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = []


def request_uses_prompt_caching(body: dict[str, Any]) -> bool:
    """Whether any cache_control breakpoint is present in the request."""
    if not isinstance(body, dict):
        return False

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and "cache_control" in tool:
                return True

    system = body.get("system")
    if isinstance(system, dict) and "cache_control" in system:
        return True
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                return True

    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if "cache_control" in block:
                    return True
                # Breakpoints can sit one level deeper, on a sub-block of a
                # tool_result's own content list. Missing one means mutating
                # effort on a cached request, a full-prefix rewrite.
                inner = block.get("content")
                if isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and "cache_control" in sub:
                            return True

    return False


def route_effort(
    body: dict[str, Any],
    kind: TurnKind,
    settings: OutputShaperSettings,
) -> tuple[list[str], str | None]:
    """Lower effort spend on mechanical continuations.

    Returns (labels, decision). labels holds a mutation label when the body
    was edited (empty list = untouched). decision is "lowered", "pinned", or
    None.
    """
    if kind is not TurnKind.MECHANICAL_CONTINUATION:
        return [], None

    # Modern lever: output_config.effort. Only lower a value the client
    # explicitly sent — presence proves the target model accepts the param.
    output_config = body.get("output_config")
    if not isinstance(output_config, dict):
        return [], None

    effort = output_config.get("effort")
    lowered = lower_effort_value(effort, settings.mechanical_effort)
    if lowered is None:
        return [], None

    # Flipping output_config.effort busts the Anthropic prompt cache (measured
    # 2026-07-19, two full-prefix rewrites per flip), so a cached request
    # keeps its current effort instead.
    if request_uses_prompt_caching(body):
        return [], "pinned"

    output_config["effort"] = lowered
    return [f"output_shaper:effort:{effort}->{lowered}"], "lowered"


def route_openai_reasoning_effort(
    body: dict[str, Any],
    kind: TurnKind,
    settings: OutputShaperSettings,
) -> list[str]:
    """Lower explicitly-present OpenAI reasoning effort on mechanical turns."""
    if kind is not TurnKind.MECHANICAL_CONTINUATION:
        return []

    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return []
    effort = reasoning.get("effort")
    target = settings.mechanical_effort
    lowered = lower_effort_value(effort, target)
    if lowered is not None:
        reasoning["effort"] = lowered
        return [f"output_shaper:reasoning_effort:{effort}->{lowered}"]
    return []


def route_openai_text_verbosity(body: dict[str, Any]) -> list[str]:
    """Set or lower OpenAI ``text.verbosity`` conservatively."""
    text_config = body.get("text")
    can_create = can_create_openai_text_verbosity(body.get("model"))
    if text_config is None:
        if not can_create:
            return []
        body["text"] = {"verbosity": "low"}
        return ["output_shaper:text_verbosity:unset->low"]
    if not isinstance(text_config, dict):
        return []

    verbosity = text_config.get("verbosity")
    if verbosity is None:
        if not can_create:
            return []
        text_config["verbosity"] = "low"
        return ["output_shaper:text_verbosity:unset->low"]
    lowered = lower_text_verbosity_value(verbosity)
    if lowered is not None:
        text_config["verbosity"] = lowered
        return [f"output_shaper:text_verbosity:{verbosity}->{lowered}"]
    return []


def shape_openai_responses_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply OpenAI Responses output-shaping levers in place."""
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_openai_responses_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    kind = classify_openai_responses_input(body.get("input"))
    if settings.effort_router_enabled:
        labels = route_openai_reasoning_effort(body, kind, settings)
        if labels:
            result.changed = True
            result.labels.extend(labels)
            logger.debug("OpenAIOutputShaper: turn=%s mutations=%s", kind.value, labels)

    labels = route_openai_text_verbosity(body)
    if labels:
        result.changed = True
        result.labels.extend(labels)

    return result


def shape_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply all output-shaping levers to an Anthropic request body in place.

    ``level_override`` supersedes ``settings.verbosity_level`` when given — the
    handler passes the level resolved by :func:`resolve_verbosity_level` (learned
    profile / controller / env) so the body-mutating core stays level-agnostic.
    """
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees this

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    if settings.effort_router_enabled:
        kind = classify_turn(body.get("messages", []))
        labels, effort_decision = route_effort(body, kind, settings)
        if labels:
            result.changed = True
            result.labels.extend(labels)
        result.effort_decision = effort_decision
        logger.debug(
            "OutputShaper: turn=%s mutations=%s decision=%s",
            kind.value,
            labels,
            effort_decision,
        )

    return result


# ---------------------------------------------------------------------------
# OpenAI Responses format (Codex, /v1/responses HTTP + WebSocket)
# ---------------------------------------------------------------------------

# Responses ``reasoning.effort`` uses "minimal" as its floor (Anthropic's
# ``output_config.effort`` does not), so it gets its own rank table.
_RESPONSES_EFFORT_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}

# Trailing ``input`` item types that represent tool output coming back to the
# model — the Responses counterpart of an Anthropic ``tool_result`` block.
_RESPONSES_TOOL_OUTPUT_TYPES = frozenset(
    {
        "function_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "computer_call_output",
    }
)


def _responses_tool_output_is_error(item: dict[str, Any]) -> bool:
    """Structural error sniff on a Responses tool-output item.

    The Responses format has no ``is_error`` flag, but agent harnesses encode
    failure structurally in the ``output`` payload: a JSON object with a
    nonzero ``exit_code``, ``success: false``, or a truthy ``error`` field.
    Only those JSON fields are inspected — never prose content.
    """
    output = item.get("output")
    data: Any = output
    if isinstance(output, str):
        stripped = output.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            return False
    if not isinstance(data, dict):
        return False
    # Direct fields, plus the common {"output": ..., "metadata": {...}} nesting.
    scopes: list[dict[str, Any]] = [data]
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        scopes.append(metadata)
    for scope in scopes:
        exit_code = scope.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        if scope.get("success") is False:
            return True
        if scope.get("error"):
            return True
    return False


def classify_responses_turn(input_data: Any) -> TurnKind:
    """Classify a Responses request's turn from its ``input`` field.

    Mirrors :func:`classify_turn` semantics on the Responses item list: the
    trailing run of tool-output items decides the turn. A trailing user
    message is a new ask; tool outputs are mechanical unless any carries a
    structural error marker. Purely structural — item types and JSON fields,
    no content regexes.
    """
    if isinstance(input_data, str):
        return TurnKind.NEW_USER_ASK if input_data.strip() else TurnKind.UNKNOWN
    if not isinstance(input_data, list) or not input_data:
        return TurnKind.UNKNOWN

    saw_tool_output = False
    saw_error = False
    for item in reversed(input_data):
        if not isinstance(item, dict):
            return TurnKind.UNKNOWN
        itype = item.get("type")
        if itype in _RESPONSES_TOOL_OUTPUT_TYPES:
            saw_tool_output = True
            if _responses_tool_output_is_error(item):
                saw_error = True
            continue
        # First non-tool-output item ends the trailing run.
        if saw_tool_output:
            break
        if itype == "message" or (itype is None and "role" in item):
            role = item.get("role")
            if role == "user":
                return TurnKind.NEW_USER_ASK
            return TurnKind.UNKNOWN
        return TurnKind.UNKNOWN

    if saw_error:
        return TurnKind.ERROR_CONTINUATION
    if saw_tool_output:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN


def apply_responses_verbosity_steering(body: dict[str, Any], level: int) -> bool:
    """Append the steering block to the tail of ``instructions``.

    ``instructions`` is the Responses cache hot zone: the appended block is
    byte-stable per level, so within a conversation every shaped turn sends
    identical instructions bytes and the provider prefix cache stays hot
    after the first shaped turn (the same contract as the Anthropic
    system-tail append).
    """
    return apply_openai_responses_verbosity_steering(body, level)


def route_responses_effort(
    body: dict[str, Any],
    kind: TurnKind,
    settings: OutputShaperSettings,
) -> list[str]:
    """Lower ``reasoning.effort`` on mechanical continuations.

    Only lowers a value the client explicitly sent — presence proves the
    target model accepts the parameter. Never injects ``reasoning`` where
    absent, and never touches new asks or error continuations.
    """
    if kind is not TurnKind.MECHANICAL_CONTINUATION:
        return []

    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return []
    effort = reasoning.get("effort")
    target = settings.mechanical_effort
    if (
        isinstance(effort, str)
        and effort in _RESPONSES_EFFORT_RANK
        and target in _RESPONSES_EFFORT_RANK
        and _RESPONSES_EFFORT_RANK[effort] > _RESPONSES_EFFORT_RANK[target]
    ):
        reasoning["effort"] = target
        return [f"output_shaper:effort:{effort}->{target}"]
    return []


def shape_responses_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply all output-shaping levers to a Responses payload in place.

    The Responses counterpart of :func:`shape_request`: same settings, same
    labels, same level-resolution contract.
    """
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees this

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_responses_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    if settings.effort_router_enabled:
        kind = classify_responses_turn(body.get("input"))
        labels = route_responses_effort(body, kind, settings)
        if labels:
            result.changed = True
            result.labels.extend(labels)
        logger.debug("OutputShaper(responses): turn=%s mutations=%s", kind.value, labels)

    return result
