from headroom.proxy.operational_audit import OperationalAudit


def _body(call_id: str, value: str = "x") -> dict:
    return {
        "system": "sys",
        "messages": [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": "Read",
                        "input": {"path": value},
                    }
                ],
            },
        ],
    }


def test_audit_counts_model_substitutions() -> None:
    audit = OperationalAudit()
    audit.record_model_substitution("opus", "sonnet", "cap")
    snapshot = audit.snapshot()
    assert snapshot["model_substitutions_total"] == 1
    assert snapshot["model_substitutions"] == {"opus -> sonnet (cap)": 1}


def test_audit_counts_repeated_signature_once_without_recounting_history() -> None:
    audit = OperationalAudit()
    first = _body("call-1")
    audit.observe_anthropic_tools(first)
    audit.observe_anthropic_tools(first)

    second = _body("call-2")
    audit.observe_anthropic_tools(second)
    snapshot = audit.snapshot()
    assert snapshot["tool_calls_observed"] == 2
    assert snapshot["duplicate_tool_calls"] == 1
    assert snapshot["duplicate_tool_calls_by_tool"] == {"Read": 1}


def test_audit_does_not_flag_different_inputs() -> None:
    audit = OperationalAudit()
    audit.observe_anthropic_tools(_body("call-1", "a"))
    audit.observe_anthropic_tools(_body("call-2", "b"))
    assert audit.snapshot()["duplicate_tool_calls"] == 0


def test_audit_counts_effort_lowerings_and_pins_separately() -> None:
    audit = OperationalAudit()
    audit.record_effort_routing("lowered")
    audit.record_effort_routing("lowered")
    audit.record_effort_routing("pinned")
    snapshot = audit.snapshot()
    assert snapshot["effort_lowerings"] == 2
    assert snapshot["effort_pins"] == 1


def test_audit_ignores_unknown_effort_routing_actions() -> None:
    audit = OperationalAudit()
    audit.record_effort_routing("unchanged")
    snapshot = audit.snapshot()
    assert snapshot["effort_lowerings"] == 0
    assert snapshot["effort_pins"] == 0
