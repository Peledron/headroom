from __future__ import annotations

from headroom.transforms.text_optical import TextOpticalCompressor, extract_optical_facts


def _prose() -> str:
    paragraph = (
        "The cache keeps an immutable prefix while the current observation remains live. "
        "A rebase is allowed only after measured savings exceed the provider cache cost. "
        "Each rendered page carries a stable digest so future turns can reuse identical bytes.\n"
    )
    return paragraph * 120


def test_optical_render_is_deterministic_and_cacheable(tmp_path) -> None:
    compressor = TextOpticalCompressor(tmp_path, count_tokens=lambda text: len(text) * 10)
    first = compressor.compress(
        _prose(),
        provider="openai",
        model="gpt-4o-mini",
        immutable=True,
        age_turns=3,
        detail="low",
        generation=2,
    )
    second = compressor.compress(
        _prose(),
        provider="openai",
        model="gpt-4o-mini",
        immutable=True,
        age_turns=3,
        detail="low",
        generation=2,
        warm_prefix=True,
    )
    assert first is not None
    assert second is not None
    assert first.manifest == second.manifest
    assert [page.sha256 for page in first.pages] == [page.sha256 for page in second.pages]
    assert first.estimated_tokens_saved > 0


def test_warm_prefix_never_renders_a_new_optical_generation(tmp_path) -> None:
    compressor = TextOpticalCompressor(tmp_path)
    result = compressor.compress(
        _prose(),
        provider="openai",
        model="gpt-4o-mini",
        immutable=True,
        age_turns=3,
        generation=9,
        warm_prefix=True,
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_exact_text_categories_and_code_heavy_content_are_rejected(tmp_path) -> None:
    compressor = TextOpticalCompressor(tmp_path)
    code = "def exact_identifier_name():\n    return 42\n" * 400
    assert compressor.eligible(code, immutable=True, age_turns=3, content_kind="code")[0] is False
    assert compressor.eligible(code, immutable=True, age_turns=3, content_kind="prose")[0] is False
    assert compressor.eligible(_prose(), immutable=False, age_turns=3)[0] is False


def test_provider_specific_image_token_estimators() -> None:
    openai_low = TextOpticalCompressor.estimate_image_tokens(
        provider="openai", model="gpt-4o-mini", width=1024, height=1024, detail="low"
    )
    openai_patch = TextOpticalCompressor.estimate_image_tokens(
        provider="openai", model="gpt-4.1-mini", width=1024, height=1024, detail="high"
    )
    openai_high = TextOpticalCompressor.estimate_image_tokens(
        provider="openai", model="gpt-4o-mini", width=1024, height=1024, detail="high"
    )
    anthropic = TextOpticalCompressor.estimate_image_tokens(
        provider="anthropic", model="claude-sonnet", width=1024, height=1024, detail="high"
    )
    anthropic_standard_resize = TextOpticalCompressor.estimate_image_tokens(
        provider="anthropic", model="claude-haiku-4-5", width=1920, height=1080, detail="high"
    )
    anthropic_high_resolution = TextOpticalCompressor.estimate_image_tokens(
        provider="anthropic", model="claude-opus-4-8", width=1920, height=1080, detail="high"
    )
    openai_gpt56 = TextOpticalCompressor.estimate_image_tokens(
        provider="openai", model="gpt-5.6", width=1024, height=1024, detail="original"
    )
    assert openai_low == 2_833
    assert openai_high == 25_501
    assert openai_patch == 1_659
    assert anthropic == 1_369
    assert anthropic_standard_resize == 1_560
    assert anthropic_high_resolution == 2_691
    assert openai_gpt56 == 1_024
    assert anthropic != openai_patch


def test_unknown_openai_cost_profile_fails_closed(tmp_path) -> None:
    assert (
        TextOpticalCompressor(tmp_path).compress(
            _prose(),
            provider="openai",
            model="unpublished-vision-model",
            immutable=True,
            age_turns=3,
        )
        is None
    )


def test_provider_adapters_include_manifest_and_page_hashes(tmp_path) -> None:
    result = TextOpticalCompressor(tmp_path, count_tokens=lambda text: len(text) * 10).compress(
        _prose(),
        provider="openai",
        model="gpt-4o-mini",
        immutable=True,
        age_turns=3,
    )
    assert result is not None
    assert result.openai_parts()[0]["type"] == "input_text"
    assert result.anthropic_blocks()[0]["type"] == "text"
    assert result.pages[0].sha256[:12] in result.manifest


def test_optical_factsheet_keeps_exact_identifiers_in_text(tmp_path) -> None:
    source = (
        _prose()
        + " Verification marker CITRINE-4821 appears twice: CITRINE-4821. "
        + "Artifact /srv/build/report.json belongs to BUILD_OUTPUT_DIR."
    )

    facts = dict(extract_optical_facts(source))
    result = TextOpticalCompressor(tmp_path, count_tokens=lambda text: len(text) * 10).compress(
        source,
        provider="openai",
        model="gpt-4o-mini",
        immutable=True,
        age_turns=3,
    )

    assert facts["CITRINE-4821"] == 2
    assert "/srv/build/report.json" in facts
    assert "BUILD_OUTPUT_DIR" in facts
    assert result is not None
    assert "CITRINE-4821 x2" in result.manifest


def test_anthropic_message_adapter_keeps_text_when_extracting_is_cheaper(tmp_path) -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "old", "content": _prose()}],
        },
        {"role": "assistant", "content": "summary"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "new", "content": _prose()}],
        },
    ]
    transformed, count = TextOpticalCompressor(tmp_path).compress_anthropic_messages(
        messages,
        model="claude-sonnet-4-6",
        generation=1,
        mutable_tail_messages=1,
    )
    assert count == 0
    assert transformed == messages
    assert transformed[2] == messages[2]
    assert messages[0]["content"][0]["content"] == _prose()


def test_anthropic_message_adapter_changes_only_old_prose_when_optical_is_cheaper(
    tmp_path, monkeypatch
) -> None:
    class ExpensiveTextCrusher:
        def compress(self, text: str, **_kwargs):
            return type("Result", (), {"compressed": text})()

    monkeypatch.setattr("headroom.transforms.text_crusher.TextCrusher", ExpensiveTextCrusher)
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "old", "content": _prose()}],
        },
        {"role": "assistant", "content": "summary"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "new", "content": _prose()}],
        },
    ]

    transformed, count = TextOpticalCompressor(tmp_path).compress_anthropic_messages(
        messages,
        model="claude-sonnet-4-6",
        generation=1,
        mutable_tail_messages=1,
    )

    assert count == 1
    old_content = transformed[0]["content"][0]["content"]
    assert old_content[0]["type"] == "text"
    assert old_content[1]["type"] == "image"
    assert transformed[2] == messages[2]
    assert messages[0]["content"][0]["content"] == _prose()
