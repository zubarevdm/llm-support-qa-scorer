"""Unit tests for LLMJudge response parsing (no network)."""

import json

from support_qa.judge import LLMJudge, _extract_json


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_code_fence():
    fenced = "```json\n{\"a\": 1}\n```"
    assert _extract_json(fenced) == {"a": 1}


def test_parse_clamps_and_fills(tiny_rubric):
    content = json.dumps(
        {
            "criterion_scores": [
                {"key": "resolution", "score": 9, "reasoning": "great"},
                {"key": "unknown", "score": 5, "reasoning": "ignored"},
            ],
            "summary": "ok",
        }
    )
    scores, summary = LLMJudge._parse(content, tiny_rubric)
    by_key = {s.key: s for s in scores}
    # unknown key dropped, empathy backfilled, resolution clamped to scale_max
    assert set(by_key) == {"resolution", "empathy"}
    assert by_key["resolution"].score == tiny_rubric.scale_max
    assert by_key["empathy"].score == 1
    assert summary == "ok"
