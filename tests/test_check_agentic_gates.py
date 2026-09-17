"""Unit tests for gate A5 in examples/alpha/eval_sft/check_agentic_gates.py.

A5 = think + 도구호출 동시 경로 (KNOWN_ISSUES 2026-09-14). 판정 함수는 순수라 GPU 없이 검증한다.
결함 표본은 보존된 SWE 궤적(iter1800, `extra.response` = vLLM 원응답)에서 그대로 옮겼다 —
tool_calls 는 있는데 reasoning_content 가 null 이고 content 에 `</think>` 가 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/alpha/eval_sft"))

import check_agentic_gates as G  # noqa: E402


def _tc(cmd: str) -> list[dict]:
    return [{"type": "function", "function": {"name": "bash", "arguments": f'{{"command": "{cmd}"}}'}}]


# iter1800 SWE 궤적의 실제 vLLM 원응답 3건 (content·reasoning_content 원문 그대로).
DEFECT_REAL = [
    {"content": "Let me look at the separability_matrix function in the core.py file, and also look at "
                "the separable.py file to understand how it works.",
     "reasoning_content": None, "tool_calls": _tc("cat /testbed/astropy/modeling/separable.py")},
    {"content": "Let me find the TimeSeries class in the astropy.time.timeseries module.",
     "reasoning_content": None, "tool_calls": _tc("grep -l 'class TimeSeries'")},
    {"content": "Let me search for the specific code mentioned in the PR description that handles the "
                "conversion of structured ndarrays to NdarrayMixin.",
     "reasoning_content": None, "tool_calls": _tc("grep -r data_view /testbed")},
]
HEALTHY_FIELD = {"content": "I'll search the files.", "reasoning_content": "The user wants main(). grep -l.",
                 "tool_calls": _tc("grep -l 'def main' *.py")}
HEALTHY_FIELD_ALT = {"content": "", "reasoning": "Need grep.", "tool_calls": _tc("grep -l main *.py")}
HEALTHY_INLINE = {"content": "Need to grep for def main.</think>Searching.", "reasoning_content": None,
                  "tool_calls": _tc("grep -l main *.py")}
NO_CALL = {"content": "I would run grep.", "reasoning_content": "hmm", "tool_calls": None}
NO_REASON = {"content": "", "reasoning_content": None, "tool_calls": _tc("ls")}


# ---------------------------------------------------------------- judge_a5

@pytest.mark.parametrize("msg", DEFECT_REAL)
def test_real_defect_responses_fail(msg):
    v, why = G.judge_a5(msg)
    assert v == "fail", why


def test_reasoning_field_passes():
    assert G.judge_a5(HEALTHY_FIELD)[0] == "pass"


def test_reasoning_alt_key_passes():
    """vLLM 은 버전에 따라 `reasoning` 키를 쓴다 — 둘 다 받아야 한다."""
    assert G.judge_a5(HEALTHY_FIELD_ALT)[0] == "pass"


def test_inline_think_passes():
    assert G.judge_a5(HEALTHY_INLINE)[0] == "pass"


def test_field_plus_residual_marker_passes_with_note():
    v, why = G.judge_a5({**HEALTHY_FIELD, "content": "x</think>y"})
    assert v == "pass" and "잔존" in why


def test_no_tool_call_is_inconclusive():
    assert G.judge_a5(NO_CALL)[0] == "no_tool_call"


def test_tool_call_without_any_text_is_inconclusive():
    assert G.judge_a5(NO_REASON)[0] == "no_reasoning"


# ---------------------------------------------------------------- gate_a5 (샘플링·다수결)

def _feed(monkeypatch, seq):
    calls = {"n": 0}

    def fake_post(base_url, body, timeout=300):
        i = calls["n"]
        calls["n"] += 1
        m = seq[i] if i < len(seq) else seq[-1]
        if m is None:
            return False, "HTTP 500"
        assert "enable_thinking" not in str(body.get("chat_template_kwargs", "")), "A5 는 thinking ON 이어야 한다"
        assert body.get("tools") and body.get("tool_choice") == "auto"
        return True, {"choices": [{"message": m}]}

    monkeypatch.setattr(G, "_post", fake_post)
    return calls


def test_defective_fleet_fails_after_two(monkeypatch):
    calls = _feed(monkeypatch, [DEFECT_REAL[0], DEFECT_REAL[1]])
    ok, msg = G.gate_a5("http://x/v1")
    assert not ok and calls["n"] == 2
    assert "REASONING_PARSER" in msg


def test_healthy_fleet_passes_after_two(monkeypatch):
    calls = _feed(monkeypatch, [HEALTHY_FIELD])
    ok, _ = G.gate_a5("http://x/v1")
    assert ok and calls["n"] == 2


def test_never_observed_path_fails(monkeypatch):
    """경로를 못 보면 통과로 쓰지 않는다."""
    calls = _feed(monkeypatch, [NO_CALL])
    ok, msg = G.gate_a5("http://x/v1")
    assert not ok and calls["n"] == G.A5_MAX_ATTEMPTS and "미관측" in msg


def test_inconclusive_are_skipped(monkeypatch):
    calls = _feed(monkeypatch, [NO_CALL, NO_REASON, DEFECT_REAL[0], DEFECT_REAL[2]])
    ok, _ = G.gate_a5("http://x/v1")
    assert not ok and calls["n"] == 4


def test_disagreement_resolved_by_majority(monkeypatch):
    calls = _feed(monkeypatch, [DEFECT_REAL[0], HEALTHY_FIELD, HEALTHY_INLINE])
    ok, msg = G.gate_a5("http://x/v1")
    assert ok and calls["n"] == 3 and "pass 2 / fail 1" in msg


def test_request_failure_fails(monkeypatch):
    _feed(monkeypatch, [None])
    ok, msg = G.gate_a5("http://x/v1")
    assert not ok and "요청 실패" in msg


# ---------------------------------------------------------------- CLI --tool-path

def _stub_gates(monkeypatch, a5_ok: bool):
    monkeypatch.setattr(G, "gate_a1", lambda u: (True, "ok"))
    monkeypatch.setattr(G, "gate_a4", lambda u: (True, "ok"))
    monkeypatch.setattr(G, "gate_a5", lambda u: (a5_ok, "a5"))


def test_required_blocks_on_a5_failure(monkeypatch):
    _stub_gates(monkeypatch, a5_ok=False)
    monkeypatch.setattr(sys, "argv", ["x", "--base-url", "http://x/v1", "--skip-container"])
    assert G.main() == 1


def test_report_does_not_block(monkeypatch):
    """TB-1·TB-2 는 네이티브 tools 를 안 쓰므로 A5 실패로 막지 않는다."""
    _stub_gates(monkeypatch, a5_ok=False)
    monkeypatch.setattr(sys, "argv", ["x", "--base-url", "http://x/v1", "--skip-container", "--tool-path", "report"])
    assert G.main() == 0


def test_required_passes_when_a5_passes(monkeypatch):
    _stub_gates(monkeypatch, a5_ok=True)
    monkeypatch.setattr(sys, "argv", ["x", "--base-url", "http://x/v1", "--skip-container"])
    assert G.main() == 0


# ---------------------------------------------------------------- A4 (2026-09-17 다중 표본화)
# iter600(최종 런) 실제 응답: 도구를 안 부르고 되물었다 — 파서 실패가 아니라 경로 미관측이어야 한다.
A4_NO_CALL_REAL = {"content": "To proceed, could you please specify the command you'd like me to run in the "
                              "current directory?", "reasoning_content": None, "tool_calls": None}
A4_PARSED = {"content": "", "reasoning_content": None, "tool_calls": _tc("ls -la")}
A4_XML_UNPARSED = {"content": '<function=bash>\n<parameter=command>ls</parameter>\n</function>',
                   "reasoning_content": None, "tool_calls": None}


def test_judge_a4_parsed():
    v, why = G.judge_a4(A4_PARSED)
    assert v == "parsed" and "bash" in why


def test_judge_a4_xml_unparsed_is_parser_mismatch():
    v, why = G.judge_a4(A4_XML_UNPARSED)
    assert v == "xml_unparsed" and "qwen3_xml" in why


def test_judge_a4_real_no_call_is_not_observed():
    v, _ = G.judge_a4(A4_NO_CALL_REAL)
    assert v == "no_call"


def _feed_a4(monkeypatch, seq):
    calls = {"n": 0}

    def fake_post(base_url, body, timeout=300):
        i = calls["n"]
        calls["n"] += 1
        m = seq[i] if i < len(seq) else seq[-1]
        if m is None:
            return False, "HTTP 500"
        assert body.get("chat_template_kwargs", {}).get("enable_thinking") is False, "A4 는 thinking OFF"
        assert body.get("tools") and body.get("tool_choice") == "auto"
        return True, {"choices": [{"message": m}]}

    monkeypatch.setattr(G, "_post", fake_post)
    return calls


def test_a4_passes_after_no_call_resample(monkeypatch):
    calls = _feed_a4(monkeypatch, [A4_NO_CALL_REAL, A4_PARSED])
    ok, msg = G.gate_a4("http://x/v1")
    assert ok and calls["n"] == 2 and "미호출 1회" in msg


def test_a4_xml_unparsed_fails_immediately(monkeypatch):
    calls = _feed_a4(monkeypatch, [A4_XML_UNPARSED, A4_PARSED])
    ok, msg = G.gate_a4("http://x/v1")
    assert not ok and calls["n"] == 1 and "qwen3_xml" in msg


def test_a4_all_no_call_fails_but_names_model_not_parser(monkeypatch):
    calls = _feed_a4(monkeypatch, [A4_NO_CALL_REAL])
    ok, msg = G.gate_a4("http://x/v1")
    assert not ok and calls["n"] == G.A4_MAX_ATTEMPTS and "파서 문제가 아니라" in msg


def test_a4_request_failure_fails(monkeypatch):
    _feed_a4(monkeypatch, [None])
    ok, msg = G.gate_a4("http://x/v1")
    assert not ok and "요청 실패" in msg
