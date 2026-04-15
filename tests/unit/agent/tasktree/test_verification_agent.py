"""Unit tests for LLMVerificationAgent."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.execution.verification import (
    LLMVerificationAgent,
    _build_verification_prompt,
    _parse_verification_response,
)
from nanobot.agent.tasktree.models import Artifact, NodeResult, TaskStatus
from nanobot.agent.tasktree.scheduler import VerificationResult


class TestBuildVerificationPrompt:
    def test_includes_root_goal(self):
        prompt = _build_verification_prompt("build a bot", {})
        assert "build a bot" in prompt
        assert "Root Goal:" in prompt

    def test_includes_node_results(self):
        results = {
            "root": NodeResult(
                node_id="root",
                summary="analyzed and planned",
                artifacts=[
                    Artifact(type="plan", path=None, description="three-step plan"),
                ],
            ),
        }
        prompt = _build_verification_prompt("build a bot", results)
        assert "### Node root" in prompt
        assert "analyzed and planned" in prompt
        assert "three-step plan" in prompt

    def test_empty_results(self):
        prompt = _build_verification_prompt("simple task", {})
        assert "(no results)" in prompt


class TestParseVerificationResponse:
    def test_parses_passed_true(self):
        text = '{"passed": true, "failed_nodes": [], "reason": "all good", "evidence": ["step1"]}'
        result = _parse_verification_response(text)
        assert result.passed is True
        assert result.failed_nodes == []
        assert result.reason == "all good"
        assert result.evidence == ["step1"]

    def test_parses_passed_false_with_failed_nodes(self):
        text = '{"passed": false, "failed_nodes": ["root.0"], "reason": "incomplete", "evidence": ["missing file"]}'
        result = _parse_verification_response(text)
        assert result.passed is False
        assert result.failed_nodes == ["root.0"]
        assert result.reason == "incomplete"

    def test_json_embedded_in_markdown(self):
        text = 'Here is the result:\n```json\n{"passed": true, "failed_nodes": [], "reason": "ok", "evidence": []}\n```'
        result = _parse_verification_response(text)
        assert result.passed is True

    def test_non_json_falls_back_to_failure(self):
        text = "I think the task is done."
        result = _parse_verification_response(text)
        assert result.passed is False
        assert "Could not parse" in result.reason

    def test_malformed_json_falls_back_to_failure(self):
        text = '{"passed": true, bad json'
        result = _parse_verification_response(text)
        assert result.passed is False

    def test_empty_text_falls_back_to_failure(self):
        result = _parse_verification_response("")
        assert result.passed is False


class TestLLMVerificationAgent:
    @pytest.mark.asyncio
    async def test_verify_returns_pass_on_success(self):
        class FakeProvider:
            async def complete(self, prompt, model, max_tokens):
                return type("Resp", (), {"content": '{"passed": true, "failed_nodes": [], "reason": "goal achieved", "evidence": []}'})()

            def get_default_model(self):
                return "test-model"

        agent = LLMVerificationAgent(provider=FakeProvider())
        result = await agent.verify("build a bot", {"root": NodeResult(node_id="root", summary="done")})
        assert result.passed is True
        assert result.reason == "goal achieved"

    @pytest.mark.asyncio
    async def test_verify_returns_fail_on_llm_failure(self):
        class FailProvider:
            async def complete(self, prompt, model, max_tokens):
                raise RuntimeError("provider error")

            def get_default_model(self):
                return "test-model"

        agent = LLMVerificationAgent(provider=FailProvider())
        result = await agent.verify("build a bot", {})
        assert result.passed is False
        assert "error" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_verify_uses_specified_model(self):
        calls = []

        class CheckModelProvider:
            async def complete(self, prompt, model, max_tokens):
                calls.append(model)
                return type("Resp", (), {"content": '{"passed": true, "failed_nodes": [], "reason": "ok", "evidence": []}'})()

            def get_default_model(self):
                return "default-model"

        agent = LLMVerificationAgent(provider=CheckModelProvider(), model="custom-model")
        await agent.verify("test", {})
        assert calls == ["custom-model"]
