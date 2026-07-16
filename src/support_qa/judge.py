"""Judges that turn a ticket + rubric into per-criterion scores.

Two implementations:
- LLMJudge      — calls any OpenAI-compatible chat completions endpoint.
- HeuristicJudge — deterministic offline fallback so the project runs, is
  testable, and can be demoed without an API key.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import httpx

from support_qa.models import CriterionScore, Rubric, Ticket
from support_qa.prompts import SYSTEM_PROMPT, build_user_prompt


class Judge(Protocol):
    def score(self, ticket: Ticket, rubric: Rubric) -> tuple[list[CriterionScore], str]:
        """Return (criterion_scores, summary) for one ticket."""
        ...


class HeuristicJudge:
    """Offline, deterministic judge.

    Not a real quality model — it approximates each criterion with simple text
    signals (apologies, resolution cues, question handling, length). Its purpose
    is to keep the pipeline runnable and tests hermetic without network calls.
    """

    _APOLOGY = ("sorry", "apolog", "unfortunately", "we understand")
    _RESOLUTION = ("refund", "resolved", "fixed", "issued", "processed", "completed", "here is", "you can")
    _EMPATHY = ("understand", "appreciate", "thank", "happy to help", "glad")

    def score(self, ticket: Ticket, rubric: Rubric) -> tuple[list[CriterionScore], str]:
        agent_text = " ".join(m.text for m in ticket.messages if m.role.value == "agent").lower()
        customer_turns = sum(1 for m in ticket.messages if m.role.value == "customer")
        agent_turns = sum(1 for m in ticket.messages if m.role.value == "agent")
        scale = rubric.scale_max

        def clamp(v: int) -> int:
            return max(1, min(scale, v))

        def signal(words: tuple[str, ...]) -> int:
            hits = sum(1 for w in words if w in agent_text)
            return clamp(2 + hits)

        scores: list[CriterionScore] = []
        for c in rubric.criteria:
            key = c.key.lower()
            if "empath" in key or "tone" in key:
                val = signal(self._EMPATHY + self._APOLOGY)
                reason = "Agent language shows empathy/politeness signals." if val >= 3 else "Little empathetic language detected."
            elif "resol" in key or "outcome" in key:
                val = signal(self._RESOLUTION)
                reason = "Response contains resolution cues." if val >= 3 else "No clear resolution offered."
            elif "complet" in key or "clarity" in key:
                val = clamp(2 + (1 if agent_turns >= customer_turns else 0) + (1 if len(agent_text) > 120 else 0))
                reason = "Answer covers the request at reasonable length." if val >= 3 else "Answer looks thin relative to the question."
            elif "polic" in key or "complian" in key:
                risky = any(w in agent_text for w in ("guarantee", "promise", "for sure", "100%"))
                val = clamp(scale - (2 if risky else 0))
                reason = "No risky over-promises found." if not risky else "Contains over-promising language."
            else:
                val = clamp(3)
                reason = "Neutral baseline for this criterion."
            scores.append(CriterionScore(key=c.key, score=val, reasoning=reason))

        avg = sum(s.score for s in scores) / len(scores)
        summary = "Heuristic estimate (offline mode) — connect an LLM for a real judgment."
        _ = avg
        return scores, summary


class LLMJudge:
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def score(self, ticket: Ticket, rubric: Rubric) -> tuple[list[CriterionScore], str]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ticket, rubric)},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._parse(content, rubric)

    @staticmethod
    def _parse(content: str, rubric: Rubric) -> tuple[list[CriterionScore], str]:
        data = _extract_json(content)
        valid_keys = {c.key for c in rubric.criteria}
        scores: list[CriterionScore] = []
        for item in data.get("criterion_scores", []):
            key = item.get("key")
            if key not in valid_keys:
                continue
            raw = int(item.get("score", 1))
            scores.append(
                CriterionScore(
                    key=key,
                    score=max(1, min(rubric.scale_max, raw)),
                    reasoning=str(item.get("reasoning", "")).strip(),
                )
            )
        # Fill any criteria the model skipped so aggregation stays well-defined.
        seen = {s.key for s in scores}
        for c in rubric.criteria:
            if c.key not in seen:
                scores.append(CriterionScore(key=c.key, score=1, reasoning="No score returned by judge."))
        summary = str(data.get("summary", "")).strip() or "No summary returned."
        return scores, summary


def _extract_json(content: str) -> dict:
    """Parse a JSON object from a model response, tolerating code fences."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("judge response contained no JSON object")
        return json.loads(match.group(0))


def build_judge_from_env() -> Judge:
    """Pick a judge based on environment.

    Uses LLMJudge when LLM_API_KEY is set, otherwise the offline HeuristicJudge.
    """
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return HeuristicJudge()
    return LLMJudge(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )
