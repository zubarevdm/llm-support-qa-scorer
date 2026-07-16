"""Prompt construction for the LLM judge."""

from __future__ import annotations

from support_qa.models import Rubric, Ticket

SYSTEM_PROMPT = (
    "You are a meticulous quality-assurance reviewer for a customer support team. "
    "You evaluate a single support conversation against a fixed rubric and return "
    "a strict JSON object. Judge only what the agent actually did, be consistent, "
    "and never invent facts that are not in the transcript."
)


def build_criteria_block(rubric: Rubric) -> str:
    lines = []
    for c in rubric.criteria:
        lines.append(f"- {c.key} ({c.title}): {c.description}")
    return "\n".join(lines)


def build_user_prompt(ticket: Ticket, rubric: Rubric) -> str:
    """Assemble the judging instruction for one ticket."""
    keys = ", ".join(c.key for c in rubric.criteria)
    return f"""Evaluate the following support conversation.

Rubric "{rubric.name}" (score each criterion from 1 to {rubric.scale_max}, where 1 = poor and {rubric.scale_max} = excellent):
{build_criteria_block(rubric)}

Conversation transcript:
---
{ticket.transcript()}
---

Return ONLY a JSON object with this exact shape:
{{
  "criterion_scores": [
    {{"key": "<one of: {keys}>", "score": <int 1..{rubric.scale_max}>, "reasoning": "<one concise sentence>"}}
  ],
  "summary": "<one-sentence overall assessment>"
}}
Include exactly one entry per rubric criterion. Do not add commentary outside the JSON.
"""
