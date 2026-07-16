"""Aggregate per-criterion scores into an overall quality result."""

from __future__ import annotations

from support_qa.judge import Judge
from support_qa.models import CriterionScore, Evaluation, Rubric, Ticket


def _verdict(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "needs_improvement"
    return "poor"


def aggregate(
    scores: list[CriterionScore], rubric: Rubric
) -> float:
    """Weighted average of 1..scale_max scores, normalized to 0..100."""
    by_key = {s.key: s.score for s in scores}
    weighted = 0.0
    for c in rubric.criteria:
        weighted += c.weight * by_key.get(c.key, 1)
    weighted_avg = weighted / rubric.total_weight()
    # Map [1, scale_max] -> [0, 100].
    normalized = (weighted_avg - 1) / (rubric.scale_max - 1) * 100
    return round(normalized, 1)


def evaluate_ticket(ticket: Ticket, rubric: Rubric, judge: Judge) -> Evaluation:
    scores, summary = judge.score(ticket, rubric)
    overall = aggregate(scores, rubric)
    return Evaluation(
        ticket_id=ticket.id,
        rubric=rubric.name,
        criterion_scores=scores,
        overall_score=overall,
        verdict=_verdict(overall),
        summary=summary,
    )


def evaluate_batch(
    tickets: list[Ticket], rubric: Rubric, judge: Judge
) -> list[Evaluation]:
    return [evaluate_ticket(t, rubric, judge) for t in tickets]
