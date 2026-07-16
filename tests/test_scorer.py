import pytest

from support_qa.judge import HeuristicJudge
from support_qa.models import CriterionScore
from support_qa.scorer import aggregate, evaluate_ticket


def test_aggregate_all_max_is_100(tiny_rubric):
    scores = [
        CriterionScore(key="resolution", score=5, reasoning=""),
        CriterionScore(key="empathy", score=5, reasoning=""),
    ]
    assert aggregate(scores, tiny_rubric) == 100.0


def test_aggregate_all_min_is_0(tiny_rubric):
    scores = [
        CriterionScore(key="resolution", score=1, reasoning=""),
        CriterionScore(key="empathy", score=1, reasoning=""),
    ]
    assert aggregate(scores, tiny_rubric) == 0.0


def test_aggregate_respects_weights(tiny_rubric):
    # resolution (weight 2) high, empathy (weight 1) low -> weighted toward resolution
    high_res = [
        CriterionScore(key="resolution", score=5, reasoning=""),
        CriterionScore(key="empathy", score=1, reasoning=""),
    ]
    high_emp = [
        CriterionScore(key="resolution", score=1, reasoning=""),
        CriterionScore(key="empathy", score=5, reasoning=""),
    ]
    assert aggregate(high_res, tiny_rubric) > aggregate(high_emp, tiny_rubric)


def test_missing_criterion_defaults_to_one(tiny_rubric):
    scores = [CriterionScore(key="resolution", score=5, reasoning="")]
    # empathy missing -> treated as 1
    result = aggregate(scores, tiny_rubric)
    assert 0 < result < 100


def test_good_ticket_beats_bad_ticket(rubric, good_ticket, bad_ticket):
    judge = HeuristicJudge()
    good = evaluate_ticket(good_ticket, rubric, judge)
    bad = evaluate_ticket(bad_ticket, rubric, judge)
    assert good.overall_score > bad.overall_score
    assert good.verdict in {"excellent", "good", "needs_improvement"}


def test_evaluation_has_one_score_per_criterion(rubric, good_ticket):
    result = evaluate_ticket(good_ticket, rubric, HeuristicJudge())
    assert {s.key for s in result.criterion_scores} == {c.key for c in rubric.criteria}
    assert 0 <= result.overall_score <= 100
