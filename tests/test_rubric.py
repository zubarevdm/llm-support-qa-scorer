import pytest
from pydantic import ValidationError

from support_qa.models import Criterion, Rubric


def test_default_rubric_loads(rubric):
    assert rubric.name
    assert len(rubric.criteria) == 4
    assert rubric.total_weight() == pytest.approx(6.0)


def test_rubric_rejects_duplicate_keys():
    with pytest.raises(ValidationError):
        Rubric(
            name="dup",
            criteria=[
                Criterion(key="a", title="A", description="x", weight=1),
                Criterion(key="a", title="A2", description="y", weight=1),
            ],
        )


def test_rubric_rejects_empty_criteria():
    with pytest.raises(ValidationError):
        Rubric(name="empty", criteria=[])


def test_criterion_weight_must_be_positive():
    with pytest.raises(ValidationError):
        Criterion(key="a", title="A", description="x", weight=0)
