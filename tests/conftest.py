import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_qa.models import Criterion, Message, Role, Rubric, Ticket  # noqa: E402
from support_qa.rubric import load_rubric  # noqa: E402


@pytest.fixture
def rubric() -> Rubric:
    return load_rubric(ROOT / "config" / "rubric.yaml")


@pytest.fixture
def tiny_rubric() -> Rubric:
    return Rubric(
        name="Tiny",
        scale_max=5,
        criteria=[
            Criterion(key="resolution", title="Resolution", description="Solved?", weight=2.0),
            Criterion(key="empathy", title="Empathy", description="Polite?", weight=1.0),
        ],
    )


@pytest.fixture
def good_ticket() -> Ticket:
    return Ticket(
        id="T-good",
        messages=[
            Message(role=Role.CUSTOMER, text="I was charged twice, please help."),
            Message(
                role=Role.AGENT,
                text="I'm sorry about that. I've refunded the duplicate charge and it will "
                "be processed within a few days. Thank you for your patience.",
            ),
        ],
    )


@pytest.fixture
def bad_ticket() -> Ticket:
    return Ticket(
        id="T-bad",
        messages=[
            Message(role=Role.CUSTOMER, text="The export button does nothing."),
            Message(role=Role.AGENT, text="Try again."),
            Message(role=Role.CUSTOMER, text="Still broken."),
            Message(role=Role.AGENT, text="Ok."),
        ],
    )
