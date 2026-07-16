"""Domain models: tickets, rubric, and evaluation results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"


class Message(BaseModel):
    role: Role
    text: str


class Ticket(BaseModel):
    """A single support conversation to be evaluated."""

    id: str
    channel: str = "email"
    subject: str = ""
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("ticket must contain at least one message")
        return value

    def transcript(self) -> str:
        """Render the conversation as plain text for the judge prompt."""
        lines = [f"{m.role.value.upper()}: {m.text}" for m in self.messages]
        return "\n".join(lines)


class Criterion(BaseModel):
    """One rubric dimension the judge scores on a 1-5 scale."""

    key: str
    title: str
    description: str
    weight: float = Field(gt=0)


class Rubric(BaseModel):
    name: str
    version: str = "1.0"
    scale_max: int = Field(default=5, ge=2, le=10)
    criteria: list[Criterion]

    @field_validator("criteria")
    @classmethod
    def _unique_non_empty(cls, value: list[Criterion]) -> list[Criterion]:
        if not value:
            raise ValueError("rubric must define at least one criterion")
        keys = [c.key for c in value]
        if len(keys) != len(set(keys)):
            raise ValueError("criterion keys must be unique")
        return value

    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)


class CriterionScore(BaseModel):
    key: str
    score: int
    reasoning: str


class Evaluation(BaseModel):
    """Full judged result for one ticket."""

    ticket_id: str
    rubric: str
    criterion_scores: list[CriterionScore]
    overall_score: float = Field(ge=0, le=100)
    verdict: str
    summary: str
