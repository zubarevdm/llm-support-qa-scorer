"""FastAPI surface for scoring tickets over HTTP."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from support_qa import __version__
from support_qa.judge import build_judge_from_env
from support_qa.models import Evaluation, Rubric, Ticket
from support_qa.rubric import load_rubric
from support_qa.scorer import evaluate_ticket

DEFAULT_RUBRIC = Path(__file__).resolve().parents[2] / "config" / "rubric.yaml"

app = FastAPI(
    title="LLM Support-QA Scorer",
    version=__version__,
    description="Score support ticket quality against a configurable rubric using an LLM-as-judge.",
)


@lru_cache(maxsize=1)
def get_rubric() -> Rubric:
    return load_rubric(DEFAULT_RUBRIC)


class ScoreRequest(BaseModel):
    ticket: Ticket


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/rubric", response_model=Rubric)
def rubric() -> Rubric:
    return get_rubric()


@app.post("/score", response_model=Evaluation)
def score(request: ScoreRequest) -> Evaluation:
    judge = build_judge_from_env()
    return evaluate_ticket(request.ticket, get_rubric(), judge)
