"""Command-line entry point: score a JSON file of tickets and print a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from support_qa.judge import build_judge_from_env
from support_qa.models import Ticket
from support_qa.rubric import load_rubric
from support_qa.scorer import evaluate_batch

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUBRIC = _ROOT / "config" / "rubric.yaml"
_DEFAULT_INPUT = _ROOT / "data" / "sample_tickets.json"


def _load_tickets(path: Path) -> list[Ticket]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Ticket.model_validate(item) for item in data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score support tickets against a rubric.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Path to tickets JSON.")
    parser.add_argument("--rubric", type=Path, default=_DEFAULT_RUBRIC, help="Path to rubric YAML.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    args = parser.parse_args(argv)

    rubric = load_rubric(args.rubric)
    tickets = _load_tickets(args.input)
    judge = build_judge_from_env()
    results = evaluate_batch(tickets, rubric, judge)

    if args.json:
        print(json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2))
        return 0

    print(f"\nRubric: {rubric.name} v{rubric.version}  ·  judge: {type(judge).__name__}\n")
    print(f"{'TICKET':<14}{'SCORE':>7}  VERDICT")
    print("-" * 44)
    for r in results:
        print(f"{r.ticket_id:<14}{r.overall_score:>7}  {r.verdict}")
    avg = sum(r.overall_score for r in results) / len(results)
    print("-" * 44)
    print(f"{'AVERAGE':<14}{round(avg, 1):>7}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
