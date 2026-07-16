# LLM Support-QA Scorer

Automated quality scoring for customer-support conversations. Feed it a ticket
(a customer/agent transcript), and an **LLM-as-judge** grades it against a
configurable rubric — empathy, resolution, completeness, policy compliance —
returning a 0–100 score, a verdict, and per-criterion reasoning.

> Manually reviewing support tickets doesn't scale: teams sample a few percent
> and quality drifts between reviewers. This turns that review into a
> reproducible pipeline that can score **100% of conversations** consistently.

The repo ships with a **synthetic** dataset and generic rubric — no real data —
so you can clone it and see it work in seconds.

---

## Highlights

- **Configurable rubric** — criteria and weights live in [`config/rubric.yaml`](config/rubric.yaml); edit content without touching code.
- **Provider-agnostic judge** — any OpenAI-compatible endpoint (OpenAI, Polza.ai, local Ollama…).
- **Runs with zero secrets** — no API key? It falls back to a deterministic offline judge, so the demo, CLI, and tests all work out of the box.
- **Two interfaces** — a CLI batch report and a FastAPI HTTP service.
- **Structured output** — strict JSON, validated with Pydantic; weighted aggregation is pure, tested code (not left to the model).
- **Tested** — pytest suite covering rubric validation, aggregation math, judge parsing, and the API.

## How it works

```
ticket ─► prompt builder ─► LLM judge ─► per-criterion scores (1–5) ─► weighted aggregate ─► 0–100 + verdict
             (rubric)      (JSON out)       + reasoning                  (deterministic)
```

The LLM only does what it's good at — reading a conversation and rating each
criterion with a justification. The **scoring math is deterministic code**:
weighted average of the 1–5 ratings, normalized to 0–100, mapped to a verdict
(`excellent` ≥ 85, `good` ≥ 70, `needs_improvement` ≥ 50, else `poor`). That
split keeps results reproducible and auditable.

## Quickstart

```bash
git clone https://github.com/zubarevdm/llm-support-qa-scorer.git
cd llm-support-qa-scorer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Score the bundled synthetic tickets (offline judge, no key needed):
support-qa
```

To use a real LLM judge, copy `.env.example` to `.env`, set `LLM_API_KEY`
(and optionally `LLM_BASE_URL` / `LLM_MODEL`), then export it and re-run.

### Sample output

```
Rubric: Support Quality v1 v1.0  ·  judge: HeuristicJudge

TICKET          SCORE  VERDICT
--------------------------------------------
TCK-1001         68.8  needs_improvement
TCK-1002         50.0  needs_improvement
TCK-1004         39.6  poor
TCK-1010         72.9  good
TCK-1012         72.9  good
...
--------------------------------------------
AVERAGE          57.8
```

*(Scores above come from the offline heuristic judge — a simple text-signal
stand-in so the repo runs without a key: the over-promising ticket TCK-1004
lands lowest, the thorough ones highest. Plug in a real model via `.env` for
genuine, well-calibrated judgments.)*

## HTTP API

```bash
uvicorn support_qa.api:app --reload
```

- `GET /health` — liveness
- `GET /rubric` — the active rubric
- `POST /score` — body `{"ticket": {...}}` → an `Evaluation`

Interactive docs at `http://localhost:8000/docs`.

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "ticket": {
    "id": "T-1",
    "messages": [
      {"role": "customer", "text": "I was charged twice."},
      {"role": "agent", "text": "Sorry about that — I have refunded the duplicate charge already."}
    ]
  }
}'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_KEY` | *(empty)* | If set, use the real LLM judge; empty → offline heuristic judge |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |

Tune the rubric in [`config/rubric.yaml`](config/rubric.yaml): add/remove
criteria, reword descriptions, change weights. The CLI accepts `--rubric` and
`--input` to point at your own files, and `--json` for machine-readable output.

## Project layout

```
config/rubric.yaml        # scoring criteria + weights
data/sample_tickets.json  # synthetic demo tickets
src/support_qa/
  models.py               # Ticket, Rubric, Evaluation (Pydantic)
  rubric.py               # YAML loader + validation
  prompts.py              # judge prompt construction
  judge.py                # LLMJudge + offline HeuristicJudge
  scorer.py               # weighted aggregation -> verdict
  api.py                  # FastAPI service
  cli.py                  # batch CLI report
tests/                    # pytest suite
```

## Run the tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Notes

This is a clean-room portfolio project built from scratch on synthetic data.
The heuristic judge is intentionally simplistic — it exists so the pipeline is
runnable and testable offline; production quality comes from a real LLM behind
`LLM_API_KEY`.

## License

MIT — see [LICENSE](LICENSE).
