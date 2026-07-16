"""Load and validate a scoring rubric from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from support_qa.models import Rubric


def load_rubric(path: str | Path) -> Rubric:
    """Read a rubric YAML file and validate it into a Rubric model."""
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return Rubric.model_validate(data)
