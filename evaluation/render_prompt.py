"""Renders the production prompts for promptfoo, using the production Jinja2 environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.settings import get_llm_config  # noqa: E402

CONTEXTS = Path(__file__).resolve().parent / "contexts"


def _environment() -> Any:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    # Same settings as ExplanationPipeline. StrictUndefined matters most: a variable a
    # persona is not allowed to see must raise here, not render as an empty string.
    return Environment(
        loader=FileSystemLoader(ROOT / "prompts"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


def render(context: dict[str, Any]) -> list[dict[str, str]]:
    """promptfoo prompt function: build the chat messages for one persona."""
    persona = context["vars"]["persona"]
    config = get_llm_config()
    env = _environment()

    variables = json.loads((CONTEXTS / f"{persona}.json").read_text(encoding="utf-8"))
    # Rendered here rather than replayed from a captured string, so a template edit that
    # breaks a persona fails this suite instead of passing against a stale snapshot.
    return [
        {"role": "system", "content": env.get_template(config["prompts"]["system"]).render()},
        {
            "role": "user",
            "content": env.get_template(config["prompts"]["templates"][persona]).render(
                **variables
            ),
        },
    ]
