"""A Groq-backed judge for DeepEval, so evaluation does not require an OpenAI key."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from src.common.settings import get_settings

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# A judge should be at least as capable as the model it grades, hence 70b. Overridable
# because Groq meters tokens per model, and a spent quota must not read as a failed suite.
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "groq/llama-3.3-70b-versatile")


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge routed through LiteLLM to Groq."""

    # DeepEvalBaseLLM types these as `*args -> str`, but every metric calls the concrete
    # `(prompt, schema) -> (result, cost)` contract its own GPTModel implements.
    def __init__(self, model: str = JUDGE_MODEL) -> None:
        # Not `self.model`: the base class already owns that name for a loaded client.
        self.model_id = model
        super().__init__(model_name=model)

    def load_model(self) -> str:  # type: ignore[override]
        """DeepEval calls this before generating; there is no client to build."""
        return self.model_id

    def get_model_name(self) -> str:
        """Name shown in DeepEval's report."""
        return self.model_id

    def generate(  # type: ignore[override]
        self, prompt: str, schema: BaseModel | None = None
    ) -> tuple[Any, float]:
        """Judge one prompt, returning the result and its cost."""
        import litellm

        # Judging is a structured task, so temperature 0: the same explanation must score
        # the same way twice, or the metric is not a measurement.
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": self._with_schema(prompt, schema)}],
            "temperature": 0.0,
            "max_tokens": 2048,
            "num_retries": 2,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._complete(litellm, kwargs)
        content = response.choices[0].message.content or ""
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost = 0.0

        if schema is None:
            return content, cost
        return schema.model_validate(json.loads(_FENCE.sub("", content).strip())), cost

    async def a_generate(  # type: ignore[override]
        self, prompt: str, schema: BaseModel | None = None
    ) -> tuple[Any, float]:
        """Async path. The judge is called serially here, so it delegates."""
        return self.generate(prompt, schema)

    @staticmethod
    def _complete(litellm: Any, kwargs: dict[str, Any]) -> Any:
        """Call Groq, moving to the reserve key once the first has spent its daily budget."""
        last: Exception | None = None
        for key in _groq_keys():
            try:
                return litellm.completion(**kwargs, api_key=key)
            except Exception as exc:
                # Only a spent budget is worth another key; anything else repeats identically.
                if "rate_limit" not in str(exc).lower():
                    raise
                last = exc
        raise last if last else RuntimeError("no Groq key configured")

    @staticmethod
    def _with_schema(prompt: str, schema: BaseModel | None) -> str:
        if schema is None:
            return prompt
        # Groq has no native structured-output mode, so the schema is appended to the
        # prompt and the response is parsed. Without this the judge returns prose and
        # every metric fails on a parse error rather than on the thing being measured.
        return (
            f"{prompt}\n\n"
            f"Respond with a single JSON object matching this schema, and nothing else:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )


def _groq_keys() -> list[str]:
    settings = get_settings()
    keys = (settings.groq_api_key, settings.groq_api_key_2, settings.groq_api_key_3)
    return [k for k in keys if k]


def judge_available() -> bool:
    """Whether a judge can be built, so evaluation can skip rather than fail."""
    return bool(_groq_keys())
