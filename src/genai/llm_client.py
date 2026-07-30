"""Provider-agnostic LLM access with fallback, tracing, and an offline mock."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

from src.common.logger import get_logger, get_request_id
from src.common.settings import get_llm_config, get_settings

logger = get_logger(__name__)

# litellm logs the full prompt at INFO, which would put customer narratives in the logs.
litellm.suppress_debug_info = True

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LLMError(RuntimeError):
    """Raised when every configured provider failed."""


@dataclass(frozen=True)
class Provider:
    """One entry in the fallback chain."""

    name: str
    model: str
    api_key_env: str


@dataclass
class LLMResponse:
    """One completion, with the operational facts needed to audit and bill it."""

    content: str
    model: str
    provider: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attempt: int = 1
    is_mock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def used_fallback(self) -> bool:
        """True when the first provider in the chain did not serve this call."""
        return self.attempt > 1

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def as_json(self) -> dict[str, Any]:
        """Parse the content as JSON, tolerating markdown fences."""
        # Models fence JSON despite being told not to; stripped once, not per caller.
        return json.loads(_FENCE.sub("", self.content).strip())


class LLMClient:
    """Calls the configured provider, falls back to a second vendor, or runs offline."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.settings = get_settings()
        self.config = config or get_llm_config()
        gen = self.config["generation"]

        self.providers = [Provider(**p) for p in self.config["providers"]]
        self.temperature = gen["temperature"]
        self.max_tokens = gen["max_tokens"]
        self.timeout = gen["timeout_seconds"]
        self.num_retries = gen["num_retries"]
        self.mock_mode = self.settings.llm_mock_mode

        self._langfuse = self._init_langfuse()
        logger.info(
            "llm client ready",
            extra={
                "chain": [p.model for p in self.providers],
                "configured": [p.name for p in self.providers if self._key(p)],
                "mock_mode": self.mock_mode,
                "langfuse": self._langfuse is not None,
            },
        )

    def _init_langfuse(self) -> Any:
        if not self.settings.langfuse_enabled:
            return None
        try:
            from langfuse import Langfuse

            return Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception as exc:
            # Observability is not on the critical path; losing traces must not lose claims.
            logger.warning("langfuse unavailable", extra={"error": str(exc)})
            return None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        persona: str = "unknown",
        claim_id: str = "unknown",
        json_mode: bool = True,
    ) -> LLMResponse:
        """Get one completion, trying the primary provider then the fallback."""
        if self.mock_mode:
            return self._mock(system_prompt, user_prompt, persona=persona)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        attempt = 0
        for provider in self.providers:
            api_key = self._key(provider)
            # A provider with no key is skipped, not failed: an unconfigured vendor is a
            # deployment choice, and counting it as an outage would hide real ones.
            if not api_key:
                continue
            attempt += 1
            try:
                response = self._call(provider, messages, api_key, json_mode, attempt)
                self._trace(response, persona, claim_id, user_prompt)
                return response
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "llm provider failed",
                    extra={
                        "provider": provider.name,
                        "model": provider.model,
                        "attempt": attempt,
                        "error": str(exc)[:200],
                    },
                )

        # Truncated: quota errors run to kilobytes and this reaches a response and a log.
        raise LLMError(f"All providers failed. Last error: {str(last_error)[:300]}")

    def _key(self, provider: Provider) -> str:
        return self.settings.api_key_for(provider.api_key_env)

    def _call(
        self,
        provider: Provider,
        messages: list[dict[str, str]],
        api_key: str,
        json_mode: bool,
        attempt: int,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "num_retries": self.num_retries,
            "api_key": api_key,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        raw = litellm.completion(**kwargs)
        latency = int((time.perf_counter() - started) * 1000)

        usage = getattr(raw, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=raw) or 0.0)
        except Exception:
            # Not every model has published pricing; an unknown cost is not a failed call.
            cost = 0.0

        return LLMResponse(
            content=raw.choices[0].message.content or "",
            model=provider.model,
            provider=provider.name,
            latency_ms=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            attempt=attempt,
        )

    def _trace(self, response: LLMResponse, persona: str, claim_id: str, prompt: str) -> None:
        if self._langfuse is None:
            return
        try:
            self._langfuse.trace(
                name=self.config["observability"]["langfuse"]["trace_name"],
                metadata={
                    "persona": persona,
                    "claim_id": claim_id,
                    "model": response.model,
                    "provider": response.provider,
                    "attempt": response.attempt,
                    "used_fallback": response.used_fallback,
                    "latency_ms": response.latency_ms,
                    "total_tokens": response.total_tokens,
                    "cost_usd": response.cost_usd,
                    "request_id": get_request_id(),
                    # The prompt embeds the customer narrative, so only a digest is sent.
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                },
            )
        except Exception as exc:
            logger.warning("langfuse trace failed", extra={"error": str(exc)[:200]})

    def _mock(self, system_prompt: str, user_prompt: str, *, persona: str) -> LLMResponse:
        """Deterministic offline response, so CI and a laptop with no keys still run."""
        started = time.perf_counter()
        content = json.dumps(_MOCK_RESPONSES.get(persona, _MOCK_RESPONSES["customer"]))
        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            attempt=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            # Char-count approximation: no tokeniser belongs in an offline code path.
            prompt_tokens=len(system_prompt + user_prompt) // 4,
            completion_tokens=len(content) // 4,
            is_mock=True,
        )


_MOCK_RESPONSES: dict[str, dict[str, Any]] = {
    "customer": {
        "summary": "Your claim has been received and is being processed. We have reviewed "
        "the details you provided about your device.",
        "key_factors": [
            "The excess on your policy is small compared with your device's value.",
            "Your policy was active when the damage occurred.",
        ],
        "next_steps": [
            "No action is needed from you right now.",
            "We will contact you if we need anything further.",
        ],
    },
    "adjuster": {
        "summary": "Model routed this claim to the straight-through queue. Attribution is "
        "dominated by the excess-to-value ratio and the device value.",
        "key_factors": [
            "fee_to_rrp_ratio: 0.04, contribution -0.0408 (reduces decline risk)",
            "rrp: 1503.88 EUR, contribution -0.0318 (reduces decline risk)",
        ],
        "risk_assessment": "No forced-review triggers. The model cannot assess the "
        "circumstances of the loss, which are described only in the narrative.",
        "review_notes": "Confirm the device serial matches the policy record.",
    },
    "auditor": {
        "summary": "Decision produced by model version 1 (random_forest) under prompt "
        "version v1.0.0, routed by a threshold of 0.344.",
        "key_factors": [
            "fee_to_rrp_ratio: value 0.04, contribution -0.0408",
            "rrp: value 1503.88, contribution -0.0318",
        ],
        "provenance_record": "model_version 1, prompt_version v1.0.0, git_commit 0dd4f98.",
        "human_oversight": "Human review not required at this score band.",
        "limitations": "Trained on 2880 claims; structured fields only.",
    },
}
