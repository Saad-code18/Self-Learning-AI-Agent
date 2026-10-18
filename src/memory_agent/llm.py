"""
llm.py - a tiny wrapper around the chat model.

Why a wrapper?
    * The rest of the code should not care WHICH provider we use.  OpenAI and
      Ollama both speak the "OpenAI chat completions" protocol, so one class
      with a different `base_url` covers both (= no vendor lock-in).
    * Every call is measured: how many tokens went in/out and how long it took.
      The benchmark script uses these numbers to prove the token/cost/latency
      savings of the memory pipeline.
    * JSON mode is used by the memory pipeline (fact extraction and
      add/update/delete decisions) so we get machine-readable answers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from openai import BadRequestError, OpenAI

from memory_agent.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """What we get back from one model call."""

    text: str  # the model's answer (plain text or a JSON string)
    prompt_tokens: int  # tokens we SENT (system prompt + memories + messages)
    completion_tokens: int  # tokens the model WROTE
    latency_seconds: float  # wall-clock time of the call

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageTracker:
    """
    Adds up tokens / cost / time per "purpose".

    Purposes we use: "answer" (replying to the user), "summarize",
    "extract" (fact extraction) and "decide" (add/update/delete decisions).
    Splitting by purpose lets the benchmark separate the cost of ANSWERING
    from the cost of LEARNING.
    """

    price_input_per_1m: float = 0.0
    price_output_per_1m: float = 0.0
    calls: dict = field(default_factory=lambda: defaultdict(int))
    prompt_tokens: dict = field(default_factory=lambda: defaultdict(int))
    completion_tokens: dict = field(default_factory=lambda: defaultdict(int))
    seconds: dict = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, purpose: str, response: LLMResponse) -> None:
        """Store the numbers of one call. Thread-safe (learning may run in a thread)."""
        with self._lock:
            self.calls[purpose] += 1
            self.prompt_tokens[purpose] += response.prompt_tokens
            self.completion_tokens[purpose] += response.completion_tokens
            self.seconds[purpose] += response.latency_seconds

    def cost_usd(self, purpose: str | None = None) -> float:
        """Estimated cost in US dollars, for one purpose or for everything."""
        purposes = [purpose] if purpose else list(self.calls.keys())
        p_in = sum(self.prompt_tokens[p] for p in purposes)
        p_out = sum(self.completion_tokens[p] for p in purposes)
        return (p_in / 1_000_000) * self.price_input_per_1m + (p_out / 1_000_000) * self.price_output_per_1m

    def summary(self) -> dict:
        """A plain dict that is easy to print or save as JSON."""
        with self._lock:
            purposes = sorted(self.calls.keys())
            per_purpose = {
                p: {
                    "calls": self.calls[p],
                    "prompt_tokens": self.prompt_tokens[p],
                    "completion_tokens": self.completion_tokens[p],
                    "seconds": round(self.seconds[p], 3),
                }
                for p in purposes
            }
        return {
            "per_purpose": per_purpose,
            "total_calls": sum(self.calls.values()),
            "total_prompt_tokens": sum(self.prompt_tokens.values()),
            "total_completion_tokens": sum(self.completion_tokens.values()),
            "total_cost_usd": round(self.cost_usd(), 6),
        }

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self.prompt_tokens.clear()
            self.completion_tokens.clear()
            self.seconds.clear()


class LLMClient:
    """
    Sends chat messages to the model and returns an `LLMResponse`.

    Usage:
        llm = LLMClient(settings)
        reply = llm.chat([{"role": "user", "content": "Hi"}]).text
    """

    def __init__(self, settings: Settings, tracker: UsageTracker | None = None):
        self.settings = settings
        self.model = settings.llm_model
        self.tracker = tracker or UsageTracker(settings.price_input_per_1m, settings.price_output_per_1m)

        if settings.llm_provider == "ollama":
            # Ollama exposes an OpenAI-compatible server at <host>/v1.
            # It ignores the API key, but the SDK insists on having one.
            self._client = OpenAI(api_key="ollama", base_url=f"{settings.ollama_base_url}/v1")
        else:
            self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
        purpose: str = "answer",
    ) -> LLMResponse:
        """
        Call the model once.

        messages    : list of {"role": "system"|"user"|"assistant", "content": "..."}
        json_mode   : ask the model to answer with a JSON object only
        temperature : 0 = deterministic, 1 = creative (memory pipeline uses low values)
        purpose     : label used by the UsageTracker
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Both OpenAI and Ollama support this flag; it forces valid JSON output.
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Some newer "reasoning" models reject the classic parameters. Adapt once and retry
            # instead of failing, so switching models in .env keeps working.
            message = str(exc)
            if "max_tokens" in message and "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif "temperature" in message and "temperature" in kwargs:
                kwargs.pop("temperature")
            else:
                raise
            logger.warning("Model rejected a parameter (%s); retrying with adjusted settings", message[:80])
            completion = self._client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started

        text = (completion.choices[0].message.content or "").strip()
        usage = completion.usage
        response = LLMResponse(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=elapsed,
        )
        self.tracker.record(purpose, response)
        logger.debug(
            "LLM call purpose=%s prompt_tokens=%s completion_tokens=%s latency=%.2fs",
            purpose,
            response.prompt_tokens,
            response.completion_tokens,
            elapsed,
        )
        return response
