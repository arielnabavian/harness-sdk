import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from strands import Agent
from strands.models import Model
from strands.models.routing import ClassifierStrategy, ModelRouter, RoutingCandidate, RoutingContext
from strands_harness import create_harness, defaults
from strands_harness.models import resolve_model, resolve_web_fetch_model

from decider import LambdaDecider
from harbor_wrapper import prepare_harness_kwargs

DECIDER_FUNCTION_URL = ""

_ROUTE_RECORD = Path("/logs/agent/decider-route.json")
_BEDROCK_PROVIDER_PREFIX = "bedrock/"
_BEDROCK_REGION_PREFIXES = ("global.", "apac.", "us.", "eu.", "au.", "jp.")
_CHEAP_MODEL_BY_FAMILY = {"anthropic.claude-": "global.anthropic.claude-haiku-5-5"}
_STRONG_DESCRIPTION = "The most capable and most expensive model. Best for long, hard, or unclear tasks."
_CHEAP_DESCRIPTION = "A fast, low-cost model. Good for short tasks with clear steps."
_ROUTING_POLICY = (
    "Pick the cheap model when the task is short and clear and needs few steps. "
    "Pick the strong model when the task needs many steps, debugging, or careful reasoning."
)
_DECISION_TIMEOUT_SECONDS = 60.0
_AGENT_PROMPT_CHARS_FOR_ROUTING = 500


class Baseline:
    def create_agent(self) -> Agent:
        return create_harness(**prepare_harness_kwargs())


class Decider:
    def __init__(self, function_url: str = DECIDER_FUNCTION_URL) -> None:
        if not function_url:
            raise ValueError("set DECIDER_FUNCTION_URL in agent.py to the decider's Lambda function URL")
        self._strategy = _PinnedRoute(
            ClassifierStrategy(
                LambdaDecider(function_url),
                routing_policy=_ROUTING_POLICY,
                timeout=_DECISION_TIMEOUT_SECONDS,
                max_agent_instructions_chars=_AGENT_PROMPT_CHARS_FOR_ROUTING,
            ),
            _ROUTE_RECORD,
        )

    def create_agent(self) -> Agent:
        kwargs = prepare_harness_kwargs()
        strong_id = _bedrock_model_id(kwargs["model"])
        strong = _baseline_built_model(strong_id, kwargs)
        router = ModelRouter(
            [
                RoutingCandidate(strong, "strong", _STRONG_DESCRIPTION),
                RoutingCandidate(_cheap_model(strong_id, strong, kwargs), "cheap", _CHEAP_DESCRIPTION),
            ],
            strategy=self._strategy,
        )
        builtin_tools = _with_baseline_summarizer(kwargs.get("builtin_tools"), resolve_web_fetch_model(strong_id, None))
        return create_harness(**{**kwargs, "model": router, "builtin_tools": builtin_tools})


class _PinnedRoute:
    def __init__(self, strategy: ClassifierStrategy, record: Path) -> None:
        self._strategy = strategy
        self._record = record
        self._decided = False
        self._route: str | None = None

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        if context.attempts:
            return None
        if not self._decided:
            self._route = await self._decide(context, **kwargs)
            self._decided = True
        if self._route is None:
            return None
        return next((candidate for candidate in context.candidates if candidate.name == self._route), None)

    async def _decide(self, context: RoutingContext, **kwargs: Any) -> str | None:
        started = time.monotonic()
        candidate = await self._strategy.select(context, **kwargs)
        route = candidate.name if candidate else None
        if self._record.parent.is_dir():
            decision_ms = round((time.monotonic() - started) * 1000)
            self._record.write_text(json.dumps({"route": route, "decision_ms": decision_ms}))
        return route


def _bedrock_model_id(model: Any) -> str:
    if isinstance(model, str) and model.startswith(_BEDROCK_PROVIDER_PREFIX):
        return model.removeprefix(_BEDROCK_PROVIDER_PREFIX)
    if isinstance(model, str) and "/" not in model:
        return model
    raise ValueError(f"model=<{model}> | the decider arm runs on Bedrock model ids only")


def _cheap_model(strong_id: str, strong: Model, kwargs: dict) -> Model:
    family = next(
        (strong_id[len(prefix) :] for prefix in _BEDROCK_REGION_PREFIXES if strong_id.startswith(prefix)), strong_id
    )
    cheap_id = next((cheap for prefix, cheap in _CHEAP_MODEL_BY_FAMILY.items() if family.startswith(prefix)), None)
    if cheap_id is None:
        raise ValueError(f"model=<{strong_id}> | no cheap model is set for this model family")
    cheap = _baseline_built_model(cheap_id, kwargs)
    strong_window, cheap_window = strong.context_window_limit, cheap.context_window_limit
    if strong_window is None or cheap_window is None or cheap_window < strong_window:
        raise ValueError(
            f"strong=<{strong_id}>, strong_window=<{strong_window}>, cheap=<{cheap_id}>, cheap_window=<{cheap_window}>"
            " | the cheap model must have a known context window at least as large as the strong model's"
        )
    return cheap


def _baseline_built_model(model_id: str, kwargs: dict) -> Model:
    caching_explicit = "caching" in kwargs
    return resolve_model(
        model_id,
        defaults.DEFAULT_MODEL,
        kwargs.get("effort", defaults.DEFAULT_EFFORT),
        web_search=False,
        caching=bool(kwargs["caching"]) if caching_explicit else bool(defaults.DEFAULT_CACHING),
        caching_explicit=caching_explicit,
    )


def _with_baseline_summarizer(builtin_tools: Any, summarizer: Model) -> dict:
    if builtin_tools is None:
        return {"web_fetch": {"model": summarizer}}
    if not isinstance(builtin_tools, Mapping):
        raise ValueError("the decider arm takes builtin_tools as a mapping only")
    web_fetch = builtin_tools.get("web_fetch", builtin_tools.get("*", True))
    if web_fetch is False:
        raise ValueError("the decider arm needs web_fetch on, so that memory extraction keeps the baseline summarizer")
    configured = web_fetch if isinstance(web_fetch, Mapping) else {}
    return {**builtin_tools, "web_fetch": {"model": summarizer, **configured}}
