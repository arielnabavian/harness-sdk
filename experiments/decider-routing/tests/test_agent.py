import json

import pytest
import strands_harness.agent as harness_agent
import strands_harness.memory as harness_memory
from strands.models.routing import ModelRouter, RoutingAttempt, RoutingCandidate, RoutingContext

import agent

FUNCTION_URL = "https://abc123.lambda-url.us-east-1.on.aws/"
STRONG_MODEL = "global.anthropic.claude-opus-5"
CHEAP_MODEL = "global.anthropic.claude-haiku-5-5"
SUMMARIZER_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture
def harness_calls(monkeypatch):
    calls = []
    real_create_harness = agent.create_harness

    def spy(**kwargs):
        calls.append(kwargs)
        return real_create_harness(**kwargs)

    monkeypatch.setattr(agent, "create_harness", spy)
    return calls


@pytest.fixture
def summarizer_models(monkeypatch):
    models = {"web_fetch": [], "memory": []}

    def spy_on(module, key):
        real = module.resolve_web_fetch_model

        def spy(*args, **kwargs):
            model = real(*args, **kwargs)
            models[key].append(model)
            return model

        monkeypatch.setattr(module, "resolve_web_fetch_model", spy)

    spy_on(harness_agent, "web_fetch")
    spy_on(harness_memory, "memory")
    return models


def _routed_model(harness_calls) -> ModelRouter:
    return harness_calls[-1]["model"]


def test_should_route_between_strong_and_cheap_models_when_decider_builds_agent(monkeypatch, harness_calls):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)

    agent.Decider(FUNCTION_URL).create_agent()

    router = _routed_model(harness_calls)
    assert isinstance(router, ModelRouter)
    assert [candidate.name for candidate in router.candidates] == ["strong", "cheap"]
    assert [candidate.model.config["model_id"] for candidate in router.candidates] == [STRONG_MODEL, CHEAP_MODEL]
    assert router.default_model is router.candidates[0].model


@pytest.mark.parametrize("effort", [None, "high", "low"])
def test_should_build_strong_candidate_like_baseline_model_when_effort_varies(monkeypatch, harness_calls, effort):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    if effort:
        monkeypatch.setenv("HARBOR_REASONING_EFFORT", effort)

    baseline = agent.Baseline().create_agent()
    agent.Decider(FUNCTION_URL).create_agent()

    strong = _routed_model(harness_calls).candidates[0].model
    assert strong.config == baseline.model.config


def test_should_keep_baseline_settings_other_than_model_when_decider_builds_agent(monkeypatch, harness_calls):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    monkeypatch.setenv("STRANDSHARNESS_SEQUENTIAL_TOOLS", "1")

    agent.Baseline().create_agent()
    agent.Decider(FUNCTION_URL).create_agent()

    baseline_kwargs, decider_kwargs = harness_calls
    assert set(decider_kwargs) - set(baseline_kwargs) == {"builtin_tools"}
    for name in set(baseline_kwargs) - {"model", "tool_executor"}:
        assert decider_kwargs[name] == baseline_kwargs[name]
    assert type(decider_kwargs["tool_executor"]) is type(baseline_kwargs["tool_executor"])


def test_should_give_side_calls_the_baseline_summarizer_when_decider_builds_agent(monkeypatch, summarizer_models):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)

    agent.Baseline().create_agent()
    agent.Decider(FUNCTION_URL).create_agent()

    for key in ("web_fetch", "memory"):
        baseline_model, decider_model = summarizer_models[key]
        assert decider_model.config == baseline_model.config
        assert decider_model.config["model_id"] == SUMMARIZER_MODEL


def test_should_keep_mantle_web_search_off_when_decider_runs_on_mantle_anthropic_endpoint(monkeypatch, harness_calls):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/anthropic")

    agent.Decider(FUNCTION_URL).create_agent()

    builtin_tools = harness_calls[-1]["builtin_tools"]
    assert builtin_tools["web_search"] is False
    assert builtin_tools["web_fetch"]["model"].config["model_id"] == SUMMARIZER_MODEL


def test_should_keep_web_fetch_settings_when_overrides_configure_web_fetch(monkeypatch, harness_calls):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    monkeypatch.setenv(
        "STRANDSHARNESS_AGENT_CONFIG", json.dumps({"builtin_tools": {"web_fetch": {"transport": "direct"}}})
    )

    agent.Decider(FUNCTION_URL).create_agent()

    web_fetch = harness_calls[-1]["builtin_tools"]["web_fetch"]
    assert web_fetch["transport"] == "direct"
    assert web_fetch["model"].config["model_id"] == SUMMARIZER_MODEL


def test_should_keep_override_summarizer_when_overrides_name_a_web_fetch_model(monkeypatch, harness_calls):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    override = {"builtin_tools": {"web_fetch": {"model": f"bedrock/{CHEAP_MODEL}"}}}
    monkeypatch.setenv("STRANDSHARNESS_AGENT_CONFIG", json.dumps(override))

    agent.Decider(FUNCTION_URL).create_agent()

    assert harness_calls[-1]["builtin_tools"]["web_fetch"] == {"model": f"bedrock/{CHEAP_MODEL}"}


@pytest.mark.parametrize(
    ("builtin_tools", "message"),
    [
        ({"web_fetch": False}, "needs web_fetch on"),
        ({"*": False, "read": True}, "needs web_fetch on"),
        (["read", "web_fetch"], "as a mapping only"),
    ],
)
def test_should_raise_when_overrides_leave_side_models_unmatched(monkeypatch, builtin_tools, message):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    monkeypatch.setenv("STRANDSHARNESS_AGENT_CONFIG", json.dumps({"builtin_tools": builtin_tools}))

    with pytest.raises(ValueError, match=message):
        agent.Decider(FUNCTION_URL).create_agent()


def test_should_raise_when_cheap_model_has_a_smaller_context_window(monkeypatch):
    monkeypatch.setenv("STRANDS_MODEL", STRONG_MODEL)
    monkeypatch.setattr(agent, "_CHEAP_MODEL_BY_FAMILY", {"anthropic.claude-": SUMMARIZER_MODEL})

    with pytest.raises(ValueError, match="context window"):
        agent.Decider(FUNCTION_URL).create_agent()


def test_should_raise_when_function_url_is_empty():
    with pytest.raises(ValueError, match="DECIDER_FUNCTION_URL"):
        agent.Decider("")


@pytest.mark.parametrize("model", ["anthropic/claude-fable-5", "openai/gpt-5.6-sol"])
def test_should_raise_when_model_is_not_a_bedrock_id(monkeypatch, model):
    monkeypatch.setenv("STRANDS_MODEL", model)

    with pytest.raises(ValueError, match="Bedrock model ids only"):
        agent.Decider(FUNCTION_URL).create_agent()


def test_should_accept_bedrock_provider_prefix_when_model_names_it(monkeypatch, harness_calls):
    monkeypatch.setenv("STRANDS_MODEL", f"bedrock/{STRONG_MODEL}")

    agent.Decider(FUNCTION_URL).create_agent()

    assert _routed_model(harness_calls).candidates[0].model.config["model_id"] == STRONG_MODEL


def test_should_raise_when_model_family_has_no_cheap_model(monkeypatch):
    monkeypatch.setenv("STRANDS_MODEL", "qwen.qwen3-coder-480b-a35b-v1:0")

    with pytest.raises(ValueError, match="no cheap model"):
        agent.Decider(FUNCTION_URL).create_agent()


class _CountingStrategy:
    def __init__(self, route):
        self._route = route
        self.asks = 0

    async def select(self, context, **kwargs):
        self.asks += 1
        return next((candidate for candidate in context.candidates if candidate.name == self._route), None)


def _context(failed: bool = False) -> RoutingContext:
    candidates = [RoutingCandidate(object(), "strong"), RoutingCandidate(object(), "cheap")]
    attempts = [RoutingAttempt(candidate=candidates[1], exception=RuntimeError("throttled"))] if failed else []
    return RoutingContext(
        messages=[], system_prompt=None, tool_specs=[], candidates=candidates, invocation_state={}, attempts=attempts
    )


def _named(context: RoutingContext, name: str) -> RoutingCandidate:
    return next(candidate for candidate in context.candidates if candidate.name == name)


@pytest.mark.parametrize("route", ["cheap", "strong"])
async def test_should_ask_once_and_keep_the_route_when_the_trial_invokes_again(tmp_path, route):
    inner = _CountingStrategy(route)
    pinned = agent._PinnedRoute(inner, tmp_path / "decider-route.json")
    first, later = _context(), _context()

    assert await pinned.select(first) is _named(first, route)
    assert await pinned.select(later) is _named(later, route)
    assert inner.asks == 1


async def test_should_record_the_route_once_when_the_trial_decides(tmp_path):
    record = tmp_path / "decider-route.json"
    pinned = agent._PinnedRoute(_CountingStrategy("cheap"), record)

    await pinned.select(_context())
    await pinned.select(_context())

    recorded = json.loads(record.read_text())
    assert recorded["route"] == "cheap"
    assert recorded["decision_ms"] >= 0


async def test_should_serve_the_default_model_and_record_null_when_the_decider_declines(tmp_path):
    record = tmp_path / "decider-route.json"
    inner = _CountingStrategy(None)
    pinned = agent._PinnedRoute(inner, record)

    assert (await pinned.select(_context())) is None
    assert (await pinned.select(_context())) is None
    assert json.loads(record.read_text())["route"] is None
    assert inner.asks == 1


async def test_should_decline_without_asking_when_a_model_call_failed(tmp_path):
    inner = _CountingStrategy("cheap")
    pinned = agent._PinnedRoute(inner, tmp_path / "decider-route.json")

    assert await pinned.select(_context(failed=True)) is None
    assert inner.asks == 0


async def test_should_skip_the_record_when_the_harbor_log_folder_is_absent(tmp_path):
    pinned = agent._PinnedRoute(_CountingStrategy("cheap"), tmp_path / "missing" / "decider-route.json")

    assert (await pinned.select(_context())).name == "cheap"
    assert not (tmp_path / "missing").exists()
