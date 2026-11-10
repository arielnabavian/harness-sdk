import base64
import json
import os
from functools import partial

from strands.tools.executors import SequentialToolExecutor

_MANTLE_ROUTE_PREFIX = "openai."
_MANTLE_OPENAI_VENDOR_PREFIX = "openai.gpt"
_MANTLE_HOST = "bedrock-mantle."
_MANTLE_ANTHROPIC_BUILTIN_TOOLS = {"web_search": False}


def prepare_harness_kwargs() -> dict:
    task_workdir = os.environ.get("HARBOR_TASK_WORKDIR")
    if task_workdir:
        os.chdir(task_workdir)
    overrides = _agent_overrides()
    if os.environ.get("HARBOR_REASONING_EFFORT"):
        overrides["effort"] = os.environ["HARBOR_REASONING_EFFORT"]
    model = overrides.pop("model", None) or (_model_string() if overrides else _resolve_model())
    executor = _tool_executor()
    kwargs = {
        "model": model,
        **({"tool_executor": executor} if executor is not None else {}),
        **({"builtin_tools": dict(_MANTLE_ANTHROPIC_BUILTIN_TOOLS)} if _on_mantle_anthropic_endpoint() else {}),
        "instructions": _task_instructions(task_workdir),
        "tools": _mcp_tools(),
    }
    overrides.pop("tools", None)
    overrides.pop("instructions", None)
    kwargs.update(overrides)
    return kwargs


def _task_instructions(task_workdir: str | None) -> str | None:
    if not task_workdir:
        return None
    return (
        f"You are working inside a benchmark task container. The task root is {task_workdir}. "
        "Run `pwd` before acting, resolve task files relative to that root, and use absolute paths "
        "when writing or editing files."
    )


def _mantle_endpoint_model_id(model_id: str) -> str:
    if model_id.startswith(_MANTLE_OPENAI_VENDOR_PREFIX):
        return model_id
    return model_id.removeprefix(_MANTLE_ROUTE_PREFIX)


def _resolve_model():
    model_id = os.environ.get("STRANDS_MODEL")
    if not model_id:
        return None
    if model_id.startswith(_MANTLE_ROUTE_PREFIX):
        mantle_config = {"region": os.environ.get("AWS_REGION", "us-east-1")}
        if "gpt-5" in model_id:
            from strands.models.openai_responses import OpenAIResponsesModel

            return OpenAIResponsesModel(model_id=model_id, bedrock_mantle_config=mantle_config)
        else:
            from strands.models.openai import OpenAIModel

            return OpenAIModel(model_id=_mantle_endpoint_model_id(model_id), bedrock_mantle_config=mantle_config)
    return model_id


def _agent_overrides() -> dict:
    raw = os.environ.get("STRANDSHARNESS_AGENT_CONFIG")
    if not raw and os.environ.get("STRANDSHARNESS_AGENT_CONFIG_B64"):
        raw = base64.b64decode(os.environ["STRANDSHARNESS_AGENT_CONFIG_B64"]).decode()
    if not raw:
        return {}
    overrides = json.loads(raw)
    if not isinstance(overrides, dict):
        raise ValueError("STRANDSHARNESS_AGENT_CONFIG must be a JSON object of create_harness kwargs")
    return overrides


def _model_string() -> str | None:
    model_id = os.environ.get("STRANDS_MODEL")
    if not model_id:
        return None
    if "/" in model_id:
        return model_id
    if not model_id.startswith(_MANTLE_ROUTE_PREFIX):
        return model_id
    return f"bedrock-mantle/{_mantle_endpoint_model_id(model_id)}"


def _on_mantle_anthropic_endpoint() -> bool:
    return _MANTLE_HOST in os.environ.get("ANTHROPIC_BASE_URL", "")


def _tool_executor():
    if not os.environ.get("STRANDSHARNESS_SEQUENTIAL_TOOLS"):
        return None
    return SequentialToolExecutor()


def _mcp_tools() -> list:
    raw_servers = os.environ.get("HARBOR_MCP_SERVERS")
    if not raw_servers:
        return []

    from mcp import StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client

    try:
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
    from strands.tools.mcp import MCPClient

    clients = []
    for server in json.loads(raw_servers):
        transport = server["transport"]
        name = server.get("name") or "<unnamed>"
        if transport == "stdio":
            parameters = StdioServerParameters(
                command=server["command"],
                args=server.get("args", []),
            )
            transport_factory = partial(stdio_client, parameters)
        elif transport == "sse":
            url = server["url"]
            transport_factory = partial(sse_client, url)
        elif transport == "streamable-http":
            url = server["url"]
            transport_factory = partial(streamablehttp_client, url=url)
        else:
            raise ValueError(f"Unsupported MCP transport: {transport}")
        client = MCPClient(transport_factory)
        client.start()
        listed = client.list_tools_sync()
        if not listed:
            client.stop(None, None, None)
            raise RuntimeError(
                f"MCP server {name!r} ({transport}) connected but reported zero tools. "
                f"Refusing to launch: the model would see none of the MCP tools it was "
                f"configured with and silently fall back to built-in tools, changing what "
                f"the benchmark actually measures. Verify the server's tool exposure."
            )
        client._tool_provider_started = True
        clients.append(client)
    return clients
