from collections.abc import Generator
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import get_session
from strands.decisions import DecisionModel, DecisionResult
from strands.decisions._system_one_http_adapter import _ask_system_one, _SystemOneProviderConfig
from strands.decisions.types import DecisionInput, Question

_PROVIDER_NAME = "strands-decider"
_SYSTEM_ONE_PATH = "v1/systemone"
_DECIDER_MODEL = "strands-decider-latest"
_LAMBDA_PAYLOAD_LIMIT_BYTES = 6 * 1024 * 1024
_HTTP_TIMEOUT_SECONDS = 120.0
_BEARER_REPLACED_BY_SIGV4 = "unused-sigv4-signs-each-request"
_NO_UNCERTAINTY_FLOOR = 0.0


class LambdaDecider(DecisionModel):
    def __init__(self, function_url: str) -> None:
        super().__init__()
        self._endpoint = urljoin(function_url, _SYSTEM_ONE_PATH)
        self._signer = _LambdaSigV4(_region_of(function_url))

    async def _ask(self, state: DecisionInput, questions: dict[str, Question], **kwargs: Any) -> DecisionResult:
        async with httpx.AsyncClient(auth=self._signer, timeout=_HTTP_TIMEOUT_SECONDS) as client:
            config = _SystemOneProviderConfig(
                provider_name=_PROVIDER_NAME,
                endpoint=self._endpoint,
                api_key=_BEARER_REPLACED_BY_SIGV4,
                model_id=_DECIDER_MODEL,
                default_min_confidence=_NO_UNCERTAINTY_FLOOR,
                max_body_bytes=_LAMBDA_PAYLOAD_LIMIT_BYTES,
                http_client=client,
            )
            return await _ask_system_one(config, state, questions)


class _LambdaSigV4(httpx.Auth):
    requires_request_body = True

    def __init__(self, region: str) -> None:
        self._region = region
        self._session = get_session()

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        credentials = self._session.get_credentials()
        if credentials is None:
            raise RuntimeError("no AWS credentials found to sign the decider request")
        signed = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={"Content-Type": request.headers["Content-Type"]},
        )
        SigV4Auth(credentials.get_frozen_credentials(), "lambda", self._region).add_auth(signed)
        request.headers.update(dict(signed.headers.items()))
        yield request


def _region_of(function_url: str) -> str:
    host = urlparse(function_url).hostname or ""
    labels = host.split(".")
    if len(labels) != 5 or labels[1] != "lambda-url" or labels[3:] != ["on", "aws"]:
        raise ValueError(f"function_url=<{function_url}> | expected https://<id>.lambda-url.<region>.on.aws/")
    return labels[2]
