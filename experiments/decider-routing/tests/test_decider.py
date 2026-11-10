import json

import botocore.session
import httpx
import pytest

from decider import LambdaDecider

FUNCTION_URL = "https://abc123.lambda-url.us-east-1.on.aws/"
QUESTIONS = {"candidate": {"question": "Which model fits the task?", "choices": ["c0", "c1"], "uncertainty": False}}
DECIDER_REPLY = {
    "model": "strands-decider-2B-qwen3.5-v1-2610-q8_0",
    "answers": {
        "candidate": {"type": "choice", "choice": "c1", "probabilities": {"c0": 0.2, "c1": 0.8}, "confidence": 0.6}
    },
    "usage": {"input_tokens": 42, "output_tokens": 1},
}


@pytest.fixture
def sent_requests(monkeypatch):
    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json=DECIDER_REPLY)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(reply), **kwargs)
    )
    return requests


@pytest.mark.parametrize("region", ["us-east-1", "eu-west-1"])
async def test_should_sign_request_for_lambda_in_url_region_when_asking(sent_requests, region):
    function_url = f"https://abc123.lambda-url.{region}.on.aws/"

    await LambdaDecider(function_url).ask("Fix the failing build.", QUESTIONS)

    (request,) = sent_requests
    authorization = request.headers["Authorization"]
    assert request.method == "POST"
    assert str(request.url) == f"{function_url}v1/systemone"
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert f"/{region}/lambda/aws4_request" in authorization
    assert request.headers["X-Amz-Security-Token"] == "session-token"
    assert "Bearer" not in authorization


@pytest.mark.parametrize("function_url", [FUNCTION_URL, FUNCTION_URL.rstrip("/")])
async def test_should_post_to_system_one_route_when_url_has_or_lacks_trailing_slash(sent_requests, function_url):
    await LambdaDecider(function_url).ask("Fix the failing build.", QUESTIONS)

    (request,) = sent_requests
    assert str(request.url) == "https://abc123.lambda-url.us-east-1.on.aws/v1/systemone"


async def test_should_send_system_one_body_when_asking(sent_requests):
    await LambdaDecider(FUNCTION_URL).ask("Fix the failing build.", QUESTIONS)

    (request,) = sent_requests
    assert json.loads(request.content) == {
        "state": "Fix the failing build.",
        "model": "strands-decider-latest",
        "questions": {
            "candidate": {
                "type": "choice",
                "instructions": "Which model fits the task?",
                "criteria": {"c0": None, "c1": None},
            }
        },
    }


async def test_should_return_decider_choice_when_decider_replies(sent_requests):
    result = await LambdaDecider(FUNCTION_URL).ask("Fix the failing build.", QUESTIONS)

    assert result.answers == {"candidate": "c1"}
    assert result.usage == {"inputTokens": 42, "outputTokens": 1, "totalTokens": 43}
    assert result.metadata.probabilities == {"candidate": {"c0": 0.2, "c1": 0.8}}


@pytest.mark.parametrize(
    "function_url",
    [
        "https://example.com/v1/systemone",
        "https://abc123.lambda-url.us-east-1.amazonaws.com/",
        "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/decider/invocations",
    ],
)
def test_should_raise_when_url_is_not_a_lambda_function_url(function_url):
    with pytest.raises(ValueError, match="lambda-url"):
        LambdaDecider(function_url)


async def test_should_raise_when_no_aws_credentials_are_found(monkeypatch, sent_requests):
    monkeypatch.setattr(botocore.session.Session, "get_credentials", lambda self: None)

    with pytest.raises(RuntimeError, match="no AWS credentials"):
        await LambdaDecider(FUNCTION_URL).ask("Fix the failing build.", QUESTIONS)

    assert sent_requests == []
