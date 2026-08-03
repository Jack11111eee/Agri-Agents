from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeSDKClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def completion_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_protocol_import_does_not_load_provider_sdk():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.llm.protocol; "
                "assert 'openai' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.fixture
def llm_modules():
    import app.llm as public_module
    import app.llm.client as client_module

    yield SimpleNamespace(public=public_module, client=client_module)

    for module_name in tuple(sys.modules):
        if (
            module_name == "app.llm"
            or module_name.startswith("app.llm.")
            or module_name == "openai"
            or module_name.startswith("openai.")
        ):
            sys.modules.pop(module_name, None)


def test_injected_client_requests_deepseek_json_mode_without_environment_key(
    monkeypatch,
    llm_modules,
):
    llm = llm_modules.public
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    completions = FakeCompletions(completion_response('{"model_suggestions": []}'))
    client = llm.DeepSeekClient(sdk_client=FakeSDKClient(completions))

    assert isinstance(client, llm.LLMClient)
    assert client.generate([{"role": "system", "content": "Return JSON."}]) == (
        '{"model_suggestions": []}'
    )
    assert completions.calls == [
        {
            "model": llm.DEFAULT_DEEPSEEK_MODEL,
            "messages": [{"role": "system", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
        }
    ]


def test_default_client_reads_key_from_environment_and_sets_transport_limits(
    monkeypatch,
    llm_modules,
):
    client_module = llm_modules.client
    captured: dict[str, Any] = {}
    fake_sdk = FakeSDKClient(FakeCompletions(completion_response("{}")))

    def fake_openai(**kwargs: Any) -> FakeSDKClient:
        captured.update(kwargs)
        return fake_sdk

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(client_module, "OpenAI", fake_openai)

    client_module.DeepSeekClient()

    assert captured == {
        "api_key": "test-key",
        "base_url": client_module.DEEPSEEK_BASE_URL,
        "timeout": client_module.DEFAULT_TIMEOUT_SECONDS,
        "max_retries": client_module.DEFAULT_MAX_RETRIES,
    }


def test_missing_environment_key_is_rejected_before_sdk_construction(
    monkeypatch,
    llm_modules,
):
    llm = llm_modules.public
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(llm.LLMConfigurationError, match="DEEPSEEK_API_KEY is required"):
        llm.DeepSeekClient()


def test_blank_model_is_rejected_before_sdk_call(llm_modules):
    llm = llm_modules.public
    completions = FakeCompletions(completion_response("{}"))

    with pytest.raises(llm.LLMConfigurationError, match="model must not be blank"):
        llm.DeepSeekClient(
            sdk_client=FakeSDKClient(completions),
            model="  ",
        )

    assert completions.calls == []


@pytest.mark.parametrize(
    "messages, expected_error",
    [
        ([], "At least one LLM message is required"),
        ([{"role": "assistant", "content": "{}"}], "Unsupported LLM message role"),
        ([{"role": "user", "content": "  "}], "LLM message content must not be blank"),
    ],
)
def test_invalid_messages_are_rejected_without_calling_sdk(
    messages,
    expected_error,
    llm_modules,
):
    completions = FakeCompletions(completion_response("{}"))
    client = llm_modules.public.DeepSeekClient(
        sdk_client=FakeSDKClient(completions)
    )

    with pytest.raises(ValueError, match=expected_error):
        client.generate(messages)

    assert completions.calls == []


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        completion_response("  "),
        SimpleNamespace(choices=[SimpleNamespace(message=None)]),
    ],
)
def test_malformed_provider_response_raises_explicit_error(response, llm_modules):
    llm = llm_modules.public
    client = llm.DeepSeekClient(sdk_client=FakeSDKClient(FakeCompletions(response)))

    with pytest.raises(llm.LLMResponseError):
        client.generate([{"role": "user", "content": "Return JSON."}])


def test_provider_transport_error_bubbles_unchanged(llm_modules):
    timeout = TimeoutError("provider timed out")
    client = llm_modules.public.DeepSeekClient(
        sdk_client=FakeSDKClient(FakeCompletions(error=timeout))
    )

    with pytest.raises(TimeoutError) as captured:
        client.generate([{"role": "user", "content": "Return JSON."}])

    assert captured.value is timeout
