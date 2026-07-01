import pytest
from pydantic import ValidationError

from whispermlx_server.models import InferenceRequest


def test_inference_request_required_fields() -> None:
    req = InferenceRequest(
        file_path="/tmp/audio.mp4",
        model="large-v3",
        language="ru",
        response_format="json",
    )
    assert req.file_path == "/tmp/audio.mp4"
    assert req.diarize_enabled() is False


def test_inference_request_rejects_non_json_response_format() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(
            file_path="/tmp/audio.mp4",
            model="large-v3",
            language="en",
            response_format="text",  # type: ignore[arg-type]
        )


def test_initial_prompt_only_for_ru() -> None:
    ru_req = InferenceRequest(
        file_path="/tmp/audio.mp4",
        model="large-v3",
        language="ru",
        response_format="json",
        initial_prompt="Контекст",
    )
    en_req = InferenceRequest(
        file_path="/tmp/audio.mp4",
        model="large-v3",
        language="en",
        response_format="json",
        initial_prompt="Context",
    )
    assert ru_req.effective_initial_prompt() == "Контекст"
    assert en_req.effective_initial_prompt() is None


def test_auto_language() -> None:
    req = InferenceRequest(
        file_path="/tmp/audio.mp4",
        model="large-v3",
        language="auto",
        response_format="json",
    )
    assert req.language == "auto"


def test_empty_hotwords_become_none() -> None:
    req = InferenceRequest(
        file_path="/tmp/audio.mp4",
        model="large-v3",
        language="ru",
        response_format="json",
        hotwords="   ",
    )
    assert req.hotwords is None


def test_diarize_enabled() -> None:
    req = InferenceRequest(
        file_path="/tmp/audio.mp4",
        model="large-v3",
        language="ru",
        response_format="json",
        diarize=True,
    )
    assert req.diarize_enabled() is True
