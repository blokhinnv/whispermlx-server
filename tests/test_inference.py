import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from whispermlx_server.config import Settings
from whispermlx_server.inference import (
    InferenceError,
    _subprocess_env,
    build_argv,
    extract_text,
    run_inference_sync,
)
from whispermlx_server.models import InferenceRequest

SAMPLE_RESULT = {
    "segments": [
        {"text": " Hello.", "speaker": "SPEAKER_00"},
        {"text": " Hi there.", "speaker": "SPEAKER_01"},
    ],
    "language": "en",
}

PLAIN_RESULT = {
    "segments": [
        {"text": " Plain transcript."},
    ],
    "language": "en",
}


def test_subprocess_env_prepends_homebrew_bins() -> None:
    with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=False):
        with patch("whispermlx_server.inference.os.path.isdir", return_value=True):
            env = _subprocess_env()
    assert env["PATH"].startswith("/opt/homebrew/bin:/usr/local/bin:/usr/bin")


def test_extract_text_with_speakers() -> None:
    text = extract_text(SAMPLE_RESULT)
    assert text == "SPEAKER_00: Hello.\nSPEAKER_01: Hi there."


def test_extract_text_without_speakers() -> None:
    text = extract_text(PLAIN_RESULT)
    assert text == "Plain transcript."


def test_extract_text_empty_segments() -> None:
    assert extract_text({"segments": []}) == ""


def test_build_argv_basic(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"fake")
    request = InferenceRequest(
        file_path=str(audio),
        model="large-v3",
        language="auto",
        response_format="json",
    )
    with patch(
        "whispermlx_server.inference.shutil.which", return_value="/usr/bin/whispermlx"
    ):
        argv = build_argv(request, output_dir=tmp_path / "out", hf_token=None)
    assert argv[0] == "/usr/bin/whispermlx"
    assert str(audio) in argv
    assert "--model" in argv and "large-v3" in argv
    assert "--language" not in argv
    assert "--diarize" not in argv


def test_build_argv_ru_initial_prompt_and_hotwords(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"fake")
    request = InferenceRequest(
        file_path=str(audio),
        model="large-v3",
        language="ru",
        response_format="json",
        hotwords="ЦИАРС",
        initial_prompt="Встреча",
    )
    with patch(
        "whispermlx_server.inference.shutil.which", return_value="/usr/bin/whispermlx"
    ):
        argv = build_argv(request, output_dir=tmp_path / "out", hf_token="hf_test")
    assert "--hotwords" in argv and "ЦИАРС" in argv
    assert "--initial_prompt" in argv and "Встреча" in argv


def test_build_argv_diarize_requires_token(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"fake")
    request = InferenceRequest(
        file_path=str(audio),
        model="large-v3",
        language="en",
        response_format="json",
        diarize=True,
    )
    with patch(
        "whispermlx_server.inference.shutil.which", return_value="/usr/bin/whispermlx"
    ):
        with pytest.raises(InferenceError, match="HF_TOKEN"):
            build_argv(request, output_dir=tmp_path / "out", hf_token=None)


def test_build_argv_diarize_with_token(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"fake")
    request = InferenceRequest(
        file_path=str(audio),
        model="large-v3",
        language="en",
        response_format="json",
        diarize=True,
    )
    with patch(
        "whispermlx_server.inference.shutil.which", return_value="/usr/bin/whispermlx"
    ):
        argv = build_argv(request, output_dir=tmp_path / "out", hf_token="hf_test")
    assert "--diarize" in argv
    assert "--hf_token" in argv and "hf_test" in argv


def test_run_inference_sync(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.mp4"
    audio.write_bytes(b"fake")
    request = InferenceRequest(
        file_path=str(audio),
        model="large-v3",
        language="en",
        response_format="json",
    )
    settings = Settings()

    def fake_run(argv: list[str], **kwargs: object) -> object:
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert "PATH" in env
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        result_path = output_dir / "meeting.json"
        result_path.write_text(json.dumps(SAMPLE_RESULT), encoding="utf-8")
        from subprocess import CompletedProcess

        return CompletedProcess(argv, 0, stdout="", stderr="")

    with patch(
        "whispermlx_server.inference.shutil.which", return_value="/usr/bin/whispermlx"
    ):
        with patch("whispermlx_server.inference.subprocess.run", side_effect=fake_run):
            response = run_inference_sync(request, settings)
    assert response.text == "SPEAKER_00: Hello.\nSPEAKER_01: Hi there."


def test_run_inference_sync_missing_file() -> None:
    request = InferenceRequest(
        file_path="/nonexistent/file.mp4",
        model="large-v3",
        language="en",
        response_format="json",
    )
    with pytest.raises(InferenceError, match="file not found"):
        run_inference_sync(request, Settings())
