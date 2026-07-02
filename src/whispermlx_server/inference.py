import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from whispermlx_server.config import Settings
from whispermlx_server.models import InferenceRequest, InferenceResponse

_inference_lock = asyncio.Lock()


class InferenceError(Exception):
    """Raised when whispermlx subprocess fails or output is unusable."""


def _subprocess_env() -> dict[str, str]:
    """Child env with common macOS bin dirs prepended to PATH."""
    env = os.environ.copy()
    # ponytail: GUI-launched servers often lack Homebrew in PATH
    extra = ":".join(
        d for d in ("/opt/homebrew/bin", "/usr/local/bin") if os.path.isdir(d)
    )
    if extra:
        env["PATH"] = f"{extra}:{env.get('PATH', '')}"
    return env


def _format_argv(argv: list[str]) -> str:
    """Join argv for logs; redact secrets."""
    redacted = list(argv)
    try:
        token_idx = redacted.index("--hf_token")
        if token_idx + 1 < len(redacted):
            redacted[token_idx + 1] = "***"
    except ValueError:
        pass
    return " ".join(redacted)


def _log_inference_request(request: InferenceRequest) -> None:
    logger.info(
        "Inference start: file_path={!r} model={!r} language={!r} "
        "diarize={} hotwords={!r} initial_prompt={!r}",
        request.file_path,
        request.model,
        request.language,
        request.diarize_enabled(),
        request.hotwords,
        request.effective_initial_prompt(),
    )


def extract_text(result: dict[str, object]) -> str:
    """Build plain transcript text from whispermlx JSON segments."""
    segments = result.get("segments")
    if not isinstance(segments, list):
        return ""

    lines: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text_raw = segment.get("text")
        if not isinstance(text_raw, str):
            continue
        line = text_raw.strip()
        if not line:
            continue
        speaker = segment.get("speaker")
        if isinstance(speaker, str) and speaker:
            lines.append(f"{speaker}: {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def build_argv(
    request: InferenceRequest,
    *,
    output_dir: Path,
    hf_token: str | None,
) -> list[str]:
    """Map InferenceRequest fields to whispermlx CLI arguments."""
    whispermlx = shutil.which("whispermlx")
    if whispermlx is None:
        msg = "whispermlx executable not found in PATH"
        raise InferenceError(msg)

    argv = [
        whispermlx,
        request.file_path,
        "--model",
        request.model,
        "--output_format",
        "json",
        "--output_dir",
        str(output_dir),
        "--verbose",
        "False",
    ]

    if request.language != "auto":
        argv.extend(["--language", request.language])

    if request.hotwords:
        argv.extend(["--hotwords", request.hotwords])

    initial_prompt = request.effective_initial_prompt()
    if initial_prompt:
        argv.extend(["--initial_prompt", initial_prompt])

    if request.diarize_enabled():
        if not hf_token:
            msg = "diarize requested but HF_TOKEN is not configured on the server"
            raise InferenceError(msg)
        argv.append("--diarize")
        argv.extend(["--hf_token", hf_token])

    return argv


def _run_subprocess(argv: list[str]) -> None:
    """Run whispermlx CLI synchronously."""
    argv_log = _format_argv(argv)
    logger.info("Running whispermlx: {}", argv_log)
    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            env=_subprocess_env(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        logger.error(
            "whispermlx failed: exit_code={} argv={} stdout={!r} stderr={!r}",
            exc.returncode,
            argv_log,
            stdout or "(empty)",
            stderr or "(empty)",
        )
        detail = stderr or stdout or "(no output)"
        msg = f"whispermlx exited with code {exc.returncode}: {detail}"
        raise InferenceError(msg) from exc

    if completed.stderr.strip():
        logger.debug("whispermlx stderr: {}", completed.stderr.strip())


def _read_result_json(output_dir: Path, audio_path: Path) -> dict[str, object]:
    stem = audio_path.stem
    result_path = output_dir / f"{stem}.json"
    if not result_path.is_file():
        msg = f"whispermlx output not found: {result_path}"
        logger.error("Inference postprocess failed: {}", msg)
        raise InferenceError(msg)

    with result_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        msg = "whispermlx output JSON is not an object"
        logger.error("Inference postprocess failed: loaded={!r}", loaded)
        raise InferenceError(msg)
    return loaded


def run_inference_sync(
    request: InferenceRequest,
    settings: Settings,
) -> InferenceResponse:
    """Run whispermlx subprocess and return transcript text."""
    _log_inference_request(request)
    audio_path = Path(request.file_path)
    if not audio_path.is_file():
        msg = f"file not found: {audio_path}"
        logger.error("Inference precheck failed: {}", msg)
        raise InferenceError(msg)

    with tempfile.TemporaryDirectory(prefix="whispermlx-server-") as tmp:
        output_dir = Path(tmp)
        argv = build_argv(request, output_dir=output_dir, hf_token=settings.hf_token)
        _run_subprocess(argv)
        result = _read_result_json(output_dir, audio_path)
        text = extract_text(result)
        return InferenceResponse(text=text)


async def run_inference(
    request: InferenceRequest,
    settings: Settings,
) -> InferenceResponse:
    """Serialize inference through a single lock and worker thread."""
    async with _inference_lock:
        return await asyncio.to_thread(run_inference_sync, request, settings)
