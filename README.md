# whispermlx-server

Local HTTP wrapper around the `whispermlx` CLI for [Minuteman](https://github.com/blokhinnv).

## Requirements

- macOS with Apple Silicon
- [uv](https://docs.astral.sh/uv/)
- Python 3.10
- FFmpeg (`brew install ffmpeg`)

`whispermlx` is installed as a project dependency (`uv sync`).

## Run

```bash
uv sync --group dev
HF_TOKEN=hf_... uv run whispermlx-server --port 8081
```

## API

- `GET /` — health check
- `POST /inference` — JSON body with `file_path`, `model`, `language`, `response_format: "json"`, optional `hotwords`, `initial_prompt`, `diarize`

See Minuteman integration spec for the full contract.
