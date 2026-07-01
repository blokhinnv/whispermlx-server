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

`serve` is the default subcommand; flags work the same way:

```bash
uv run whispermlx-server serve --port 8081
```

## Download models

Pre-download MLX weights from Hugging Face Hub (same repos whispermlx uses):

```bash
uv run whispermlx-server download large-v3
uv run whispermlx-server download --list
```

Aliases: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, `turbo`.

## API

- `GET /` — health check
- `POST /inference` — JSON body with `file_path`, `model`, `language`, `response_format: "json"`, optional `hotwords`, `initial_prompt`, `diarize`

See Minuteman integration spec for the full contract.
