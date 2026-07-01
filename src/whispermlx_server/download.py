from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

from whispermlx_server.model_catalog import MODEL_REPOS, resolve_repo


def download_model(model: str, *, hf_token: str | None = None) -> Path:
    """Download a whisper MLX model from Hugging Face Hub."""
    repo_id = resolve_repo(model)
    logger.info("Downloading {} from Hugging Face Hub", repo_id)
    cache_dir = snapshot_download(repo_id=repo_id, token=hf_token)
    path = Path(cache_dir)
    logger.info("Model cached at {}", path)
    return path


def list_models() -> list[tuple[str, str]]:
    """Return unique model aliases and their Hugging Face repos."""
    seen: set[str] = set()
    models: list[tuple[str, str]] = []
    for alias, repo in MODEL_REPOS.items():
        if repo in seen:
            continue
        seen.add(repo)
        models.append((alias, repo))
    return models
