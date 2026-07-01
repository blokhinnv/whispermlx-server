MODEL_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_repo(model: str) -> str:
    """Map a whisper model alias to a Hugging Face repo id."""
    key = model.strip().lower()
    try:
        return MODEL_REPOS[key]
    except KeyError as exc:
        known = ", ".join(sorted(set(MODEL_REPOS)))
        msg = f"unknown model {model!r}; known: {known}"
        raise ValueError(msg) from exc
