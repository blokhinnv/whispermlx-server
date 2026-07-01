import pytest

from whispermlx_server.model_catalog import resolve_repo


@pytest.mark.parametrize(
    ("model", "repo"),
    [
        ("tiny", "mlx-community/whisper-tiny-mlx"),
        ("base", "mlx-community/whisper-base-mlx"),
        ("small", "mlx-community/whisper-small-mlx"),
        ("medium", "mlx-community/whisper-medium-mlx"),
        ("large-v3", "mlx-community/whisper-large-v3-mlx"),
        ("large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
        ("turbo", "mlx-community/whisper-large-v3-turbo"),
        ("LARGE-V3", "mlx-community/whisper-large-v3-mlx"),
    ],
)
def test_resolve_repo(model: str, repo: str) -> None:
    assert resolve_repo(model) == repo


def test_resolve_repo_unknown() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        resolve_repo("xlarge")
