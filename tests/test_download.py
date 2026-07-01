from pathlib import Path
from unittest.mock import patch

from whispermlx_server.download import download_model, list_models


def test_list_models() -> None:
    models = list_models()
    repos = {repo for _, repo in models}
    assert "mlx-community/whisper-large-v3-mlx" in repos
    assert "mlx-community/whisper-large-v3-turbo" in repos
    assert len(models) == 6


def test_download_model() -> None:
    with patch(
        "whispermlx_server.download.snapshot_download",
        return_value="/cache/whisper-large-v3-mlx",
    ) as mock_download:
        path = download_model("large-v3", hf_token="hf_test")

    mock_download.assert_called_once_with(
        repo_id="mlx-community/whisper-large-v3-mlx",
        token="hf_test",
    )
    assert path == Path("/cache/whisper-large-v3-mlx")
