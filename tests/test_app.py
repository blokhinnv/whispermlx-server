from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from whispermlx_server.app import create_app
from whispermlx_server.config import Settings
from whispermlx_server.models import InferenceResponse


@pytest.fixture
def app():
    return create_app(Settings())


@pytest.mark.asyncio
async def test_health(app) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_inference_success(app) -> None:
    mock_response = InferenceResponse(text="hello")
    with patch(
        "whispermlx_server.app.run_inference",
        new=AsyncMock(return_value=mock_response),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/inference",
                json={
                    "file_path": "/tmp/a.mp4",
                    "model": "large-v3",
                    "language": "en",
                    "response_format": "json",
                },
            )
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


@pytest.mark.asyncio
async def test_inference_server_error(app) -> None:
    from whispermlx_server.inference import InferenceError

    with patch(
        "whispermlx_server.app.run_inference",
        new=AsyncMock(side_effect=InferenceError("boom")),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/inference",
                json={
                    "file_path": "/tmp/a.mp4",
                    "model": "large-v3",
                    "language": "en",
                    "response_format": "json",
                },
            )
    assert response.status_code == 500
    assert response.json() == {"detail": "boom"}
