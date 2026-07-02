from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from whispermlx_server.config import Settings
from whispermlx_server.inference import InferenceError, run_inference
from whispermlx_server.models import (
    ErrorResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
)


def create_app(settings: Settings) -> FastAPI:
    """Build FastAPI application with routes and error handlers."""
    app = FastAPI(title="whispermlx-server", version="0.1.0")
    app.state.settings = settings

    @app.exception_handler(InferenceError)
    async def inference_error_handler(
        request: Request,
        exc: InferenceError,
    ) -> JSONResponse:
        client = request.client.host if request.client else "unknown"
        logger.exception(
            "Inference failed: client={} {} {} — {}",
            client,
            request.method,
            request.url.path,
            exc,
        )
        body = ErrorResponse(detail=str(exc))
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        client = request.client.host if request.client else "unknown"
        logger.exception(
            "Unhandled server error: client={} {} {} — {}",
            client,
            request.method,
            request.url.path,
            exc,
        )
        body = ErrorResponse(detail="Internal server error")
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.get("/")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/inference")
    async def inference(body: InferenceRequest) -> InferenceResponse:
        return await run_inference(body, app.state.settings)

    return app
