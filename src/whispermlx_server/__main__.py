import argparse

import uvicorn
from loguru import logger

from whispermlx_server.app import create_app
from whispermlx_server.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="whispermlx HTTP server for Minuteman")
    parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default: 8081)"
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        default=None,
        help="Hugging Face token for diarization (default: HF_TOKEN env)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env().with_overrides(
        host=args.host,
        port=args.port,
        hf_token=args.hf_token,
    )
    logger.info("Starting whispermlx-server on {}:{}", settings.host, settings.port)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
