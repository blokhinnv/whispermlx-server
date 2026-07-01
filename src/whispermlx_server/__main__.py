import argparse
import sys

import uvicorn
from loguru import logger

from whispermlx_server.app import create_app
from whispermlx_server.config import Settings
from whispermlx_server.download import download_model, list_models


def build_parser() -> argparse.ArgumentParser:
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

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run HTTP server (default)")

    download = subparsers.add_parser(
        "download",
        help="Pre-download a whisper MLX model from Hugging Face Hub",
    )
    download.add_argument(
        "model",
        nargs="?",
        help="Model alias: tiny, base, small, medium, large-v3, turbo",
    )
    download.add_argument(
        "--list",
        action="store_true",
        help="List available model aliases and Hugging Face repos",
    )

    return parser


def run_serve(args: argparse.Namespace) -> None:
    settings = Settings.from_env().with_overrides(
        host=args.host,
        port=args.port,
        hf_token=args.hf_token,
    )
    logger.info("Starting whispermlx-server on {}:{}", settings.host, settings.port)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


def run_download(args: argparse.Namespace) -> None:
    if args.list:
        for alias, repo in list_models():
            print(f"{alias}\t{repo}")
        return

    if not args.model:
        print("error: model alias required (or use --list)", file=sys.stderr)
        sys.exit(2)

    settings = Settings.from_env().with_overrides(hf_token=args.hf_token)
    try:
        path = download_model(args.model, hf_token=settings.hf_token)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        run_download(args)
        return

    run_serve(args)


if __name__ == "__main__":
    main()
