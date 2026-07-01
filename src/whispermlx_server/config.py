import os
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Server configuration loaded from environment and CLI overrides."""

    host: str = "127.0.0.1"
    port: int = 8081
    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Load defaults from environment variables."""
        port_raw = os.environ.get("WHISPERMLX_PORT", "8081")
        hf_token = os.environ.get("HF_TOKEN")
        return cls(
            host=os.environ.get("WHISPERMLX_HOST", "127.0.0.1"),
            port=int(port_raw),
            hf_token=hf_token if hf_token else None,
        )

    def with_overrides(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        hf_token: str | None = None,
    ) -> "Settings":
        """Return a copy with CLI overrides applied."""
        return Settings(
            host=host if host is not None else self.host,
            port=port if port is not None else self.port,
            hf_token=hf_token if hf_token is not None else self.hf_token,
        )
