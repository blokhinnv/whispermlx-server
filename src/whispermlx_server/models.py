from typing import Literal

from pydantic import BaseModel, field_validator


class InferenceRequest(BaseModel):
    """Minuteman POST /inference request body."""

    file_path: str
    model: str
    language: str
    response_format: Literal["json"]
    hotwords: str | None = None
    initial_prompt: str | None = None
    diarize: bool | None = None

    @field_validator("hotwords", "initial_prompt", mode="before")
    @classmethod
    def empty_str_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    def effective_initial_prompt(self) -> str | None:
        """initial_prompt is only sent when language is ru."""
        if self.language != "ru":
            return None
        return self.initial_prompt

    def diarize_enabled(self) -> bool:
        return self.diarize is True


class InferenceResponse(BaseModel):
    """Minuteman POST /inference success response."""

    text: str


class HealthResponse(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    detail: str
