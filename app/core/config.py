# config class

import os

from pydantic import BaseModel, ConfigDict
from typing import Any, Optional


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # LLM Client
    llm_api_key: Optional[str]
    base_provider_url: str

    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"

    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # system config
    debug: bool = False
    log_level: str = "INFO"

    # context engineering

    # smart summary

    # tool output

    # observerability

    # skills

    # circuit-breaker

    # chat persistence

    # subagent

    # todowrite

    # devlog

    # async lifecycle

    # streaming output

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            base_provider_url=os.getenv("BASE_PROVIDER_URL"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS"))
            if os.getenv("MAX_TOKENS")
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
