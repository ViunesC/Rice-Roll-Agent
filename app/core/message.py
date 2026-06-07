# message handling
import json

from typing import Literal, Any
from pydantic import BaseModel
from datetime import datetime

MessageRole = Literal["platform", "developer", "user", "assistant", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = {}

    def __init__(self, role: MessageRole, content: str | dict, **kwargs):
        super().__init__(
            role=role,
            content=content if isinstance(content, str) else json.dumps(content),
            timestamp=kwargs.get("timestamp") or datetime.now(),
            metadata=kwargs.get("metadata") or {},
        )

    def to_dict(self):
        """Convert to dictionary (OpenAI format)"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Convert from dictionary"""
        timestamp = data["timestamp"]

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )

    def to_text(self) -> str:
        """Formatting message for agent to read"""
        return f"[{self.role}] {self.content}"

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
