# base tool

import asyncio

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        pass

    async def arun(self, parameters: dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(parameters))
