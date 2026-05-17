# base agent

from abc import ABC, abstractmethod
from typing import Optional

from core.llm import LLMClient
from core.config import Config

DEFAULT_SYSTEM_PROMPT="""
You are a helpful assistant.
"""

class Agent(ABC):
    """Base class for agents"""

    def __init__(self, name: str, llm: LLMClient, system_prompt: Optional[str], config: Optional[Config]):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.config = config or Config.from_env()

        # TODO: tool
    
    @abstractmethod
    def run(self, user_input: str, **kwargs) -> str:
        pass