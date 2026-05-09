# LLM client
# add support to universal models with OpenAI format
# capable of no-code switch, auto infering provider, fallback
#
# Future improvement: Async invoke

import os

from openai import OpenAI
from typing import Optional, Any, Iterator

MODELSCOPE_BASE_URL="https://api-inference.modelscope.cn/v1/"
OPENAI_BASE_URL="https://api.openai.com/v1"
ANTHROPIC_BASE_URL="https://api.anthropic.com/v1"
AIHUBMIX_BASE_URL="https://aihubmix.com/v1"

class LLMClient:

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens or os.getenv("MAX_TOKENS")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.kwargs = kwargs

        self._provider = self._infer_provider(api_key, base_url)

        if self._provider == "default":
            # use default provider
            pass
        else:
            self.model = model or os.getenv("LLM_MODEL")
            self.api_key, self.base_url = self._resolve_credientials(api_key, base_url)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def invoke(self, messages: list[dict[str, Any]], **kwargs) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            return content
        except Exception as e:
            print(f"Failed to call LLM: {e}") # TODO: replace w/ custom Exception
            return ""

    def think(self, messages: list[dict[str, Any]], **kwargs) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )

            collected_content = []

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content

        except Exception as e:
            print(f"Failed to call LLM with streaming: {e}") # TODO: replace w/ custom Exception
            yield ""

    def invoke_with_tool(self) -> str:
        pass

    async def ainvoke(self):
        pass

    async def athink(self):
        pass

    async def ainvoke_with_tool(self):
        pass

    def _infer_provider(self, input_api_key, input_url) -> str:
        """Automatically infer provider based on given credientials"""

        # 1. platform specfic key
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("AIHUBMIX_API_KEY"):
            return "aihubmix"
        # ...

        # 2. infer from url
        actual_url = input_url or os.getenv("BASE_PROVIDER_URL")

        if actual_url:
            url_lower = actual_url.lower()

            if MODELSCOPE_BASE_URL.split("//")[1].split("/")[0] in url_lower:
                return "modelscope"
            elif OPENAI_BASE_URL.split("//")[1].split("/")[0] in url_lower:
                return "openai"
            elif ANTHROPIC_BASE_URL.split("//")[1].split("/")[0] in url_lower:
                return "anthropic"
            elif AIHUBMIX_BASE_URL.split("//")[1].split("/")[0] in url_lower:
                return "aihubmix" 
            elif "localhost:" in url_lower:
                if ":11434" in url_lower:
                    return "ollama"
                if ":8000" in url_lower:
                    return "vllm"
                return "local"
        
        # 3. infer from apikey (only works in some case)
        actual_api_key = input_api_key or os.getenv("LLM_API_KEY")

        if actual_api_key and actual_api_key.startswith("ms-"):
            return "modelscope"
        # ...
        
        # 4. fallback to default choice
        print("❌ Cannot infer provider based on given info, fallback to default provider...") # TODO: replace w/ logger
        return "default"
    
    def _resolve_credientials(self, input_api_key, input_base_url) -> tuple[Optional[str], str]:
        actual_api_key, actual_base_url = None, None

        if self._provider == "openai":
            actual_api_key = os.getenv("OPENAI_API_KEY")
            actual_base_url = os.getenv("BASE_PROVIDER_URL") or input_base_url or OPENAI_BASE_URL
        elif self._provider == "modelscope":
            actual_api_key = os.getenv("MODELSCOPE_API_KEY")
            actual_base_url = os.getenv("BASE_PROVIDER_URL") or input_base_url or MODELSCOPE_BASE_URL
        elif self._provider == "anthropic":
            actual_api_key = os.getenv("ANTHROPIC_API_KEY")
            actual_base_url = os.getenv("BASE_PROVIDER_URL") or input_base_url or ANTHROPIC_BASE_URL
        elif self._provider == "aihubmix":
            actual_api_key = os.getenv("AIHUBMIX_API_KEY")
            actual_base_url = os.getenv("BASE_PROVIDER_URL") or input_base_url or AIHUBMIX_BASE_URL
        else:
            actual_base_url = os.getenv("BASE_PROVIDER_URL") or input_base_url

        if not actual_api_key:
            actual_api_key = input_api_key or os.getenv("LLM_API_KEY")

        return actual_api_key, actual_base_url