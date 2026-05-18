import serpapi
import os

from typing import Any

from tools.tool import Tool, ToolParameter
from tools.registry import ToolRegistry

serpapi_engines = ["google", "baidu", "bing"]


class InternetSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="search_internet",
            description="search internet for given query",
        )

        self._client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))

    def run(self, parameters: dict[str, Any]) -> str:
        query = parameters.get("query", None)
        engine = parameters.get("engine", "google")

        if not query:
            print("❌ Search query cannot be empty")
            return ""
        if engine not in serpapi_engines:
            print(f"⚠️ Unknown engine: {engine}, will fallback to google search.")
            engine = "google"

        results = self._client.search({"engine": engine, "q": query})

        organic_results = results["organic_results"] or ""
        results_str = self._process_search_results(organic_results)

        return results_str

    def _process_search_results(self, results) -> str:
        """Sanitize search result"""
        content = []

        for entry in results:
            res_str = f"""
            position: {entry.get("position", "")},
            link: {entry.get("link", "")},
            title: {entry.get("title", "")},
            snippet: {entry.get("snippet", "")}
            """

            content.append(res_str)

        return "\n".join(content)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="the search query",
                required=True,
            ),
            ToolParameter(
                name="engine",
                type="string",
                description=f"Search engine to be used to perform search. Available engines: {serpapi_engines}",
                required=False,
                default="google",
            ),
        ]


def get_registry() -> ToolRegistry:
    """Get a tool registry containing the tool."""
    registry = ToolRegistry()
    registry.register_tool(InternetSearchTool())

    return registry
