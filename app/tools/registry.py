# tool registry

from tools.tool import Tool
from typing import Any

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, new_tool: Tool):
        if new_tool.name in self._tools:
            print(
                "ℹ️ Tool with same name exists in registry. Will override the old tool"
            )

        self._tools[new_tool.name] = new_tool

    def unregister_tool(self, name: str) -> bool:
        if name not in self._tools:
            print(f"❌ Tool {name} not exist in registry")
            return False

        del self._tools[name]
        print(f"Tool {name} has been deleted from registry")

        return True
    
    def contains(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, parameters_str: str) -> str:
        """
        Execute tool in tool registry.
        Format of parameters:

        parameters_str = "param1=value1,param2=value2,..."

        parameters = {
            "param1":value1,
            "param2":value2,
            ...
        }
        """
        if name not in self._tools:
            raise ValueError("Tool not exist in registry")

        try:
            print(f"👍 Calling tool {name}...")

            tool = self._tools[name]
            parameters = self._parse_parameters(parameters_str)

            result = tool.run(parameters)

            return result
        except Exception as e:
            print(f"❌ Attempt to run tool {name} failed: {e}")
            return ""

    def _parse_parameters(self, parameters_str: str) -> dict[str, Any]:
        pairs = parameters_str.split(",")

        parameters = {}

        for pair in pairs:
            kv = pair.split("=")
            key, value = kv[0], kv[1]

            try:
                value = float(value)
            except ValueError:
                if value == "False" or value == "True":
                    value = bool(value)
                else:
                    value = value.split("'")[1].split("'")[0]
            
            parameters[key] = value

        return parameters
    
    def get_tools_description(self) -> str:
        descriptions = []

        for _, tool in self._tools.items():
            parameters = tool.get_parameters()

            if parameters:
                parameter_descriptions = []
                for parameter in parameters:
                    parameter_description = (
                        f"`{parameter.name}` - {parameter.type}, {parameter.description}"
                    )

                    if not parameter.required:
                        parameter_description += ", optional"

                    if parameter.default is not None:
                        parameter_description += f", default: {parameter.default}"

                    parameter_descriptions.append(parameter_description)

                tool_parameters = "; ".join(parameter_descriptions)
            else:
                tool_parameters = "none"

            descriptions.append(
                f"- `{tool.name}`: {tool.description}. Tool parameters: {tool_parameters}"
            )

        return "\n".join(descriptions)

    def __len__(self) -> int:
        return len(self._tools)


if __name__ == "__main__":
    rg = ToolRegistry()

    print(rg._parse_parameters("param1=1,param2='value2',param3=True"))
