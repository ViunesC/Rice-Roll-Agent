# trival agent with sequential workflow
from typing import Optional, Any

from core.config import Config
from core.llm import LLMClient
from core.agent import Agent
from core.message import Message
from tools.registry import ToolRegistry

DEFAULT_TRIVAL_AGENT_PROMPT = """
You are a helpful assistant. Your task is to help user to solve their problem. You have tool(s) to help you solve problem.

This is the problem:

{problem}

This is the history of converstaion:

{history}


This is the list of tools you have, alongwith their description:

{tool_description}


Now work on the problem until you reached a solution, which you then should output strictly using the following format:
```
[FINAL_ANSWER] final answer here...
```

Or, if you need to call tool to answer question, use following format:
```
[TOOL](ToolName:Tool parameters)
```
and Tool parameters are key-value pair(s), separated by comma, for example:
```
...(test:foo='bar',x=1,y=False)
```


Full example:

1. User asked 'What is the largest city in the world?' and suppose you have a search web tool:
```
- `search_google`: ability to search google and return top answers with given query. Tool parameters: `query` - string; `top-k` - integer, return top k answer
```
Your response should contain following tool call:
```
[TOOL](search_google:query='Largest city in the world',top_k=5)
```

2. You've reached a conclusion, and you want to output final answer, your response should contain at the end:
```
[FINAL_ANSWER] The result of 1+2 is 3.
```

"""


class TrivalAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        enable_tool: bool = True,
        tools: Optional[ToolRegistry] = None,
        max_retries: int = 3,
    ):
        super().__init__(name, llm, system_prompt, config)

        self.system_prompt = system_prompt or DEFAULT_TRIVAL_AGENT_PROMPT
        self.config = config or Config.from_env()
        self.enable_tool = enable_tool
        self.max_retries = max_retries

        if self.enable_tool:
            self.tools = tools

            if self.tools:
                print(
                    f"🛠️ Tool enabled for {self.name}. Number of tools mounted: {len(tools)}"
                )
            else:
                print(
                    f"Tool calling enabled for {self.name} but there are no available tool"
                )
        else:
            self.tools = None

        self._history: list[dict[str, Any]] = []

    def run(self, user_input: str, **kwargs) -> str:
        self._history.clear()

        if not self.enable_tool:
            prompt = self.system_prompt.format(
                problem=user_input,
                history=self._history,
                tool_description="No available tool",
            )

            messages = [{"role": "user", "content": prompt}]

            response = self.llm.invoke(messages, **kwargs)

            return response

        return self._run_with_tools(user_input, **kwargs)

    def _run_with_tools(self, user_input: str, **kwargs) -> str:
        prompt = self.system_prompt.format(
            problem=user_input,
            history=self._history,
            tool_description=self.tools.get_tools_description() if self.tools else "No available tool",
        )

        messages = [{"role": "user", "content": prompt}]
        self._history.extend(messages)

        print(f"💡 === Agent {self.name} start answering problem ===")

        response = self.llm.invoke(messages, **kwargs)
        self._history.extend([{"role": "assistant", "content": response}])

        i = 0
        while i < self.max_retries:
            if "[FINAL_ANSWER]" in response:
                break
            
            if "[TOOL]" in response:
                tool_calling = response.split("[TOOL](")[1].split(")")[0].split(":")
                name, parameters_str = tool_calling[0], tool_calling[1]

                result = ""

                if not self.tools:
                    print(f"❌ Error when calling tool {name}: there is no tool registry binded to agent.")
                elif not self.tools.contains(name):
                    print(f"❌ Error when calling tool {name}: tool does not exist in registry.")
                else:
                    try:
                        result = self.tools.execute(name, parameters_str)
                        # print("tool call result", result)
                    except Exception as e:
                        print(f"Error when calling tool {name}: {e}")
                
                tool_call_result = [{"role": "tool", "content": result}]
                self._history.extend(tool_call_result)
            
            prompt = self.system_prompt.format(
                problem=user_input,
                history=self._dump_history(),
                tool_description=self.tools.get_tools_description() if self.tools else "No available tool",
            )

            messages = [{"role": "user", "content": prompt}]

            # print(prompt)

            self._history.extend(messages)
            response = self.llm.invoke(messages, **kwargs)
            self._history.extend([{"role": "assistant", "content": response}])

            # print(f"step{i} response", response, "\n")

            i += 1
        
        if "[FINAL_ANSWER]" in response:
            print(f"👌 {self.name} have completed execution.")
            final_answer = response.split("[FINAL_ANSWER]")[1]
        else:
            print(f"⚠️ {self.name} have reached maximum number of retries allowed. Response might be incomplete.")
            final_answer = response

        return final_answer
    
    def _dump_history(self) -> str:
        entries = []

        for i, entry in enumerate(self._history): 
            if entry["role"] != "user": # discard all system prompts
                entries.append(f"{i}. {entry["role"]}: {entry["content"]}")
        
        return "\n".join(entries)