# react agent with reasoning-act-observation flow
from typing import Optional, Any

from core.config import Config
from core.llm import LLMClient
from core.agent import Agent
from core.message import Message
from tools.registry import ToolRegistry

DEFAULT_REACT_AGENT_PROMPT = """
You are a helpful assistant who are capable of reason and act. You are good at breaking down problem by reasoning through it, then perform action to 
obtain oberservation that helps you to reason further, until you reachs a conclusion.


## You have following available tools to perform 'action':

{tool_description}


## Problem

{problem}


## Procedure
Strictly follows the rules to reply, one step at a time:

[REASON]
Your thought process here, which were used to analyze the problem, break down the tasks and plan for next action.

[ACT]

The action you decide to perform, MUST be one of the following:

1. Tool calling, which you should use following format:
```
[TOOL](ToolName:Tool parameters)
```

2. You are confident that you reached the solution, which you then should output strictly using the following format:
```
[FINAL_ANSWER] final answer here...
```


## Important reminder
1. You MUST have both [REASON] and [ACT] everytime you reply. Do not output closing tag such as [/ACT].
2. Tool parameters are key-value pair(s), separated by comma, for example:
```
[TOOL](test:foo='bar',x=1,y=False)
```
3. You should output [FINAL_ANSWER] only if you have sufficient information for answering problem.
4. If the tool result in observation lucks information you need, keep trying different tool or same tool with different parameters.


## Execution history

{history}

Now start reasoning and act:
"""


class ReActAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tools: Optional[ToolRegistry] = None,
        max_retries: int = 3,
        max_iterations: int = 5,
    ):
        super().__init__(name, llm, system_prompt, config)

        self.system_prompt = system_prompt or DEFAULT_REACT_AGENT_PROMPT
        self.config = config or Config.from_env()
        self.max_retries = max_retries

        # Maximum iterations for react flow (each reason - act - observation count as 1)
        self.max_iterations = max_iterations

        self.tools = tools

        if self.tools:
            print(
                f"🛠️ Tool enabled for {self.name}. Number of tools mounted: {len(tools)}"
            )
        else:
            raise Exception(f"❌ Cannot mount tool to {self.name}. Program exits.")

        self._history: list[dict[str, Any]] = []

    def run(self, user_input: str, **kwargs) -> str:
        self._history.clear()

        prompt = self.system_prompt.format(
            problem=user_input,
            history=self._history,
            tool_description=self.tools.get_tools_description()
            if self.tools
            else "No available tool",
        )

        messages = [{"role": "user", "content": prompt}]
        self._history.extend(messages)

        print(f"💡 === Agent {self.name} start answering problem ===")

        i = 0
        response = ""
        while i < self.max_iterations:
            print(f"🤔 === Agent {self.name} is thinking... ===")

            response = self.llm.invoke(messages, **kwargs)

            reasoning, action = self._parse_response(response)
            if not reasoning or not action:
                raise Exception(f"❌ Unknown error occurs, agent return None. Complete response: {response}")

            self._history.extend(
                [
                    {"role": "assistant", "react_type": "reasoning", "content": reasoning},
                    {"role": "assistant", "react_type": "action", "content": reasoning}
                ]
            )

            print(f"❗️ === Agent {self.name} produces an action... ===")

            if "[FINAL_ANSWER]" in action:
                break

            if "[TOOL]" in action:
                tool_calling = response.split("[TOOL](")[1].split(")")[0].split(":")
                name, parameters_str = tool_calling[0], tool_calling[1]

                result = ""

                if not self.tools:
                    print(
                        f"❌ Error when calling tool {name}: there is no tool registry binded to agent."
                    )
                elif not self.tools.contains(name):
                    print(
                        f"❌ Error when calling tool {name}: tool does not exist in registry."
                    )
                else:
                    try:
                        result = self.tools.execute(name, parameters_str)
                        # print("tool call result", result)
                    except Exception as e:
                        print(f"Error when calling tool {name}: {e}")

                tool_call_result = [{"role": "tool", "react_type": "observation", "content": result}]
                self._history.extend(tool_call_result)

            prompt = self.system_prompt.format(
                problem=user_input,
                history=self._dump_history(),
                tool_description=self.tools.get_tools_description()
                if self.tools
                else "No available tool",
            )

            messages = [{"role": "user", "content": prompt}]

            # print(prompt)
            self._history.extend(messages)
            print(f"🔍 === Agent {self.name} gets an observation... ===")

            # print(f"step{i} response", response, "\n")

            i += 1

        if "[FINAL_ANSWER]" in response:
            print(f"👌 {self.name} have completed execution.")
            final_answer = response.split("[FINAL_ANSWER]")[1]

            self._history.extend([{"role": "assistant", "react_type": "reasoning", "content": final_answer}])
        else:
            print(
                f"⚠️ {self.name} have reached maximum number of retries allowed. Response might be incomplete."
            )
            final_answer = response

        return final_answer

    def _parse_response(self, response: str) -> tuple[str, str]:
        reasoning, action = "", ""

        if "[REASON]" in response and "[ACT]" in response:
            chunks = response.split("[REASON]")[1].split("[ACT]")
            reasoning, action = chunks[0], chunks[1]
        
        return reasoning, action
            

    def _dump_history(self) -> str:
        entries = []

        for i, entry in enumerate(self._history):
            if entry["role"] != "user":  # discard all system prompts
                entries.append(f"{i}. {entry['react_type']}: {entry['content']}")

        return "\n".join(entries)
