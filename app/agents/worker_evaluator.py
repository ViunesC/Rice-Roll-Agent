# Worker-evaluator agent flow
import ast
import json
import re
from typing import Optional, Any
from pydantic import BaseModel, Field

from core.agent import Agent
from core.config import Config
from core.llm import LLMClient
from core.message import Message

from tools.registry import ToolRegistry


DEFAULT_WORKER_AGENT_PROMPT = """
You are a worker in a worker-evaluator workflow. Your job is to solve the user's task.

This is the task:

{task}

Your last attempt on the task:

{last_attempt}

{feedback}


Available tools for you to call:
{tool_description}

Now you need to work on the task, then when it's done you should either respond with:

1. Tool calling, which you should use following format:
```
[TOOL](ToolName:Tool parameters)
```

2. You are confident that you reached the solution, which you then should output strictly using the following format:
```
[FINAL_ANSWER] final answer here...
```


## Important reminder
1. Tool parameters are key-value pair(s), separated by comma, for example:
```
[TOOL](test:foo='bar',x=1,y=False)
```
2. If there is feedback from evaluator, you need to adhere to the feedback and rework on the task.
3. You should output [FINAL_ANSWER] only if you have sufficient information for answering problem.
4. If the tool result in latest history lucks information you need, keep trying different tool or same tool with different parameters.
"""

DEFAULT_EVALUATOR_AGENT_PROMPT = """
You are an expert evaluator in a worker-evaluator workflow. Your job is to evaluate the worker's response on a certain task.

This is the task

{task}

This is the response from worker:

{last_attempt}


You should evaluate the quality of the answer based on following rules:
1. Evaluate the logical coherence and the soundness of argument.
2. If the response looks legitimate, challenge it by considering from different perspective, and edge cases that are reasonable but niche.
3. Don't go too far and ask the answer to be perfect. It only need to be generally acceptable and correct.
"""


class EvaluatorResponseSchema(BaseModel):
    need_rework: bool = Field(
        title="need_rework",
        description="Whether or not worker should revise their response",
        default=False,
    )
    feedback: str = Field(title="feedback", description="Your feedback", default="")


class WorkerEvaluatorAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[dict[str, str]] = None,
        config: Optional[Config] = None,
        enable_tool: bool = True,
        tools: Optional[ToolRegistry] = None,
        max_retries: int = 3,
        max_steps: int = 5,
    ):
        self.name = name
        self.llm = llm

        self.worker_promopt = (
            system_prompt["worker"] if system_prompt else None
        ) or DEFAULT_WORKER_AGENT_PROMPT
        self.evaluator_promopt = (
            system_prompt["evaluator"] if system_prompt else None
        ) or DEFAULT_EVALUATOR_AGENT_PROMPT

        self.config = config or Config.from_env()
        self.enable_tool = enable_tool
        self.max_retries = max_retries
        self.max_steps = max_steps

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

        self._history: list[Message] = []

    def run(self, user_input: str, **kwargs) -> str:
        feedback_template = "Your last attempt was evaluated by an expert, and here is the feedback: \n {feedback}"
        self._history.clear()

        print(f"⚙️ ======== Worker {self.name} start working on the task ========")
        response = self._work(user_input, **kwargs)

        i = 0
        while i < self.max_steps:
            print(
                f"🥸 ======== Evalutor {self.name} is evaluating the previous attempt on task ========"
            )
            eval_result = self._evaluate(user_input, response, **kwargs)

            if not eval_result["need_rework"]:
                print(
                    f"🀄️ ======== Evalutor {self.name} accepted the previous attempt========"
                )
                break

            print(
                f"🥺 ======== Worker {self.name} receives the feedback and start reworking========"
            )
            feedback = feedback_template.format(feedback=eval_result["feedback"])
            response = self._work(user_input, feedback, **kwargs)

            i += 1

        if i == self.max_steps:
            print("⚠️ Maximum retries reached. Response might be incomplete.")

        if "[FINAL_ANSWER]" not in response:
            return response

        final_response = response.split("[FINAL_ANSWER]", 1)[1]
        return final_response or response

    def _work(self, task: str, feedback: str = "", **kwargs) -> str:
        prompt = self.worker_promopt.format(
            task=task,
            tool_description=self.tools.get_tools_description()
            if self.tools
            else "No available tool",
            last_attempt=self._dump_last_attempt(),
            feedback=feedback if feedback else "",
        )

        messages = [Message(role="user", content=prompt)]
        self._history.extend(messages)

        i = 0
        response = ""
        while i < self.max_retries:
            response = self.llm.invoke(messages, **kwargs)
            self._history.extend(
                [
                    Message(
                        role="assistant",
                        content=response,
                        metadata={"subagent_type": "worker"},
                    )
                ]
            )

            if "[TOOL]" in response:
                tool_calling = response.split("[TOOL](")[1].split(")")[0].split(":", 1)
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

                tool_call_result = [Message(role="tool", content=result)]
                self._history.extend(tool_call_result)

                if i == self.max_retries - 1:
                    print(
                        "⚠️ Maximum tool call retry amount reached. The final tool call might be incomplete."
                    )
                    self._history.extend(
                        [
                            Message(
                                role="tool",
                                content="Maximum tool call retry reached. The attempt on previous step might be incomplete/failed.",
                            )
                        ]
                    )

                prompt = self.worker_promopt.format(
                    task=task,
                    tool_description=self.tools.get_tools_description()
                    if self.tools
                    else "No available tool",
                    last_attempt=self._dump_last_attempt(),
                    feedback=feedback if feedback else "",
                )

                messages = [Message(role="user", content=prompt)]
                self._history.extend(messages)
            else:
                break

            i += 1

        return response

    def _evaluate(self, task: str, last_attempt: str, **kwargs) -> dict[str, Any]:
        prompt = self.evaluator_promopt.format(task=task, last_attempt=last_attempt)
        messages = [Message(role="user", content=prompt)]
        self._history.extend(messages)

        response = self.llm.invoke(
            messages,
            structured_output=True,
            output_schema=EvaluatorResponseSchema,
            **kwargs,
        )
        self._history.extend(
            [
                Message(
                    role="assistant",
                    content=response,
                    metadata={"subagent_type": "evaluator"},
                )
            ]
        )

        if isinstance(response, dict) and "need_rework" in response:
            if response["need_rework"]:
                print(
                    f"✅ Evaluator {self.name} evaluated the work and provided feedback."
                )
            else:
                print(f"✅ Evaluator {self.name} evaluated the work and accepted it.")

            return response

        raise Exception(
            f"❌ Something went wrong during evaluation: evaluator returned empty/invalid response. The evalutor response is {response}"
        )

    def _dump_history(self) -> str:
        entries = []

        for i, entry in enumerate(self._history):
            if entry["role"] != "user":  # discard all system prompts
                entries.append(f"{i}. {entry['role']}: {entry['content']}")

        return "\n".join(entries) if entries else ""

    def _dump_last_attempt(self) -> str:
        for entry in reversed(self._history):
            if entry["role"] == "assistant" and entry["subagent_type"] == "worker":
                return f"Assistant: {entry['content']}"

        return ""
