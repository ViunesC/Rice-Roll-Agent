# Plan and solve agent
import ast
from typing import Optional, Any

from core.agent import Agent
from core.config import Config
from core.llm import LLMClient
from tools.registry import ToolRegistry


DEFAULT_PLANNING_AGENT_PROMPT = """
You are the planning agent in a plan-and-solve workflow. Your job is to turn the user's task into a short, executable plan that a separate solving agent can follow.

User task:
{task}

Conversation and tool history:
{history}

Available tools:
{tool_description}

Rules:
- If the task is already clear, produce a plan immediately.
- If important information is missing and an available tool can get it, call exactly one tool.
- Do not solve the task yourself unless the plan would be a single direct answer step.
- Keep the plan compact, ordered, and specific enough for another agent to execute.
- Include tool use in the plan only when it is actually useful.
- Your response must contain either one tool call or one plan, never both.
- The plan must be a plain Python list of strings inside an unlabelled code block.

Tool call format:
```
[TOOL](ToolName:param1='value',param2=1,param3=False)
```

Plan format:
[PLAN]
```
[
    "First concrete step",
    "Second concrete step",
    ...more steps here...
    "Final step that produces the answer for the user"
]
```
"""

DEFAULT_SOLVING_AGENT_PROMPT = """
You are the solving agent in a plan-and-solve workflow. Execute only the current step while using prior step results as context.

User task:
{task}

Current step:
{step}

Previous step results and tool outputs:
{history}

Available tools:
{tool_description}

Rules:
- Complete the current step as far as possible.
- If you need external information or computation and an available tool can provide it, call exactly one tool.
- If a tool result is present in the history, use it to continue the same step instead of repeating the same call.
- Do not invent tool results, facts, files, or observations.
- For non-final steps, return the useful intermediate result plainly and concisely.
- For the final step, return the complete final answer to the user.
- Your response must contain either one tool call or step output, never both.

Tool call format:
```
[TOOL](ToolName:param1='value',param2=1,param3=False)
```
"""

class PlanAndSolveAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[dict[str, str]] = None,
        config: Optional[Config] = None,
        enable_tool: bool = True,
        tools: Optional[ToolRegistry] = None,
        max_retries: int = 3,
    ):
        self.name = name
        self.llm = llm

        self.planner_promopt = (
            system_prompt["planner"] if system_prompt else None
        ) or DEFAULT_PLANNING_AGENT_PROMPT
        self.solver_promopt = (
            system_prompt["solver"] if system_prompt else None
        ) or DEFAULT_SOLVING_AGENT_PROMPT

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

        plan = self._plan(user_input, **kwargs)

        if not plan:
            print("❌ Planner failed to provide a plan. Agent terminates.")
            return ""

        solution = self._solve(user_input, plan, **kwargs)

        if not solution:
            print(
                "❌ Planner provided a plan but solver cannot execute the plan. Agent terminates."
            )
            return ""
        else:
            return solution

    def _plan(self, task: str, max_iterations: int = 5, **kwargs) -> list[str]:
        prompt = self.planner_promopt.format(
            task=task,
            tool_description=self.tools.get_tools_description()
            if self.tools
            else "No available tool",
            history=self._dump_history(),
        )

        messages = [{"role": "user", "content": prompt}]
        self._history.extend(messages)

        print(f"🧠 ====Planner {self.name} start to draft a plan.====")
        response = self.llm.invoke(messages, **kwargs)

        i = 0
        no_inst_given = False
        while i < max_iterations:
            if "[PLAN]" in response:
                break
            elif "[TOOL]" in response:
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

                tool_call_result = [{"role": "tool", "content": result}]
                self._history.extend(tool_call_result)
            else:
                no_inst_given = True
            
            prompt = self.planner_promopt.format(
                task=task,
                tool_description=self.tools.get_tools_description()
                if self.tools
                else "No available tool",
                history=self._dump_history()
            )

            if no_inst_given:
                prompt += "NOTICE: You NEED to either call a tool to get more information or provide a plan in correct format. Please try again."

            messages = [{"role": "user", "content": prompt}]
            self._history.extend(messages)

            # print(prompt)
            response = self.llm.invoke(messages, **kwargs)
            self._history.extend([{"role": "assistant", "content": response}])

            i += 1

        self._history.extend([{"role": "assistant", "subagent_type": "planner", "content": response}])
        
        # extract plan
        raw_plan = response.split("[PLAN]")[1].split("```")[1].split("```")[0].strip()
        plan = ast.literal_eval(raw_plan)

        if plan and isinstance(plan, list):
            print(f"✅ Planner {self.name} have drafted a plan.")
            return plan
        else:
            return []

    def _solve(self, task: str, plan: list[str], **kwargs) -> str:
        print(f"⚙️ ====Solver {self.name} start executing the plan.====")

        response = ""
        
        for i, step in enumerate(plan):
            prompt = self.solver_promopt.format(
                task=task,
                step=f"Step {i+1}: {step}",
                tool_description=self.tools.get_tools_description()
                if self.tools
                else "No available tool",
                history=self._dump_prev_steps()
            )
            messages = [{"role": "user", "content": prompt}]
            self._history.extend(messages)

            response = self.llm.invoke(messages, **kwargs)
            self._history.extend([{"role": "assistant", "subagent_type": "solver", "content": response}])

            j = 0
            while "[TOOL]" in response and j < self.max_retries:
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

                tool_call_result = [{"role": "tool", "content": result}]
                self._history.extend(tool_call_result)

                prompt = self.solver_promopt.format(
                    task=task,
                    step=f"Step {i+1}: {step}",
                    tool_description=self.tools.get_tools_description()
                    if self.tools
                    else "No available tool",
                    history=self._dump_prev_steps()
                )

                messages = [{"role": "user", "content": prompt}]

                # print(prompt)

                self._history.extend(messages)
                response = self.llm.invoke(messages, **kwargs)
                self._history.extend([{"role": "assistant", "subagent_type": "solver", "content": response}])

                j += 1
            
            if j == self.max_retries:
                print("⚠️ Maximum tool call retry amount reached. The final tool call might be incomplete.")
                self._history.extend([{"role": "tool", "content": "Maximum tool call retry reached. The attempt on previous step might be incomplete/failed."}])

        final_response = response
        
        if not final_response:
            print("❌ Something went wrong when executing plan. No final response was given.")
        elif "[TOOL]" in final_response:
            print("😵 Leftover tool call detected. Final response might be incomplete.")
            return final_response.split("[TOOL]")[0]

        return final_response

    
    def _dump_history(self) -> str:
        entries = []

        for i, entry in enumerate(self._history): 
            if entry["role"] != "user": # discard all system prompts
                entries.append(f"{i}. {entry['role']}: {entry['content']}")
        
        return "\n".join(entries) if entries else ""
    
    def _dump_prev_steps(self) -> str:
        entries = []

        for i, entry in enumerate(self._history): 
            if entry["role"] == "assistant" and entry.get("subagent_type") == "solver": # discard all system prompts
                entries.append(f"{i}. {entry['role']}: {entry['content']}")
            elif entry["role"] == "tool":
                entries.append(f"{i}. Toolcalling result: {entry['content']}")
        
        return "\n".join(entries) if entries else ""
