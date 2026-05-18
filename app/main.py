from core.llm import LLMClient
from dotenv import load_dotenv
from agents.trival import TrivalAgent
from agents.plan_and_solve import PlanAndSolveAgent
from tools.concrete.internet_search import get_registry

load_dotenv()

if __name__ == "__main__":
    llm_client = LLMClient()

    # alex = TrivalAgent("Alex", llm_client, tools=get_registry())
    # question = "Could do a little 3-card Tarot prediction on my fortune?"
    
    # print("\n", alex.run(question))

    # print(alex._dump_history())

    humphrey = PlanAndSolveAgent("Humphrey", llm_client, tools=get_registry())
    logic_task = "There are 4 vacant rooms in Hotel Mossuri on the first night. " \
    "On the second day there 2 guests moved-in and occupied 1 room. Then on the same day there are 3 guests moved out" \
    "and 2 rooms became vacant. On the third day there are a group of tourists arrived and taken 4 rooms, then there is a family" \
    "moved out and a new room became empty. Question: How many vacant room are there on the fourth day?"

    tool_calling_task = "Plan a trip to Toronto, Canada. Give me recommendation in every aspect of a trip."
    print("\n", humphrey.run(tool_calling_task))
    print(humphrey._dump_history())
