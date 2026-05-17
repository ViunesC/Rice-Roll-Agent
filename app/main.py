from core.llm import LLMClient
from dotenv import load_dotenv
from agents.trival import TrivalAgent
from tools.concrete.tarot import get_registry

load_dotenv()

if __name__ == "__main__":
    llm_client = LLMClient()

    alex = TrivalAgent("Alex", llm_client, tools=get_registry())
    question = "Could do a little 3-card Tarot prediction on my fortune?"
    
    print("\n", alex.run(question))

    # print(alex._dump_history())
