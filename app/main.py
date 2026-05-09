from core.llm import LLMClient
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    llm_client = LLMClient()

    messages = [{"role":"user","content":"What are some of the largest SUVs on the world?"}]
    for chunk in llm_client.think(messages):
        print(chunk, end="", flush=True)
    print()
