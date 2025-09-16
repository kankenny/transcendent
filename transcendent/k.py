from dotenv import load_dotenv
import os

from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool


def load_agent_k():
    load_dotenv()
    token = os.environ.get("SMOLAGENTS_API_KEY")

    model = InferenceClientModel(token=token)
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, max_steps=2)

    return agent


def ask_agent_k(agent, prompt):
    result = agent.run(prompt)
    print(result)


if __name__ == "__main__":
    agent = load_agent_k()

    try:
        while True:
            prompt = input("\nHello there! (Ctrl+C to exit): ")
            ask_agent_k(agent, prompt)
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
