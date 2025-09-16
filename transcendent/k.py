from dotenv import load_dotenv
import os

from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool


def load_agent():
    load_dotenv()

    token = os.environ.get("SMOLAGENTS_API_KEY")
    instructions = os.environ.get("AGENT_INTSTRUCTIONS")

    model = InferenceClientModel(token=token)
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        max_steps=2,
        instructions=instructions,
    )

    print(dir(agent))
    print(agent.system_prompt)

    return agent


def ask_agent(agent, prompt):
    result = agent.run(prompt)
    print(result)


if __name__ == "__main__":
    k = load_agent()

    try:
        while True:
            prompt = input("\nHello there! (Ctrl+C to exit): ")
            ask_agent(k, prompt)
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
