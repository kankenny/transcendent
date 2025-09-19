from dotenv import load_dotenv
import os

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
)

from models import build_local_model, build_remote_model


def load_agent():
    load_dotenv()

    instructions = os.environ.get("AGENT_INSTRUCTIONS")
    force_local = os.environ.get("FORCE_LOCAL", "").lower() in ("1", "true", "yes")

    if force_local:
        model = build_local_model()
    else:
        try:
            model = build_remote_model()
            print("Using remote model")
        except Exception as e:
            print(f"Remote model unavailable: {e}")
            model = build_local_model()

    return CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        max_steps=2,
        instructions=instructions,
    )


def ask_agent(agent, prompt):
    result = agent.run(prompt)
    print(result)


if __name__ == "__main__":
    agent = load_agent()

    try:
        while True:
            prompt = input("\nHello there! (Ctrl+C to exit): ")
            ask_agent(agent, prompt)
    except KeyboardInterrupt:
        ask_agent("\nGoodbye!")
