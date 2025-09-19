from dotenv import load_dotenv
import os

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    DuckDuckGoSearchTool,
    TransformersModel,
)


def build_remote_model():
    token = os.environ.get("SMOLAGENTS_API_KEY")
    if not token:
        raise ValueError("SMOLAGENTS_API_KEY not found")
    return InferenceClientModel(token=token)


def build_local_model():
    print("Using local model (transformers backend).")
    return TransformersModel(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
    )


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
    print(f"\nAgent response:\n{result}\n")


if __name__ == "__main__":
    agent = load_agent()

    try:
        while True:
            prompt = input("\nHello there! (Ctrl+C to exit): ")
            ask_agent(agent, prompt)
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
