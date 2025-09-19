from dotenv import load_dotenv
import os
import psutil
import shutil

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
    free_ram_gb = psutil.virtual_memory().available / (1024**3)
    free_disk_gb = shutil.disk_usage("/").free / (1024**3)

    print(f"Detected free RAM: {free_ram_gb:.1f} GB")
    print(f"Detected free disk: {free_disk_gb:.1f} GB")

    if free_ram_gb < 8 or free_disk_gb < 15:
        print("Low resources detected. Using TinyLlama (1.1B).")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    return TransformersModel(
        model_id=model_id,
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
        ask_agent("\nGoodbye!")
