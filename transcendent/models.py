import psutil
import shutil
import os

from smolagents import (
    InferenceClientModel,
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

    if free_ram_gb < 8 or free_disk_gb < 30:
        print("Low resources detected. Using TinyLlama (1.1B).")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    return TransformersModel(
        model_id=model_id,
        device_map="auto",
    )
