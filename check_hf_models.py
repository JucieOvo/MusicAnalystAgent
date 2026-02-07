
import os
import sys
from huggingface_hub import HfApi, list_models

# 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def check_models():
    print("Searching for CLaMP models...")
    try:
        api = HfApi()
        models = api.list_models(author="sander-wood", search="clamp")
        found = False
        for model in models:
            print(f"- {model.modelId} ({model.downloads} downloads)")
            found = True
        
        if not found:
            print("No models found under 'sander-wood' with 'clamp'.")
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    check_models()
