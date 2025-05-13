import os
import pandas as pd

def load_dataset(data_dir="data/raw"):
    data = []
    for model in ["gpt4", "claude", "llama"]:
        folder_path = os.path.join(data_dir, model)
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    data.append({"text": text, "label": model})
    return pd.DataFrame(data)
