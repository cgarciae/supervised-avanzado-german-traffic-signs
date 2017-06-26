import os

root = os.getenv('MODEL_PATH', "")
network_name = "fire-2"
model_path = os.path.join(root, "models", network_name)
