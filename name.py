import os

root = os.getenv('MODEL_PATH', "")
network_name = "red-mediana-mejorada"
model_path = os.path.join(root, "models", network_name)
