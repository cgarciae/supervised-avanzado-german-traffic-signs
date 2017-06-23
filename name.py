import os

root = os.getenv('MODEL_PATH', "")
network_name = "res-squeeze-net"
model_path = os.path.join(root, "models", network_name)
