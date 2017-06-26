import os

root = os.getenv('MODEL_PATH', "")
network_name = "batch-norm-squeeze-net"
model_path = os.path.join(root, "models", network_name)
