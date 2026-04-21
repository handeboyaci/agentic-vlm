
import torch
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.skills.predictor import load_model

def check_model():
    model_path = "models/gnn_predictor.pth"
    if not os.path.exists(model_path):
        print(f"{model_path} does not exist.")
        return
    
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        print(f"Use protein encoder: {model.use_protein_encoder}")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
