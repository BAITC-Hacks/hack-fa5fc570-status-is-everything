"""Compatibility entry point; legacy model files are intentionally not loaded."""
from src.train import train_all as train_all_models
from src.predict import predict_power

if __name__ == "__main__":
    train_all_models()
