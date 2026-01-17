import torch

new_checkpoint_path = r"model\final_model_CHASM-1110_moved.pt"

new_model = torch.load(new_checkpoint_path, map_location="cpu", weights_only=False)
print("New model type:", type(new_model))
print("New model class path:", type(new_model).__module__, type(new_model).__name__)
