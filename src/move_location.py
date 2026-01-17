import sys
import torch

# --- Step 1: Load the old checkpoint using alias ---
# The checkpoint expects src.MultiChannelCHDetection.chronnos.train.model.CHRONNOS
import src.MultiChannelCHDetection.chronnos.train.model as old_module
sys.modules["src.MultiChannelCHDetection.chronnos.train.model"] = old_module

old_checkpoint_path = r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\model\final_model_CHASM-1407.pt"
orig_model = torch.load(old_checkpoint_path, map_location="cpu", weights_only=False)

print("Original model type:", type(orig_model))
print("Original model class path:", type(orig_model).__module__, type(orig_model).__name__)

# --- Step 2: Rebind the class to the new module path ---
# Import the new module where you want the class to live
from MultiChannelCHDetection.chronnos.train.model import CHRONNOS as NewCHRONNOS

# Assign the new class object to the model
orig_model.__class__ = NewCHRONNOS

# --- Step 3: Save the checkpoint under the new module path ---
new_checkpoint_path = r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\model\final_model_CHASM-1407_moved.pt"
torch.save(orig_model, new_checkpoint_path)

# --- Step 4: Test loading without hacks ---
new_model = torch.load(new_checkpoint_path, map_location="cpu", weights_only=False)
print("New model type:", type(new_model))
print("New model class path:", type(new_model).__module__, type(new_model).__name__)
