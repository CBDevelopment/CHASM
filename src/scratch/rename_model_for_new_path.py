import sys
import types
import importlib
import torch
import MultiChannelCHDetection as newfile  # your refactored module

# -----------------------------
# Step 0: Create shim for old path
# -----------------------------
src = types.ModuleType("src")
sys.modules["src"] = src
sys.modules["src.MultiChannelCHDetection"] = newfile
setattr(src, "MultiChannelCHDetection", newfile)

# -----------------------------
# Step 1: Load the old checkpoint
# -----------------------------
old_ckpt_path = r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\data\final_model_CHASM-1110.pt"
model = torch.load(old_ckpt_path, map_location="cpu", weights_only=False)

# -----------------------------
# Step 2: Recursive class rebinder
# -----------------------------
def rebind_classes(obj, old_prefix="src.", new_prefix=""):
    """
    Recursively replace classes from old_prefix with ones from new_prefix.
    """
    if hasattr(obj, "__class__"):
        cls = obj.__class__
        if cls.__module__.startswith(old_prefix):
            # Compute new module name
            new_module = cls.__module__.replace(old_prefix, new_prefix, 1)
            # Import the new module
            mod = importlib.import_module(new_module)
            # Get the new class
            new_cls = getattr(mod, cls.__name__)
            # Rebind
            obj.__class__ = new_cls
            print(f"Rebinding {cls.__module__}.{cls.__name__} → {new_module}.{new_cls.__name__}")

    # Recurse into common containers
    if isinstance(obj, dict):
        for v in obj.values():
            rebind_classes(v, old_prefix, new_prefix)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            rebind_classes(v, old_prefix, new_prefix)
    elif hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            rebind_classes(v, old_prefix, new_prefix)

# Apply the rebinder
rebind_classes(model)

# -----------------------------
# Step 3: Save the cleaned checkpoint
# -----------------------------
new_ckpt_path = r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\data\final_model_CHASM-1110_clean.pt"
torch.save(model, new_ckpt_path)
print(f"Saved cleaned checkpoint to: {new_ckpt_path}")
