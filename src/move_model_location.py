import sys, types, torch
import MultiChannelCHDetection as newfile  # your real module



# Create a fake "src" package
src = types.ModuleType("src")
sys.modules["src"] = src

# Register the module under src
sys.modules["src.MultiChannelCHDetection"] = newfile
setattr(src, "MultiChannelCHDetection", newfile)  # <--- important!

# Now load
model = torch.load(
    r"D:\CHASM_TABLES\final_model_CHASM-967.pt",
    map_location="cpu", weights_only=False
)

# Resave it with new path references
torch.save(
    model,
    r"D:\CHASM_TABLES\final_model_CHASM-967_moved.pt"
)
