old_path = b"src.MultiChannelCHDetection"
new_path = b"MultiChannelCHDetection"

with open("data/final_model_CHASM-1407.pt", "rb") as f:
    data = f.read()

data = data.replace(old_path, new_path)

with open("data/final_model_CHASM-1407_clean.pt", "wb") as f:
    f.write(data)
