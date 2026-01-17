
# Suppose we already have an AIADataset for inputs
aia_dataset = AIADataset(data_path, ...)

# Set prediction folder
pred_path = r"D:\CHASM_TABLES\TESTING_NEW_PIPELINE\predictions"
pred_dataset = PredictionDataset(pred_path)

# If no predictions exist yet, build them
if len(pred_dataset) == 0:
    pred_dataset.build(model, aia_dataset, device="cuda", batch_size=16)

print("Number of predictions:", len(pred_dataset))
sample = pred_dataset[0]
print("Prediction shape:", sample.shape)