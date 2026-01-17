from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch
import os

from MultiChannelCHDetection.chronnos.evaluate.detect import CHRONNOSDetector
from datasets.chronnos_dataset import CHRONNOSDataset


class NumpyDataset(Dataset):
    def __init__(self, tensors, names):
        self.tensors = tensors
        self.names = names

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.names[idx]

class PredictionDataset(Dataset):
    def __init__(self, prediction_path):
        """
        Args:
            prediction_path (str or Path): Folder containing saved .npy predictions.
        """
        self.prediction_path = Path(prediction_path)
        self.prediction_path.mkdir(parents=True, exist_ok=True)

        # Load list of saved predictions
        self.files = sorted(self.prediction_path.glob("*.npy"))


    # TODO remove this old build that doesn't work as well as with the ipredict_numpy method
    # def build(self, model, chronnos_dataset, device="cuda", force_recompute=False, batch_size=8):
    #     """
    #     Run predictions on an CHRONNOSDataset and save them into prediction_path as .npy.

    #     Args:
    #         model: PyTorch model (already loaded and ready).
    #         chronnos_dataset: CHRONNOSDataset
    #         device (str): Device to run predictions on.
    #         force_recompute (bool): If True, overwrite existing .npy files.
    #         batch_size (int): Batch size for inference.
    #     """

    #     model.eval()
    #     loader = DataLoader(chronnos_dataset.getPredictionInputsDataset(), batch_size=batch_size, shuffle=False)

    #     for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Building predictions"):
    #         # Create filename
    #         fname = self.prediction_path / f"pred_{idx:05d}.npy"
    #         if fname.exists() and not force_recompute:
    #             continue

    #         with torch.no_grad():
    #             batch, _ = batch # Batch in form tensor, file_path
    #             batch = batch.to(device)
    #             print("BATCH SHAPE ", batch.shape)
    #             preds = model(batch)

    #         # Save each prediction in batch separately
    #         preds = preds.cpu().numpy()
    #         for j, pred in enumerate(preds):
    #             np.save(self.prediction_path / f"pred_{idx*batch_size + j:05d}.npy", pred)

    #     # Refresh file list
    #     self.files = sorted(self.prediction_path.glob("*.npy"))
    
        
    def build(self, model_path, chronnos_dataset: CHRONNOSDataset, device="cuda", num_workers=8, batch_size=16, resolution=512):

        loader = DataLoader(chronnos_dataset.get_map_dataset(resolution=resolution), batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Load model
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()

        for batch, names in tqdm(loader, desc="Predicting and saving"):
            batch = batch.to(device)
            with torch.no_grad():
                preds = model(batch)
                preds = preds.cpu().numpy()
                for pred, name in zip(preds, names):
                    base_name = Path(name).stem
                    npy_path = self.prediction_path / f"{base_name}.npy"
                    np.save(npy_path, pred)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return np.load(self.files[idx], allow_pickle=True)
