import argparse
import gzip
import os

from matplotlib import pyplot as plt
from sunpy.map import Map
from tqdm import tqdm

from MultiChannelCHDetection.chronnos.data.convert import get_intersecting_files, sdo_norms
from MultiChannelCHDetection.chronnos.evaluate.detect import CHRONNOSDetector
import torch

from datasets.drawings_dataset import DrawingsDataset
from datasets.aia_dataset import AIADataset
from datasets.chasm_dataset import CHASMDataset, CHASM960, CHASM1111, CHASM1407
from datasets.fits_dataset import FITSDataset
from datasets.chronnos_dataset import CHRONNOSDataset, InferenceChronnosConfig, DefaultChronnosConfig
from datasets.prediction_dataset import PredictionDataset

# TODO document this
def build_prediction_dataset(
    drawings=None,
    aia=None,
    chasm=None,
    fits=None,
    chronnos=None,
    prediction_path=None,
    model_path=None,
    device="cuda",
    force_recompute=False,
    fetch_online=True,
    num_workers=8,
):
    """
    Build and return a PredictionDataset from given dataset objects or root paths.

    Args:
        drawings (str | DrawingsDataset): path to drawings folder or a dataset object
        aia (str | AIADataset): path to AIA folder or a dataset object
        chasm (str | CHASM1407): path to CHASM folder or a dataset object
        fits (str | FITSDataset): path to FITS folder or a dataset object
        chronnos (str | CHRONNOSDataset): path to CHRONNOS folder or a dataset object
        prediction_path (str): path where predictions are stored
        model_path (str): path to model checkpoint
        data_path (str): path containing chronnos_merged_inputs
        device (str): device string ("cuda" or "cpu")
        force_recompute (bool): whether to recompute predictions

    Returns:
        PredictionDataset
    """

    # Wrap paths into dataset objects if necessary
    
    aia_ds = (
        aia if isinstance(aia, AIADataset)
        else AIADataset(root=aia, fetch_online=fetch_online) if aia else None
    )
    
    drawings_ds = (
        drawings if isinstance(drawings, DrawingsDataset)
        else DrawingsDataset(root=drawings, fetch_online=fetch_online) if drawings else None
    )

    chasm_ds = (
        chasm if isinstance(chasm, CHASMDataset)
        # TODO instantiation of genertic dataset type doesn't work here. need to fix
        else CHASMDataset(root=chasm, fetch_online=fetch_online, needs_aia=True) if chasm else None
    )

    fits_ds = (
        fits if isinstance(fits, FITSDataset) # TODO in documentation and everything make distinction between this and internal FITSDataet from original repo
        else FITSDataset(root=fits) if fits else None # TODO check what difference calibrate makes
    )

    chronnos_ds = (
        chronnos if isinstance(chronnos, CHRONNOSDataset)
        else CHRONNOSDataset(root=chronnos, config=InferenceChronnosConfig()) if chronnos else None
    )

    # ---- Build sub-datasets where needed ----
    fits_ds.build(drawings=drawings_ds, chasm=chasm_ds)
    chronnos_ds.build(aia_ds, fits_ds, inference=True)

    # ---- Prediction dataset ----
    prediction_dataset = PredictionDataset(prediction_path=prediction_path)

    # Compute intersecting map paths if data_path is given
    # map_paths = None
    # if data_path:
    #     map_paths = get_intersecting_files(
    #         data_path,
    #         dirs=[94, 131, 171, 193, 211, 304, 335, 6173]
    #     )

    # Build prediction dataset
    prediction_dataset.build(
        model_path=model_path,
        chronnos_dataset=chronnos_ds,
        device=device,
        num_workers=num_workers,
    )

    return prediction_dataset

def build_prediction_from_base(basefolder: str, model_path: str):
    return build_prediction_dataset(
        drawings=os.path.join(basefolder, "drawings"),
        aia=os.path.join(basefolder, "aia"),
        chasm=CHASM1111(
            root=os.path.join(basefolder, "chasm"),
            fetch_online=True,
            needs_aia=False
        ),
        fits=os.path.join(basefolder, "fits"),
        chronnos=os.path.join(basefolder, "chronnos"),
        prediction_path=os.path.join(basefolder, "predictions"),
        model_path=model_path,
        device="cuda",
        force_recompute=False,
    )

def main():
    parser = argparse.ArgumentParser(description="Build prediction dataset from a base folder")
    parser.add_argument("model_path", help="Path to the trained model file (.pt)")
    parser.add_argument("basefolder", help="Top-level folder containing drawings, aia, chasm, fits, chronnos_maps, predictions")
    args = parser.parse_args()

    prediction_ds = build_prediction_from_base(args.basefolder, args.model_path)
    print("Prediction dataset built successfully!")

if __name__ == "__main__":
    main()
