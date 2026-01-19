from .google_drive.drawings_dataset import DrawingsDataset
from .google_drive.aia_dataset import AIADataset
from .google_drive.chasm_dataset import CHASMDataset
from .fits_dataset import FITSDataset
from .chronnos_dataset import (
    CHRONNOSDataset,
    InferenceChronnosConfig,
    DefaultChronnosConfig,
)
from .prediction_dataset import PredictionDataset

"""
Compositional wrapper for datasets involved in predicting coronal holes
"""


class CombinedDataset:
    def __init__(self):
        pass

    # TODO modify this existing (and working) function to fit into the larger ecosystem with this combined dataset
    def build_prediction_dataset(
        self,
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
        drawings_ds = (
            drawings
            if isinstance(drawings, DrawingsDataset)
            else DrawingsDataset(root=drawings, fetch_online=fetch_online)
            if drawings
            else None
        )

        aia_ds = (
            aia
            if isinstance(aia, AIADataset)
            else AIADataset(root=aia, fetch_online=fetch_online)
            if aia
            else None
        )

        chasm_ds = (
            chasm
            if isinstance(chasm, CHASMDataset)
            # TODO instantiation of genertic dataset type doesn't work here. need to fix
            else CHASMDataset(root=chasm, fetch_online=fetch_online, needs_aia=False)
            if chasm
            else None
        )

        fits_ds = (
            fits
            if isinstance(fits, FITSDataset)
            else FITSDataset(root=fits)
            if fits
            else None
        )

        chronnos_ds = (
            chronnos
            if isinstance(chronnos, CHRONNOSDataset)
            else CHRONNOSDataset(root=chronnos, config=InferenceChronnosConfig())
            if chronnos
            else None
        )

        # ---- Build sub-datasets where needed ----
        fits_ds.build(drawings=drawings_ds, chasm=chasm_ds)
        chronnos_ds.build(aia_ds, fits_ds)

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
            model=model_path,
            chronnos_dataset=chronnos_ds,
            device=device,
            force_recompute=force_recompute,
        )

        return prediction_dataset

    # TODO implement CombinedDataset (the old version was quite poor)
    # should just be a composition of smaller datasets with some larger functions
    # like "predict"

    # # TODO implement build_fits to add a fits folder automatically here
    # def __init__(self, root="download_data", fetch_online=True, build_fits=True):
    #     # TODO make sure this works both for downloading and without
    #     self.root = root
    #     self.sam = None
    #     self.app = None
    #     self.app_save_dir = os.path.join(self.root, "chasm")

    #     if fetch_online:
    #         self.google_drive.aia_dataset = AIADataset(root=os.path.join(root, "aia_wavelengths"))
    #         self.google_drive.drawings_dataset = DrawingsDataset(root=os.path.join(root, "drawings"))
    #         self.google_drive.chasm_dataset = CHASMDataset(root=os.path.join(root, "chasm"))

    # def process_sam_masks(self):
    #     self.sam = SAMDataset(root=os.path.join(self.root, "sam_masks"))
    #     self.sam.build()

    # def load_app(self):
    #     if self.sam == None:
    #         print("Cannot load app without SAM masks")
    #     if self.app==None:
    #         self.app = CoronalHoleClassifier(self.google_drive.drawings_dataset.filenames(),
    #                                          self.sam,
    #                                          self.app_save_dir,
    #                                          max_width=1000,
    #                                          max_height=800)

    # def preprocess_chronnos(self):
    #     self.fits_dataset = FITSDataset(root=os.path.join(self.root, "fits"))
    #     self.chronnos = CHRONNOSDataset(root=os.path.join(self.root, "chronnos"))
    #     self.chronnos.build(self.google_drive.aia_dataset, self.fits_dataset)


# Usage example:
if __name__ == "__main__":
    # TODO add example
    pass
