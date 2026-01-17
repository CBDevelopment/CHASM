import os
from download.datasets import DrawingsDataset, CHASMDataset, AIADataset, CombinedDataset, FITSDataset
from CHASM.preprocessing.sam.preprocess import preprocess_sam
from CHASM.app.app import run_app
from CHASM.preprocessing.circle_detection import SolarDiskDetector, crop_masks
from src.polished_pipeline.train_model import train_models_swpc, preprocess_chronnos, remove_unmatched_files_recursive

def download_data(path="data", includes='all') -> None:
    """
    Download files based on the 'includes' option.
    includes: 'all', 'drawings', or 'masks'
    """
    if includes=='all':
        includes = ['drawings', 'masks', 'aia'] # TODO add AIA after getting that uploaded to Google Drive

    if os.path.exists(path):
        print(f"Path already exists! Nothing is being downloaded to {path}. Exiting.")
        return 
    
    if 'drawings' in includes:
        DrawingsDataset(root=os.path.join(path, "drawings"), fetch_online=True)

    if 'masks' in includes:
        CHASMDataset(root=os.path.join(path, "chasm"), fetch_online=True)

    if 'aia' in includes:
        AIADataset(root=os.path.join(path, "aia"), fetch_online=True)


# TODO add a default download for aia in download_data as well
def collect_aia_data(base_path: str | CombinedDataset, year: int = None) -> None:
    """
    Collect AIA data from the specified drawings path.
    drawings_path: Path to drawings directory.
    """
    pass

    
def preprocess_sam_drawings(base_dir: str | CombinedDataset, checkpoint) -> None:
    """
    Preprocess SAM drawings from the specified path.
    drawings_path: Path to drawings directory.
    """
    if type(base_path)==CombinedDataset:
        base_path = base_path.get_root()
    preprocess_sam(base_dir, checkpoint=checkpoint)

def crop_masks_to_solar_disc(base_path: str | CombinedDataset, year: str) -> None:

    crop_masks(base_path, year)

def train_chronnos_model(base_path: str | CombinedDataset, convert: bool = True) -> None:
    """
    Train Chronnos model on the given data path.
    base_path: Path to training data.
    """
    if type(base_path)==CombinedDataset:
        base_path = base_path.get_root()
    train_models_swpc(base_path, convert=convert)

def run_chasm(base_path: str, year: int = None) -> None:
    """
    Run CHASM processing from the given base path.
    base_path: Path to CHASM base directory.
    """
    if type(base_path)==CombinedDataset:
        base_path = base_path.get_root()
    run_app(base_path, year=year) # If broken up into years, this would handle it

def preprocess_chronnos_data(
    base_path: str = None,
    dest_path: str = None,
    verbose: bool = True,
) -> None:
    """
    Preprocess CHRONNOS data.
    # TODO document
    """
    if type(base_path)==CombinedDataset:
        base_path = base_path.get_root()
    preprocess_chronnos(base_path, dest_path, verbose)

# TODO have this take FITS
# def preprocess_chronnos_data_dataset(fits: FITSDataset):

if __name__=="__main__":
    path = r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\data_test_set_5"
    #download_data(path, includes="all")
    #preprocess_chronnos(path, True)
    #remove_unmatched_files_recursive(os.path.join(path, "chronnos_inputs", "map"), os.path.join(path, "chronnos_inputs", "mask"))
    #remove_unmatched_files_recursive(os.path.join(path, "chronnos_inputs", "mask"), os.path.join(path, "chronnos_inputs", "map"))
    # TODO add NAN check for x and y in preprocessing
    preprocess_chronnos(r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\data_test_set_5",
                        r"C:\Users\evang\Desktop\WPI\Coronal Holes ISP\CoronalHoles\chronnos_inputs_test_11",
                        verbose=True)
    #train_chronnos_model(path, convert=False)