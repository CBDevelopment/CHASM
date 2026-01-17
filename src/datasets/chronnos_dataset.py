
import os
from pathlib import Path
import warnings
import astropy 
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize, LinearStretch
from astropy.nddata import block_reduce
from astropy import units as u
from aiapy.calibrate import correct_degradation
from multiprocessing import Pool
from datasets.folder_utils import extract_date
from aiapy.calibrate.util import get_correction_table
from sunpy.map import Map, all_coordinates_from_map
from sunpy.map import Map as sunpyMap
from sunpy.map.sources import HMIMap
from MultiChannelCHDetection.chronnos.train.model import Trainer
from MultiChannelCHDetection.chronnos.data.convert import sdo_norms_dict


import shutil
import logging
import re
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch
from dataclasses import dataclass, field

# TODO make this not global?

# Create output directories for each wavelength and masks

logging.basicConfig(
level=logging.INFO,
handlers=[
    logging.StreamHandler()
])

logging.info('====================== CONVERTING DATA ======================')

@dataclass(frozen=True)
class DefaultChronnosConfig:
    image_channels: int = 8
    start_resolution: int = 8
    n_dims: list[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64, 32, 16])
    n_convs: list[int] = field(default_factory=lambda: [3, 3, 3, 2, 2, 2, 1])
    resolutions: list[int] = field(init=False)
    channels: list = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "resolutions",
            [self.start_resolution * 2 ** i for i in range(len(self.n_dims))]
        )

@dataclass(frozen=True)
class InferenceChronnosConfig:
    image_channels: int = 8
    start_resolution: int = 512
    n_dims: list[int] = field(default_factory=lambda: [512])  # one stage at 512
    n_convs: list[int] = field(default_factory=lambda: [3])   # number of convs at 512
    resolutions: list[int] = field(init=False)
    channels: list = None

    def __post_init__(self):
        object.__setattr__(self, "resolutions", [self.start_resolution])

class _MapConverter:

    def __init__(self, converted_path, resolutions, correction_table, replace):
        self.converted_path = converted_path
        self.resolutions = resolutions
        self.replace = replace
        self.max_res = max(resolutions)
        self.correction_table = correction_table

        sdo_norms = [ImageNormalize(vmin=0, vmax=445.5, stretch=AsinhStretch(0.005), clip=True),  # 94
            ImageNormalize(vmin=0, vmax=981.3, stretch=AsinhStretch(0.005), clip=True),  # 131
            ImageNormalize(vmin=0, vmax=6457.5, stretch=AsinhStretch(0.005), clip=True),  # 171
            ImageNormalize(vmin=0, vmax=7757.31, stretch=AsinhStretch(0.005), clip=True),  # 193
            ImageNormalize(vmin=0, vmax=6539.8, stretch=AsinhStretch(0.005), clip=True),  # 211
            ImageNormalize(vmin=0, vmax=3756, stretch=AsinhStretch(0.005), clip=True),  # 304
            ImageNormalize(vmin=0, vmax=915, stretch=AsinhStretch(0.005), clip=True),  # 335
            ImageNormalize(vmin=-100, vmax=100, stretch=LinearStretch(), clip=True),  # mag
            ]
        self.sdo_norms_dict = {k: v for k, v in zip([94, 131, 171, 193, 211, 304, 335, 6173], sdo_norms)}

    def convert(self, c_files):
        # check if file already exists
        try:
            if all([os.path.exists(
                    os.path.join(self.converted_path, '%d' % res, os.path.basename(c_files[0]).replace('.fits', '.npy')))
                for
                res in self.resolutions]) and self.replace == False:
                #logging.info("HITTING SHORT CIRCUIT RETURN")
                return
            
            #logging.info("GETTING MAP DATA")
            #logging.info("CFILES " + str(c_files))
            maps_data = [self.getMapData(c_file, self.max_res, self.correction_table) for c_file in c_files]
            #logging.info("LOGGING SHAPES ")
            # for md in maps_data:
            #     logging.info(str(md.shape))
                
            maps_data = np.stack(maps_data, -1)

            for resolution in self.resolutions:
                # TODO remove this comment, it was changed before deletion of this file
                # if len(self.resolutions)==1: # If only one resolution, no need to make specific folder for it. Makes inference smoother
                #     path = os.path.join(self.converted_path,
                #                         os.path.basename(c_files[0]).replace('.fits', '.npy'))
                # else:
                path = os.path.join(self.converted_path, '%d' % resolution,
                                    os.path.basename(c_files[0]).replace('.fits', '.npy'))
                block = (maps_data.shape[0] // resolution, maps_data.shape[1] // resolution, 1)
                map_data_reduced = block_reduce(maps_data, block, np.mean)

                np.save(path, map_data_reduced.astype(np.float32))

        except Exception as e: # TODO put this back to general exception
            logging.info(f"Was unable to convert map for {c_files}. ")
            logging.info(str(e))
            #print("Was unable to convert map for {c_files}. ", e)

    def pad_to_square(self, arr, pad_value=0):
            """
            Pads a 2D numpy array to make it square using the larger of the two dimensions.
            Prints a warning if padding is applied.
            
            Parameters:
                arr (np.ndarray): Input 2D array.
                pad_value (numeric, optional): Value to use for padding. Defaults to 0.
            
            Returns:
                np.ndarray: Square padded array.
            """
            if arr.ndim != 2:
                raise ValueError("Input array must be 2D")

            rows, cols = arr.shape
            if rows == cols:
                return arr  # Already square
            
            max_size = max(rows, cols)
            pad_rows = max_size - rows
            pad_cols = max_size - cols

            print(f"Warning: array is not square ({rows}x{cols}). Padding to {max_size}x{max_size}.")
            
            # pad ((top, bottom), (left, right))
            padded_arr = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=pad_value)
            return padded_arr

    def getMapData(self, file, resolution, correction_table=None, remove_off_disk=False, calibrate=True, needs_exptime=False):
        """Returns the prepared data of the FITS file.

        :param file: the FITS map file
        :param resolution: pixels along x- and y-axis
        :param correction_table: the AIA correction table (aiapy.calibrate.util.get_correction_table)
        :param remove_off_disk: True to set all off-limb pixels to min. HMI images are automatically truncated.
        :return: 2D numpy array
        """
        s_map = Map(file)
        s_map = prepMap(s_map, resolution) # TODO figure out why this works here, but isn't used in _MaskConvert now to make things work, despite being in OG code
        #
        if isinstance(s_map, HMIMap): # truncate boundary for HMI images
            data = np.nan_to_num(s_map.data)
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            data[r > 1] = 0
        elif calibrate:
            s_map = correct_degradation(s_map, correction_table=correction_table)
            data = np.nan_to_num(s_map.data)

            if "exptime" in s_map.meta:
                data = data / s_map.meta["exptime"]

            # Error if exptime needed
            if "exptime" not in s_map.meta and needs_exptime==True:
                logging.error(f"{file} missing EXPTIME but needs_exptime=True. Exiting.")
                raise ValueError(f"{file} missing EXPTIME but needs_exptime=True.")

        else:
            data = np.nan_to_num(s_map.data)
        if remove_off_disk:
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            data[r > 1] = np.min(s_map.data)
        data = self.sdo_norms_dict[int(s_map.wavelength.value)](data).data
        data = data.astype(np.float32)
        data = self.pad_to_square(data)
        return data
    
    
class MapDataset(Dataset):
    def __init__(self, files, channel=None):
        """
        Load saved maps for model training.
        :param files: list of npy files
        :param channel (optional): select subset of channels (idx)
        """
        self.files = files
        self.channel = None if channel is None else channel if isinstance(channel, list) else [channel]
        super().__init__()

    def __getitem__(self, index):
        file = self.files[index]
        x = np.load(file)
        x = x * 2 - 1  # scale to [-1, 1]
        x = np.transpose(x, axes=[2, 0, 1])
        if self.channel is not None:
            x = x[self.channel]
        return np.array(x.data.tolist(), dtype=np.float32), file

    def __len__(self):
        return len(self.files)
    
# TODO make not just a floating function
def prepMap(s_map: Map, resolution: int, padding_factor: float = 0.1):
        """Returns the adjusted map. The solar disk is centered to the image,
        the north pole axis aligned with the y-axis and the scale adjusted to (1 + padding) * R_Sun[arcsec] / resolution

        :param s_map: '~sunpy.map.Map' object
        :param resolution: pixels along x- and y-axis
        :param padding_factor: distance between solar limb and image border given in R_Sun
        :return: adjusted sunpy map
        """
        warnings.simplefilter("ignore")  # ignore warnings
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + padding_factor) * r_obs_pix
        scale_factor = resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=3)
        arcs_frame = (resolution / 2) * s_map.scale[0].value
        s_map = s_map.submap(SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                            top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
        # remove overlap after submap
        pad_x = s_map.data.shape[0] - resolution
        pad_y = s_map.data.shape[1] - resolution
        s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                            top_right=[pad_x // 2 + resolution - 1, pad_y // 2 + resolution - 1] * u.pix)
        #
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']
        return s_map


class _MaskConverter:
    def __init__(self, converted_path, resolutions, replace):
        self.converted_path = converted_path
        self.resolutions = resolutions
        self.replace = replace
        self.max_res = max(resolutions)

    # conversion function
    # TODO check if this is a reasonable conversion function? In line with original code?
    def convert(self, file):
        try:
            # check if file already exists
            if all([os.path.exists(os.path.join(self.converted_path, '%d' % res,
                                                # TODO note this was .fits.gz. If both need to be supported that will need to be added
                                                os.path.basename(file).replace('.fits', '.npy')))
                    for res in self.resolutions]) and self.replace == False:
                return
            
            #s_map = Map(file)  # TODO figure out why giving issues, check original code to compare (maybe not use prepMap?)
            
            s_map = fits.getdata(file) # NOTE: This was using the "dummy" fits to properly resize. Since we are treating just like an image this isn't needed. TODO could just have the CHRONNOS system we have use npy instead for this reason all the way through.

            #s_map = prepMap(s_map, self.max_res) # TODO figure out why prepMap isn't working for the "Cropped centered masks". Is the FITS file misconfigured somehow?
            #data = s_map.data > 0.3  # back to binary
            data = s_map > 0.3
            data = data[..., None] # expand last dimension
            for resolution in self.resolutions:
                path = os.path.join(self.converted_path, '%d' % resolution,
                                    # TODO note this was .fits.gz. If both need to be supported that will need to be added
                                    os.path.basename(file).replace('.fits', '.npy'))
                block = (data.shape[0] // resolution, data.shape[1] // resolution, 1)
                mask_data_reduced = block_reduce(data, block, np.mean)
                np.save(path, mask_data_reduced.astype(np.float32))
            
        # TODO need to figure out why this often doesn't work, especially for folder 512
        except Exception as e:
            print(f"Could not convert mask for {file}, {e}")

class MaskDataset(Dataset):
    def __init__(self, files):
        """
        Load saved masks for model training.
        :param files: list of npy files
        """
        self.files = files
        super().__init__()

    def __getitem__(self, index):
        file = self.files[index]
        y = np.load(file)
        y = (y >= 0.1).astype(np.float32)  # make hard labels
        y = np.transpose(y, axes=[2, 0, 1])
        return np.array(y.data.tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.files), file
    
class CHRONNOSDataset():

    # TODO add resolutiosn
    # TODO check if whole config is needed
    def __init__(self, root="download_data/chronnos", config = DefaultChronnosConfig()):
        self.root = root
        self.config = config
        self.grouped_files = None

    @staticmethod
    def get_intersecting_files_date(path, dirs, extensions=None):
        #logging.info("GETTING INTERSECTING FILES")
        path = Path(path)
        extensions = extensions if extensions is not None else ['.fits'] * len(dirs)

        # Collect basenames for each directory
        dir_to_files = []
        for directory, ext in tqdm(zip(dirs, extensions), desc="Zipping Intersecting Files", total=min(len(dirs), len(extensions))):
            folder = path / str(directory)
            files = {f.name for f in folder.iterdir() if f.suffix == ext}
            dir_to_files.append(files)

        # Find intersection
        common_basenames = sorted(set.intersection(*dir_to_files))

        # Build grouped full paths
        grouped_files = [
            [str(path / str(directory) / basename) for directory in dirs]
            for basename in common_basenames
        ]

        return grouped_files

    def get_local_correction_table(self):
        """Load AIA correction table from home directory.

        Downloads a new table if no file exists.

        :return: the correction table
        """
        path = os.path.join(Path.home(), 'aiapy', 'correction_table.dat')
        if os.path.exists(path):
            return get_correction_table(path)
        os.makedirs(os.path.join(Path.home(), 'aiapy'), exist_ok=True)
        correction_table = get_correction_table("jsoc") # NOTE: Changed source to JSOC to prevent errors
        astropy.io.ascii.write(correction_table, path)
        return correction_table

    # Convert maps and all the needed helper functions
    def convertMaps(self, grouped_files, converted_path, resolutions, n_workers=8, replace=False):
        """Convert FITS files for model training.

        :param grouped_files: list of FITS files in the format (channel, file)
        :param converted_path: path where the converted data is stored
        :param resolutions: set of resolutions used for training
        :param n_workers: number of parallel worker threads
        :param replace: replace existing files
        :return: None
        """

        [os.makedirs(os.path.join(converted_path, '%d' % res), exist_ok=True) for res in resolutions]
        correction_table = self.get_local_correction_table()
        #logging.info("RESOLUTIONS GOING INTO MAP {resolutions}")
        converter = _MapConverter(converted_path, resolutions, correction_table, replace)
        # async conversion
        # TODO could add sync and async option to build that gets passed down
        for g in tqdm(np.array(grouped_files), desc="Iterating through grouped files", total=len(grouped_files)):
            converter.convert(g)

        # with Pool(n_workers) as p:
        #     [None for _ in tqdm(p.imap_unordered(converter.convert, np.array(grouped_files)), total=len(grouped_files[0]))]
    
    def convertMasks(self, files, converted_path, resolutions, n_workers=8, replace=False):
        """Convert label masks for model training.

        :param files: list of FITS files
        :param converted_path: path where the converted data is stored
        :param resolutions: set of resolutions used for training
        :param n_workers: number of parallel worker threads
        :param replace: replace existing files
        :return: None
        """
        [os.makedirs(os.path.join(converted_path, '%d' % res), exist_ok=True) for res in resolutions]
        converter = _MaskConverter(converted_path, resolutions, replace)

        # async conversion
        with Pool(n_workers) as p:
            [None for _ in tqdm(p.imap_unordered(converter.convert, files), total=len(files))]

    # Magnetometer resizing stuff
    @staticmethod
    def process_single_magnetometer(fits_file, output_path):
        """Process a single FITS file to create a resampled version."""
        try:
            # Create and resample the map
            mag_map = sunpy.map.Map(fits_file)
            resampled_map = mag_map.resample([512, 512]*u.pix)

            # Create a new header with proper ordering
            new_header = fits.Header()

            # Add required keywords in the correct order
            new_header['SIMPLE'] = True
            new_header['BITPIX'] = -32
            new_header['NAXIS'] = 2
            new_header['NAXIS1'] = 512
            new_header['NAXIS2'] = 512

            # Get original header and copy keywords
            with fits.open(fits_file) as hdul:
                original_header = hdul[1].header

                # Copy essential keywords
                essential_keys = [
                    'TELESCOP', 'INSTRUME', 'WAVELNTH', 'WCSNAME',
                    'CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2',
                    'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
                    'CUNIT1', 'CUNIT2', 'RSUN_OBS', 'RSUN_REF', 'R_SUN',
                    'DATE-OBS', 'T_OBS', 'T_REC', 'BUNIT'
                ]

                for key in essential_keys:
                    if key in original_header:
                        try:
                            new_header[key] = original_header[key]
                        except Exception as e:
                            print(
                                f"Warning: Could not copy keyword {key} for {fits_file}: {str(e)}")

                # Copy HIERARCH keywords
                for key in original_header:
                    if key.startswith('HIERARCH'):
                        try:
                            new_header[key] = original_header[key]
                        except Exception as e:
                            print(
                                f"Warning: Could not copy HIERARCH keyword {key} for {fits_file}: {str(e)}")

            # Save resampled file
            compressed_resampled_map = CompImageHDU(
                data=resampled_map.data,
                header=new_header,
                compression_type='HCOMPRESS_1',
                quantize_level=16.0
            )
            compressed_hdul = HDUList([PrimaryHDU(), compressed_resampled_map])
            compressed_hdul.writeto(output_path, overwrite=True)
            return f"Successfully processed {fits_file}"

        except Exception as e:
            return f"Error processing {fits_file}: {str(e)}"

    # TODO delete this
    # def resize_magnetometers(self, path):
    #     # Get list of FITS files
    #     fits_files = sorted([f for f in os.listdir(f"{self.root}/{self.year}/{self.wavelength}")
    #                         if f.endswith(".fits")])

    #     # Create output directory if it doesn't exist
    #     os.makedirs(f"{self.root}/{self.year}/resampled_mag", exist_ok=True)

    #     # Process files in parallel
    #     num_processes = 8
    #     with Pool(processes=num_processes) as pool:
    #         results = pool.map(self.process_single_file, fits_files)

    # TODO document, but returns a dataset just to iterate over for prediction inputs from the CHRONNOS preprocessing
    def getPredictionInputsDataset(self):
        class NumpyFolderDataset(Dataset):
            def __init__(self, folder_path, transform=None):
                self.folder_path = folder_path
                self.transform = transform
                # Only include image files
                self.files = [f for f in os.listdir(folder_path) if f.endswith((".npy"))]

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                file_path = os.path.join(self.folder_path, self.files[idx])
                array = np.load(file_path)
                tensor = torch.from_numpy(array).float()  # convert to float tensor
                return tensor, file_path
            
        return NumpyFolderDataset(os.path.join(self.root, 'inputs', 'map'))

    def get_inputs_path(self):
        return os.path.join(self.root, "inputs")
    
    def get_map_path(self):
        return os.path.join(self.root, "inputs", 'map')

    def get_mask_path(self):
        return os.path.join(self.root, "inputs", 'mask')
    
    def get_map_dataset(self, resolution=512):
        map_dir = os.path.join(self.get_map_path(), str(resolution))
        files = [os.path.join(map_dir, f) for f in os.listdir(map_dir)]
        return MapDataset(files)

    def get_mask_dataset(self, resolution=512):
        mask_dir = os.path.join(self.get_mask_path(), str(resolution))
        files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        return MaskDataset(files)
    
    def get_intersecting_files(self):
        # TODO could make these constants in the class definition, cleaner setup
        intersecting_dirs = ['94', '131', '171', '193', '211', '304', '335', '6173', 'cropped_centered_masks']
        chronnos_raw_inputs = os.path.join(self.root, "chronnos_merged_inputs")
        return CHRONNOSDataset.get_intersecting_files_date(chronnos_raw_inputs,
                                                           dirs=intersecting_dirs, 
                                                           extensions=['.fits'] * 8)
    
    def build(self, aia, fits, verbose=False, inference=True):
        if os.path.exists(self.root):
            print(f"Path to root {self.root} exists. Exiting build. ")
            return

        os.makedirs(self.root)
        chronnos_raw_inputs = os.path.join(self.root, "chronnos_merged_inputs")

        intersecting_dirs = ['94', '131', '171', '193', '211', '304', '335', '6173', 'cropped_centered_masks']
        # if not os.path.exists(chronnos_raw_inputs):
        #     CHRONNOSDataset.merge_directories(aia_path=aia.root, fits_path=fits.root, output_path=chronnos_raw_inputs)
        #     self.grouped_files = CHRONNOSDataset.get_intersecting_files_date(chronnos_raw_inputs,
        #                                         dirs=intersecting_dirs, 
        #                                         extensions=['.fits'] * 8) # NOTE: Making this last folder .fits instead of .fits.gz made it work
        # else:
        #     self.grouped_files = [
        #                     [str(p) for p in sorted(Path(chronnos_raw_inputs).joinpath(d).glob("*.fits"))]
        #                     for d in intersecting_dirs
        #                 ]

        CHRONNOSDataset.merge_directories(aia_path=aia.root, fits_path=fits.root, output_path=chronnos_raw_inputs,
                                          merge_dirs=intersecting_dirs)
        self.grouped_files = CHRONNOSDataset.get_intersecting_files_date(chronnos_raw_inputs,
                                            dirs=intersecting_dirs, 
                                            extensions=['.fits'] * 8)
        
        # TODO add nan checked in these functions
        # TODO don't use converted_path and use root instead
        os.makedirs(Path(self.root) / "inputs")
        map_path = self.get_map_path()
        mask_path = self.get_mask_path()
        os.makedirs(map_path)
        os.makedirs(mask_path)

        # TODO need this to work with normal convertMaps/convertMasks since otherwise the model breaks
        logging.info('====================== CONVERTING MAPS ======================')
        #logging.info(self.grouped_files)
        self.convertMaps([g[:-1] for g in self.grouped_files[:-1]], map_path, self.config.resolutions)
        logging.info('====================== CONVERTING MASKS  ======================')
        self.convertMasks([g[-1] for g in self.grouped_files[:-1]], mask_path, self.config.resolutions)

        if not inference:
            CHRONNOSDataset.remove_unmatched_files_recursive(map_path, mask_path) # Make sure file names match (prune out non matching from missing aia wavelengths, etc.)
            CHRONNOSDataset.remove_unmatched_files_recursive(mask_path, map_path)

    def train_models_swpc(self, results_path: str = None):

        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.StreamHandler()
            ])

        config = DefaultChronnosConfig()
        channels = None if config.channels is None else [list(sdo_norms_dict.keys()).index(int(wl)) for wl in config.channels]
        logging.info('====================== INIT TRAINER ======================')
        converted_path = self.get_inputs_path()
        trainer = Trainer(results_path, converted_path, image_channels=config.image_channels, start_resolution=config.start_resolution,
                        n_dims=config.n_dims, n_convs=config.n_convs, channels=channels)
        trainer.train() # NOTE: Was trainer.train(args.compress.lower() == 'true') check if this gets passed in like this. I don't see that argument actually getting used?

    @staticmethod
    def merge_directories(aia_path, fits_path, output_path, merge_dirs=None):

        # Output base
        os.makedirs(output_path, exist_ok=True)

        # ---- Merge AIA ----
        # Get years if present (folder has years has wavelengths) or just wavelengths (folder has wavelengths)
        years = [d for d in os.listdir(aia_path)
                if os.path.isdir(os.path.join(aia_path, d)) and
                    any(os.path.isdir(os.path.join(aia_path, d, sub))
                        for sub in os.listdir(os.path.join(aia_path, d)))]
        if not years:
            years = [None]

        for year in tqdm(years, desc="Merging directories AIA [year]"):
            year_path = os.path.join(aia_path, year) if year else aia_path
            if not os.path.isdir(year_path):
                continue

            # Filter only desired wavelength folders if specified
            if merge_dirs:
                wavelength_folders = [w for w in os.listdir(year_path)
                                    if w in merge_dirs and os.path.isdir(os.path.join(year_path, w))]
            else:
                wavelength_folders = [w for w in os.listdir(year_path)
                                    if os.path.isdir(os.path.join(year_path, w))]

            for wavelength in tqdm(wavelength_folders, desc=f"Merging Directories [Wavelength]"):
                wavelength_path = os.path.join(year_path, wavelength)
                if not os.path.isdir(wavelength_path):
                    continue

                target_folder = os.path.join(output_path, wavelength)
                os.makedirs(target_folder, exist_ok=True)

                for file_name in tqdm(os.listdir(wavelength_path), desc="Merging Directories [File Names]"):
                    ext = os.path.splitext(file_name)[-1]
                    date_file_name = extract_date(file_name) # Use consistent naming convention to work for CHRONNOS preprocessing
                    src_file = os.path.join(wavelength_path, file_name)
                    dst_file = os.path.join(target_folder, date_file_name + ext)
                    # TODO remove this, handled now in the AIADataset
                    # if wavelength=="6173": # If magnetogram, need to resize since in different resolution
                    #     CHRONNOSDataset.process_single_magnetometer(src_file, dst_file)
                    # else:
                    shutil.copy2(src_file, dst_file)

        # ---- Merge FITS ----
        cropped_folder = os.path.join(output_path, "cropped_centered_masks")
        os.makedirs(cropped_folder, exist_ok=True)

        # Handle flat or yeared FITS
        years = [d for d in os.listdir(fits_path)
                if os.path.isdir(os.path.join(fits_path, d)) and
                    any(os.path.isdir(os.path.join(fits_path, d, sub))
                        for sub in os.listdir(os.path.join(fits_path, d)))]
        if not years:
            years = [None]

        for year in tqdm(years, desc="Merging Directory FITS [year]"):
            # Handle sorted by year vs flat directory structure
            if year!=None:
                year_path = os.path.join(fits_path, year)
                if not os.path.isdir(year_path):
                    continue

            else:
                year_path = fits_path # Flat directory just use outer directory

            for file_name in os.listdir(year_path):
                ext = os.path.splitext(file_name)[-1]
                date_file_name = extract_date(file_name) # Use consistent naming convention to work for CHRONNOS preprocessing
                if date_file_name==None:
                    print(f"Could not find date for {file_name}")
                    continue

                src_file = os.path.join(year_path, file_name)
                dst_file = os.path.join(cropped_folder, date_file_name + ext)
                shutil.copy2(src_file, dst_file)

    @staticmethod
    def remove_unmatched_files_recursive(folder_a, folder_b):
        """
        Remove files in subfolders of folder_a that do not have a matching filename
        in the corresponding subfolder of folder_b.

        Assumes folder structure like:
            folder_a/
                subfolder1/
                    file1, file2
                subfolder2/
            folder_b/
                subfolder1/
                subfolder2/

        :param folder_a: path to folder where unmatched files will be deleted
        :param folder_b: path to reference folder for matching files
        """
        # Iterate over all subfolders in folder_a
        for subfolder_name in os.listdir(folder_a):
            subfolder_a_path = os.path.join(folder_a, subfolder_name)
            subfolder_b_path = os.path.join(folder_b, subfolder_name)

            # Only process if both subfolders exist
            if not os.path.isdir(subfolder_a_path):
                continue
            if not os.path.isdir(subfolder_b_path):
                print(f"Skipping {subfolder_a_path}, no matching subfolder in {folder_b}")
                continue

            # Get filenames in both subfolders
            files_a = set(os.listdir(subfolder_a_path))
            files_b = set(os.listdir(subfolder_b_path))

            unmatched = files_a - files_b

            for filename in unmatched:
                path_to_remove = os.path.join(subfolder_a_path, filename)
                if os.path.isfile(path_to_remove):
                    print(f"Removing {path_to_remove}")
                    os.remove(path_to_remove)
    
    def __len__(self):
        pass # TODO implement me

    def __getitem__(self, idx):
        pass # TODO implement me


# TODO add usage examples down here