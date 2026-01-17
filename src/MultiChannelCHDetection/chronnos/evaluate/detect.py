import os
from pathlib import Path
from urllib import request

import numpy as np
import torch
from sunpy.map import all_coordinates_from_map
from torch.utils.data import DataLoader
from tqdm import tqdm

from MultiChannelCHDetection.chronnos.data.convert import buildFITS, get_intersecting_files
from MultiChannelCHDetection.chronnos.data.generator import FITSDataset
from sunpy.map import Map

class CHRONNOSDetector:

    def __init__(self, model_path=None, model_name='chronnos_v1_0.pt', device=None):
        """Initializes a CHRONNOS model for detections.

        Specify either model_path or model_name. A local copy of 'model_name' is downloaded to the home directory.

        :param model_path: path to a local .pt file
        :param model_name: name of the CHRONNOS model
        :param device: optional device to use
        """
        assert (model_path is not None) or (
                    model_name is not None), 'Either model_path or model_name need to be specified.'
        model_path = load_model_path(model_name) if model_path is None else model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

    def predict_dir(self, data_path, dirs=None, **kwargs):
        """Find files in directory and detect coronal holes.

        :param data_path: base path to the data
        :param dirs: directories to scan
        :return: detected coronal holes as list of SunPy Maps
        """
        dirs = ['94', '131', '171', '193', '211', '304', '335', '6173'] if dirs is None else dirs
        map_paths = get_intersecting_files(data_path, dirs=dirs)
        return self.predict(map_paths, **kwargs)

    def predict(self, map_paths, reproject=False, num_workers=4, calibrate=True, threshold=False):
        """Detect coronal holes for a set of observations.

        Use predict_dir to auto scan a directory.

        :param map_paths: paths of the FITS files. The format is (channel, file_path).
        :param calibrate: adjust device degradation and exposure time. For pre-processed files set calibration=False.
        :return: detected coronal holes as list of SunPy Maps
        """
        map_ds = FITSDataset(list(zip(*map_paths)), calibrate=calibrate)
        map_loader = DataLoader(map_ds, batch_size=2, shuffle=False, num_workers=num_workers)

        predictions = []
        with torch.no_grad():
            for batch in tqdm(map_loader, desc='model prediction'):
                batch = batch.to(self.device)
                pred = self.model(batch).detach().cpu()
                if threshold!=False:
                    pred = pred >= threshold
                predictions.append(pred)

        predictions = torch.cat(predictions, 0).numpy()

        res_maps = []
        for prediction, ref in tqdm(zip(predictions, map_paths[3]), desc='converting to Map', total=len(predictions)):
            s_map = buildFITS(prediction[0], ref, reproject=reproject, threshold=threshold)
            hpc_coords = all_coordinates_from_map(s_map)
            r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
            s_map.data[r > 1] = 0
            #
            res_maps += [s_map]
        return res_maps

    def ipredict(self, map_paths, reproject=False, num_workers=4, calibrate=True, threshold=False):
        """Detect coronal holes for a set of observations.

        The results are returned as data stream.

        :param map_paths: paths of the FITS files. The format is (channel, file_path).
        :param calibrate: adjust device degradation and exposure time. For pre-processed files set calibration=False.
        :return: generator object that returns SunPy Maps
        """
        # TODO make this so that the FITS Dataset is act
        map_ds = FITSDataset(list(zip(*map_paths)), calibrate=calibrate)

        map_loader = DataLoader(map_ds, batch_size=1, shuffle=False, num_workers=num_workers)

        with torch.no_grad():
            for batch, ref in tqdm(zip(map_loader, map_paths[3]), desc="Predicting"):
                batch = batch.to(self.device)
                pred = self.model(batch).detach().cpu()
                print("THRESHOLDING ", threshold)
                print("PRED ", pred)
                if threshold!=False:
                    pred = pred >= threshold
                pred = pred[0, 0].numpy()  # remove batch and channel
                s_map = buildFITS(pred, ref, reproject=reproject, threshold=threshold) # TODO make this not BINARY 
                print("S MAP AFTER BUILD FITS ", s_map.data)
                hpc_coords = all_coordinates_from_map(s_map)
                r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / s_map.rsun_obs
                s_map.data[r > 1] = 0
                yield s_map

    # TODO should be able to get rid of this since CHRONNOS_Dataset should do the preprocessing with InferenceConfig
    def ipredict_numpy(self, map_paths, reproject=False, crop=False, num_workers=8, calibrate=True, threshold=False, save_dir=None):
        """
        Predict coronal holes for a set of observations, return/save NumPy arrays instead of SunPy Maps.

        :param map_paths: paths of the FITS files. Format is (channel, file_path).
        :param calibrate: adjust device degradation and exposure time.
        :param save_dir: optional directory to save .npy files.
        :return: generator of (prediction_array, reference_path)
        """
        map_ds = FITSDataset(list(zip(*map_paths)), calibrate=calibrate)
        if num_workers==1:
            map_loader = DataLoader(map_ds, batch_size=1, shuffle=False)
        else:
            map_loader = DataLoader(map_ds, batch_size=1, shuffle=False, num_workers=num_workers)

        # TODO fix inhomogenous shape error that occurs here in building
        with torch.no_grad():
            for batch, ref in tqdm(zip(map_loader, map_paths[3]), desc="Predicting"):
                batch = batch.to(self.device)
                pred = self.model(batch).detach().cpu()

                if threshold is not False:
                    pred = pred >= threshold

                pred = pred[0, 0].numpy()  # remove batch and channel
                #print("PREDICTION SHAPE ", pred.shape)
                # import matplotlib.pyplot as plt

                # plt.imshow(pred, cmap='gray')
                # plt.title("Prediction")
                # plt.colorbar()
                # plt.show()
                
                # Optional: crop outside solar disk
                ref_map = Map(ref)
                hpc_coords = all_coordinates_from_map(ref_map)
                r = np.sqrt(hpc_coords.Tx**2 + hpc_coords.Ty**2) / ref_map.rsun_obs
                if crop:
                    pred[r > 1] = 0

                if save_dir is not None:
                    out_path = Path(save_dir) / (Path(ref).stem + ".npy")
                    np.save(out_path, pred)
                
                yield pred

# TODO should be able to remove this
def load_model_path(model_name):
    """
    Get the path to the local model and download a copy if it does not exist.

    :param model_name: model identifier
    :return: path to the local model
    """
    model_path = os.path.join(Path.home(), '.chronnos', model_name)
    os.makedirs(os.path.join(Path.home(), '.chronnos'), exist_ok=True)
    if not os.path.exists(model_path):
        request.urlretrieve('http://kanzelhohe.uni-graz.at/iti/' + model_name, filename=model_path)
    return model_path