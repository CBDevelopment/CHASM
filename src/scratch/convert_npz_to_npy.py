import os
import numpy as np

def convert_npz_to_npy(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.npz'):
                npz_path = os.path.join(root, filename)
                with np.load(npz_path) as data:
                    arr = data['data']
                npy_path = os.path.join(root, filename[:-4] + '.npy')
                np.save(npy_path, arr)
                os.remove(npz_path)

if __name__ == "__main__":
    folder = "data_test_set\chronnos_processed_data"  # Change this to your folder path
    convert_npz_to_npy(folder)