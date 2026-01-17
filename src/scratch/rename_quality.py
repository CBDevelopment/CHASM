import os
import numpy as np
from tqdm import tqdm

def rename_quality_keys(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.npz'):
            input_path = os.path.join(input_folder, filename)
            data = np.load(input_path, allow_pickle=True)
            new_data = {}
            for key in data.files:
                if key == "Quality":
                    # Assume Quality is an array of "Good"/"Bad" strings
                    quality = data[key]
                    if quality=="Good":
                        new_data["good_quality"] = True
                    else:
                        new_data["good_quality"] = False
                else:
                    new_data[key] = data[key]
            output_path = os.path.join(output_folder, filename)
            np.savez(output_path, **new_data)

if __name__ == "__main__":
    input_folder = r"download_data\chasm_as_npz_QualityColumn\2024"
    output_folder = r"download_data\chasm_as_npz\2024"
    rename_quality_keys(input_folder, output_folder)