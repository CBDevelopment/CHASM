import numpy as np

# NOTE: This will be removed, just checking new CHASM seutp has the intergrated good quality instead of the extra .csv file

# Replace with your .npz file path
npz_path = r"D:\CHASM_TABLES\PREDICTION_PIPELINE_TEST_2\chasm\2019\2019-01-03T06-31_CHASMSelection.npz"

with np.load(npz_path) as data:
    print("Keys in the .npz file:")
    print(list(data.keys()))