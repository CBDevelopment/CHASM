import sys
import numpy as np

def print_npz_contents(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        #data = data["info"]
        #print(data["data"])
        print(data["Quality"])
        print(f"Keys in '{npz_path}': {list(data.keys())}")
        # for key in data:
        #     print(f"\nKey: {key}")
        #     print(f"Shape: {data[key].shape}")
        #     print(f"Data:\n{data[key]}")

if __name__ == "__main__":

    print_npz_contents(r"chasm_data\2024new\DATA-boul_neutl_fd_20240103_0500.npz")