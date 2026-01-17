import numpy as np
import sys
import os

def add_quality_to_npz(npz_path, output_path):
    # Load the .npz file
    data = np.load(npz_path, allow_pickle=True)
    info = data["info"]

    # Prepare quality value
    flags = [entry['flagged'] for entry in info]
    quality = "Good" if all(flag is False for flag in flags) else "Bad"

    # Prepare data to save: copy all arrays, add "Quality"
    new_data = {key: data[key] for key in data.files}
    new_data["Quality"] = quality

    # Save new .npz file
    np.savez(output_path, **new_data)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_quality_to_npz.py <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".npz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            add_quality_to_npz(input_path, output_path)
            print(f"Processed {filename}")