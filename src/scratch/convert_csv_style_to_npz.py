import os
import numpy as np
import pandas as pd
import argparse

def integrate_csv_to_npz(npz_folder, csv_path, output_folder):
    # Read CSV file
    df = pd.read_csv(csv_path)
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for idx, row in df.iterrows():
        npz_file = row["Filename"]
        npz_file = npz_file.replace('.jpg', '.npz')
        base_dir, base_name = os.path.split(npz_file)
        base_name = 'DATA-' + base_name
        npz_file = os.path.join(base_dir, base_name)

        # If npz_file is not absolute, join with npz_folder
        if not os.path.isabs(npz_file):
            npz_file = os.path.join(npz_folder, npz_file)
        if not os.path.exists(npz_file):
            print(f"Warning: {npz_file} does not exist, skipping.")
            continue

        # Load original npz data
        data = dict(np.load(npz_file, allow_pickle=True))

        # Add CSV fields (excluding npz_path)
        for col in df.columns:
            if col != 'Filename':
                data[col] = row[col]
        # Save to new npz in output_folder
        out_name = os.path.basename(npz_file)
        out_path = os.path.join(output_folder, out_name)

        # Add quality key if needed 
        if "Quality" not in data.keys():
            good = np.all([not ch['flagged'] for ch in data["info"]])
            if good:
                data["Quality"] = "Good"
            else:
                data["Quality"] = "Bad"

        np.savez_compressed(out_path, **data)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate CSV fields into NPZ files.")
    parser.add_argument("--npz_folder", required=True, help="Folder containing original .npz files")
    parser.add_argument("--csv_path", required=True, help="CSV file with npz_path and attributes")
    parser.add_argument("--output_folder", required=True, help="Output folder for new .npz files")
    args = parser.parse_args()
    integrate_csv_to_npz(args.npz_folder, args.csv_path, args.output_folder)