import os
import sys
from astropy.io import fits

# TODO make it just do hdul 0 in the proper place
def convert_hdul1_to_hdul0(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.fits'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with fits.open(input_path) as hdul:
                if len(hdul) < 2:
                    print(f"Skipping {filename}: less than 2 HDUs.")
                    continue

                # Move hdul[1] to hdul[0]
                new_hdul = fits.HDUList([hdul[1]])
                # Optionally, copy header from hdul[0] if needed:
                # new_hdul[0].header.extend(hdul[0].header, update=True)

                new_hdul.writeto(output_path, overwrite=True)
                print(f"Converted {filename}.")

if __name__ == "__main__":

    convert_hdul1_to_hdul0(r"data_test_set\aia\2017\94", r"data_test_set\aia\2017\94")