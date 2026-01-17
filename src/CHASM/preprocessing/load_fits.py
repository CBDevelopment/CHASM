import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import astropy_mpl_style, simple_norm, ImageNormalize, AsinhStretch
from sunpy.visualization.colormaps import color_tables as ct
from sunpy.map import Map
import argparse
import astropy.units as u
from datetime import datetime
import os

# Set the plot style
plt.style.use(astropy_mpl_style)

def find_closest_file(directory, target_datetime):
    # Parse the target datetime into a datetime object
    target_datetime = datetime.strptime(target_datetime, "%Y-%m-%dT%H%M%S")

    closest_file = None
    closest_time_diff = None

    # Iterate through files in the directory
    for file in os.listdir(directory):
        if file.endswith('.fits'):
            # Extract the timestamp part of the filename
            timestamp_str = file.split('.')[0]  # Get the part before '.fits'
            
            try:
                # Convert the timestamp into a datetime object
                file_datetime = datetime.strptime(timestamp_str, "%Y-%m-%dT%H%M%S")
                
                # Calculate the absolute time difference between the file's timestamp and the target
                time_diff = abs(file_datetime - target_datetime)

                # Update the closest file if this one is closer
                if closest_time_diff is None or time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_file = file

            except ValueError:
                # Handle any file names that don't match the expected format
                continue

    return closest_file

def main(date, time, wavelength):
    ROOT = "D:\WPI\MQP\CoronalHoles\src\experiments"

    # Load the FITS file
    # Find the file with the closest date and time in the directory
    fits_file = f"{ROOT}/{wavelength}/{find_closest_file(f'{ROOT}/{wavelength}', f'{date}{time}')}"
    print(f"Loading FITS file: {fits_file}")
    # fits_file = f"{ROOT}/{wavelength}/mag.fits"
    # s_map = Map(fits_file)
    # print(s_map.instrument)

    with fits.open(fits_file) as hdul:
        hdul.info()  # Display information about the FITS file
        # hdul[1] for compressed images
        header = hdul[1].header
        image_data = hdul[1].data

    # Check the shape of the data
    print(f"Data shape: {image_data.shape}")

    # Use SunPy's Map to handle proper orientation and metadata
    solar_map = Map(image_data, header)

    fig = plt.figure()
    # norm = ImageNormalize(vmin=30, vmax=5000, stretch=AsinhStretch())
    # plt.imshow(image_data, cmap=ct.aia_color_table(int(wavelength) * u.angstrom), origin='lower', norm=norm)
    # plt.imshow(image_data, cmap=ct.hmi_mag_color_table(), origin='lower')
    # plt.grid(False)
    ax = plt.subplot(projection=solar_map)
    try:
        im = solar_map.plot(axes=ax, cmap=ct.aia_color_table(int(wavelength) * u.angstrom))
    except:
        im = solar_map.plot(axes=ax, cmap=ct.hmi_mag_color_table())
    plt.colorbar(im, ax=ax, label='Intensity (DN/s)')
    plt.title(f'Solar Image at {wavelength} Å')
    observation_date = solar_map.date.strftime('%Y-%m-%d %H:%M:%S')
    plt.figtext(0.5, 0.01, f'Observation Date: {observation_date}', ha='center', fontsize=10)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--date', type=str, help='Date of the solar image')
    parser.add_argument('--time', type=str, help='Time of the solar image', default='T000000')
    parser.add_argument('--wavelength', type=str, help='Wavelength of the solar image', default='193')
    args = parser.parse_args()
    main(args.date, args.time, args.wavelength)

