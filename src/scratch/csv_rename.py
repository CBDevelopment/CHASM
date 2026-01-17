import pandas as pd
import re
from datetime import datetime

# Rename DATA-boul_neutl_fd_20171223_0745.npz to 2017-12-23T07-45_CHASMSelection.npz
def convert_filename(old_name):
    # Example: DATA-boul_neutl_fd_20171223_0745.npz
    match = re.match(r'boul_neutl_fd_(\d{8})_(\d{4})\.jpg', old_name)
    if not match:
        return None
    date_str, time_str = match.groups()
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    new_name = dt.strftime("%Y-%m-%dT%H-%M_CHASMSelection.npz")
    print(new_name)
    return new_name

# def convert_filename(old_name):
#     # Example: boul_neutl_fd_20170101_0330.jpg
#     match = re.match(r'.*_(\d{8})_(\d{4})\.jpg', old_name)
#     if not match:
#         return None
#     date_str, time_str = match.groups()
#     # Convert date and time
#     dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
#     # Add 2 hours 25 minutes as in your example (03:30 -> 05:55)
#     dt = dt.replace(hour=dt.hour + 2, minute=dt.minute + 25)
#     # Format as 2017-01-03T05-55_CHASMSelection.npz
#     new_name = dt.strftime("%Y-%m-%dT%H-%M_CHASMSelection.npz")
#     return new_name

# Load your CSV (replace 'your_file.csv' with your actual file)
df = pd.read_csv(r'download_data\chasm_renamed\2024\detection_fractions_old.csv')

# Apply conversion
df['Filename'] = df['Filename'].apply(convert_filename)

# Save to new CSV
df.to_csv(r'download_data\chasm_renamed\2024\detection_fractions.csv', index=False)