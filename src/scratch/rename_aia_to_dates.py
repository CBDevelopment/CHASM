import os
import re

def extract_date(filename):
    # Try to match 'YYYYMMDD' or 'YYYY-MM-DD'
    match1 = re.search(r'(\d{8})', filename)
    match2 = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match2:
        date = match2.group(1)
    elif match1:
        date_raw = match1.group(1)
        date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
    else:
        return None
    # Get extension
    ext = os.path.splitext(filename)[1]
    return f"{date}{ext}"

def rename_files_in_folder(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            new_name = extract_date(fname)
            if new_name and fname != new_name:
                src = os.path.join(dirpath, fname)
                dst = os.path.join(dirpath, new_name)
                print(f"Renaming: {src} -> {dst}")
                os.rename(src, dst)

if __name__ == "__main__":
    folder = r"data\aia\2019"
    rename_files_in_folder(folder)