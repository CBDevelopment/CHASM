import requests
from datetime import datetime, timedelta
import bs4 as bs
import re
from pathlib import Path

# 2021 only has December 2021, pulling 2022 instead


class SWPC:
    DRAWING_URL = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-imagery/composites/full-sun-drawings/boulder/"

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_synoptic_map(date: str, time: str, save_path: str):
        year, month, day = date.split("-")
        hours, minutes, _sec = time.split(":")

        url = f"{SWPC.DRAWING_URL}/{year}/{month}/"
        response = requests.get(url)

        if response.status_code == 200:
            soup = bs.BeautifulSoup(response.content, "html.parser")

            image_name_regex = re.compile(r"boul_neutl_fd_\d{8}_\d{4}\.(jpg|pdf)")
            image_links = soup.find_all("a", href=image_name_regex)

            image_candidates = []
            for image_link in image_links:
                if f"{year}{month}{day}" in image_link["href"]:
                    image_candidates.append(image_link)

            if len(image_candidates) == 0:
                print(f"No synoptic map found for {date}")
                return None
            else:
                print(image_candidates)

                best_candidate = image_candidates[0]
                best_candidate_time = int(
                    best_candidate["href"].split("_")[-1].split(".")[0]
                )
                for candidate in image_candidates:
                    candidate_time = int(candidate["href"].split("_")[-1].split(".")[0])
                    if abs(candidate_time - int(f"{hours}{minutes}")) < abs(
                        best_candidate_time - int(f"{hours}{minutes}")
                    ):
                        best_candidate = candidate
                        best_candidate_time = candidate_time

                image_url = url + best_candidate["href"]
                print(f"Downloading image from {image_url}")

                image_response = requests.get(image_url)

                # Save the image locally
                if image_response.status_code == 200:
                    save_file = Path(save_path) / best_candidate["href"]
                    with open(save_file, "wb") as file:
                        file.write(image_response.content)
                        print(f"Image saved as {best_candidate['href']}")
                        return best_candidate["href"]
                else:
                    print(f"Failed to download the image from {image_url}")
                    return None
        else:
            return None


def load_SWPC_files(start_date, end_date, save_path):
    curr_date = start_date
    while curr_date <= end_date:
        date_str = datetime.strftime(curr_date, "%Y-%m-%d")
        SWPC.get_synoptic_map(date_str, "00:00:00", save_path)  # Get day map
        curr_date += timedelta(days=1)


if __name__ == "__main__":
    # Gets data from 2018 to 2024
    start_date = datetime(year=2022, month=1, day=1)
    end_date = datetime(year=2022, month=12, day=31)
    save_path = r"data\dataset\2022drawings"
    load_SWPC_files(start_date, end_date, save_path)
