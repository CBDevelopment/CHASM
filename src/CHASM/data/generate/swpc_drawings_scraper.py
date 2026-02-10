import requests
from datetime import datetime, timedelta
import bs4 as bs
import re
from pathlib import Path
import pypdfium2 as pdfium
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Repository of all SWPC synoptic drawings
_DRAWING_URL = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-imagery/composites/full-sun-drawings/boulder/"


class SWPCDrawingsScraper:
    def __init__(
        self,
        save_path: Path,
        drawing_url: str = _DRAWING_URL,
        convert_pdf_to_jpg: bool = True,
    ):
        self.drawing_url = drawing_url
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.convert_pdf_to_jpg = convert_pdf_to_jpg

    def _fetch_directory_listing(self, url: str) -> requests.Response | None:
        """Fetch the directory listing from the given URL."""
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            print(f"Failed to access URL: {url}")
            return None

    def _parse_image_links(self, html_content: bytes, date_str: str) -> list:
        """Parse HTML content and return image links matching the date."""
        soup = bs.BeautifulSoup(html_content, "html.parser")
        image_name_regex = re.compile(r"boul_neutl_fd_\d{8}_\d{4}\.(jpg|pdf)")
        image_links = soup.find_all("a", href=image_name_regex)

        image_candidate_links = []
        for image_link in image_links:
            if date_str in image_link["href"]:
                image_candidate_links.append(image_link)

        return image_candidate_links

    def _find_best_candidate(
        self, candidate_links: list, target_time: str
    ) -> dict | None:
        """Find the candidate with the closest time to the target time."""
        if not candidate_links:
            return None

        best_candidate_link = candidate_links[0]
        best_candidate_time = int(
            best_candidate_link["href"].split("_")[-1].split(".")[0]
        )

        for candidate in candidate_links:
            candidate_time = int(candidate["href"].split("_")[-1].split(".")[0])
            if abs(candidate_time - int(target_time)) < abs(
                best_candidate_time - int(target_time)
            ):
                best_candidate_link = candidate
                best_candidate_time = candidate_time

        return best_candidate_link

    def _download_and_save_image(self, image_url: str, filename: str) -> Path | None:
        """Download an image from the URL and save it to disk."""
        logger.info(f"Downloading image from {image_url}")

        image_response = requests.get(image_url)

        if image_response.status_code == 200:
            save_file = self.save_path / filename
            with open(save_file, "wb") as f:
                f.write(image_response.content)
            logger.info(f"Saved synoptic map to {save_file}")
            if self.convert_pdf_to_jpg and save_file.suffix.lower() == ".pdf":
                pdf = pdfium.PdfDocument(str(save_file))
                page = pdf.get_page(0)
                pil_image = page.render(scale=4).to_pil()
                jpg_save_file = save_file.with_suffix(".jpg")
                pil_image.save(jpg_save_file, "JPEG")
                logger.info(f"Converted PDF to JPG: {jpg_save_file}")
                save_file.unlink()  # Remove the original PDF file
                return jpg_save_file
            return save_file
        else:
            logger.warning(f"Failed to download image from {image_url}")
            return None

    def get_drawing_for_date(
        self, date: str = "2026-01-01", time: str = "00:00"
    ) -> Path | None:
        """Get the synoptic drawing closest to the given date and time."""
        year, month, day = date.split("-")
        hours, minutes = time.split(":")

        url = f"{self.drawing_url}/{year}/{month}/"
        response = self._fetch_directory_listing(url)

        if not response:
            return None

        date_str = f"{year}{month}{day}"
        image_candidates = self._parse_image_links(response.content, date_str)

        if len(image_candidates) == 0:
            logger.warning(f"No synoptic map found for {date}")
            return None

        target_time = f"{hours}{minutes}"
        best_candidate = self._find_best_candidate(image_candidates, target_time)

        if not best_candidate:
            return None

        filename = best_candidate["href"]
        if self.convert_pdf_to_jpg and filename.endswith(".pdf"):
            final_filename = filename.replace(".pdf", ".jpg")
        else:
            final_filename = filename
        final_path = self.save_path / final_filename

        if final_path.exists():
            logger.info(f"File already exists: {final_path}")
            return final_path

        image_url = url + filename
        return self._download_and_save_image(image_url, filename)

    def get_drawings_for_date_range(
        self, start_date: str, end_date: str, time: str = "00:00"
    ) -> list[Path]:
        """Get synoptic drawings for a range of dates."""
        logger.info(f"Downloading drawings from {start_date} to {end_date}...")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        total_days = (end_dt - start_dt).days + 1
        downloaded_files = []

        for i in tqdm(range(total_days), desc="Downloading drawings for date range"):
            current_dt = start_dt + timedelta(days=i)
            date_str = current_dt.strftime("%Y-%m-%d")
            downloaded_file = self.get_drawing_for_date(date_str, time)
            if downloaded_file:
                downloaded_files.append(downloaded_file)

        return downloaded_files

    def get_drawings_for_dates(
        self, date_list: list[str], time: str = "00:00"
    ) -> list[Path]:
        """Get synoptic drawings for a list of specific dates or datetimes."""
        logger.info("Downloading drawings for specified dates...")
        downloaded_files = []

        for date_str in tqdm(date_list, desc="Downloading drawings for dates"):
            try:
                # Try to parse as datetime (YYYY-MM-DDTHH:MM)
                dt = datetime.fromisoformat(date_str)
                date = dt.strftime("%Y-%m-%d")
                time_to_use = dt.strftime("%H:%M")
            except ValueError:
                # If not a datetime, assume it's a date string and use provided time
                date = date_str
                time_to_use = time

            downloaded_file = self.get_drawing_for_date(date, time_to_use)
            if downloaded_file:
                downloaded_files.append(downloaded_file)

        return downloaded_files
