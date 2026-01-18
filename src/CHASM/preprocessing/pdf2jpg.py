import pypdfium2 as pdfium
from pathlib import Path
from time import perf_counter
from tqdm import tqdm

DRAWINGS_PATH = Path("D:/WPI/MQP/CoronalHoles/src/paper/drawings")
JPG_PATH = Path("D:/WPI/MQP/CoronalHoles/src/paper/drawings/jpg")

drawings = [f for f in DRAWINGS_PATH.iterdir() if f.suffix == ".pdf"]

JPG_PATH.mkdir(parents=True, exist_ok=True)

start = perf_counter()
for drawing in tqdm(drawings, desc="Converting PDFs to JPGs"):
    jpg_path = JPG_PATH / f"{drawing.stem}.jpg"
    if jpg_path.exists():
        continue
    pdf = pdfium.PdfDocument(str(drawing))
    for i in range(len(pdf)):
        page = pdf[i]
        image = page.render(scale=4).to_pil()
        image.save(str(jpg_path))
    pdf.close()
end = perf_counter()

print(f"Time taken: {end - start:.2f} seconds")
print("Number of PDFs converted: ", len(drawings))
print(f"Time per PDF: {(end - start) / len(drawings):.2f} seconds")
