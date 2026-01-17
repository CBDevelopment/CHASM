
import pypdfium2 as pdfium
import os
from time import perf_counter
from tqdm import tqdm

DRAWINGS_PATH = "D:/WPI/MQP/CoronalHoles/src/paper/drawings"
JPG_PATH = "D:/WPI/MQP/CoronalHoles/src/paper/drawings/jpg"

drawings = os.listdir(DRAWINGS_PATH)
# Remove any non-PDF files
drawings = [drawing for drawing in drawings if drawing.endswith(".pdf")]

if not os.path.exists(JPG_PATH):
    os.mkdir(JPG_PATH)

start = perf_counter()
for drawing in tqdm(drawings, desc="Converting PDFs to JPGs"):
    if os.path.exists(f"{JPG_PATH}/{drawing.split('.')[0]}.jpg"):
        continue
    pdf = pdfium.PdfDocument(f"{DRAWINGS_PATH}/{drawing}")
    for i in range(len(pdf)):
        page = pdf[i]
        image = page.render(scale=4).to_pil()
        image.save(f"{JPG_PATH}/{drawing.split('.')[0]}.jpg")
    pdf.close()
end = perf_counter()

print(f"Time taken: {end-start:.2f} seconds")
print("Number of PDFs converted: ", len(drawings))
print(f'Time per PDF: {(end-start)/len(drawings):.2f} seconds')