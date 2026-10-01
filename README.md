# Emirates ID extractor

Reads the **English name** and Emirates ID number from card fronts, saves each printed portrait as a JPEG, and creates an Excel workbook.

## Input folder

Place Emirates ID files directly in `input/`. The extractor processes supported files in that folder: `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, `webp`, and `pdf`. Put one card front in each image or PDF page. A PDF can contain multiple pages; each page gets its own workbook row. The input files are not changed.

## Output folder

The extractor creates `output/` automatically:

```text
output/
  emirates_ids.xlsx  # Source file, page, English name, ID number, photo path, status, details
  photos/            # Portrait JPEGs named after the English name
```

Repeated names get `_2`, `_3`, etc. The `Status` and `Details` columns flag fields or photos that need review; missing values are left blank. Running the extractor again updates photos with the same generated names and replaces the workbook.

## Install

Python 3.11 or 3.12 is recommended. No Tesseract installation or API key is required; the OCR model runs locally. With `uv` installed:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python.exe extract_emirates_ids.py
```

To choose different folders:

```powershell
.venv\Scripts\python.exe extract_emirates_ids.py --input C:\cards --output C:\results
```
