# Emirates ID Extractor

Reads the **English name** and **Emirates ID number** from Emirates ID card fronts, saves the printed portrait and the full card as JPEGs, and writes everything to an Excel workbook.

OCR and face detection run locally. You don't need Tesseract, an API key, or an internet connection after installing.

## Quick start

Requires Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/).

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe extract_emirates_ids.py
```

To use other folders:

```powershell
.venv\Scripts\python.exe extract_emirates_ids.py --input C:\cards --output C:\results
```

## Input

Put files directly in `input/`. Subfolders are not scanned, and the input files are never modified.

- **Supported formats:** `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, `webp`, `pdf`
- **One card front per image or PDF page.** A PDF can have many pages; each page is treated as a separate card.
- **Rotated scans are fine.** Each card is tried at 0°, 90°, 180° and 270°.

## Output

`output/` is created automatically:

```text
output/
  emirates_ids.xlsx   # One row per card
  photos/             # Portrait crops, e.g. "Ahmad Rashid Mohammed.jpg"
  id_cards/           # Full, upright card images, same naming
```

| Column | Contents |
| --- | --- |
| Source file, Page | Where the card came from |
| English name | Title-cased name |
| Emirates ID number | Formatted as `784-XXXX-XXXXXXX-X` |
| Photo, ID image | Paths to the saved JPEGs |
| Status | `ok`, `review`, or `error` |
| Details | Why a row needs review, e.g. `Low confidence name`, `Portrait not found` |

Rows are marked `review` when a field is missing or its OCR confidence is below 0.85. Missing values are left blank.

### How results are saved

- Files are named after the English name. Duplicate names get `_2`, `_3`, and so on.
- **Images:** every file gets a row. A portrait is saved only if a name was found. A full card image is saved only if both the name and the ID number were found.
- **PDF pages:** a page is kept only if both the name and the ID number are found. Other pages, such as cover sheets or card backs, are skipped and reported in the console.
- **Re-running:** the workbook is replaced and JPEGs with the same names are overwritten. Files from earlier runs that aren't regenerated are left in place.

## Project files

| Path | Purpose |
| --- | --- |
| `extract_emirates_ids.py` | The extractor |
| `data/haarcascades/` | OpenCV face model used to find the portrait (required) |
| `input/` | Sample card as JPG and PDF |

## License

See [LICENSE](LICENSE).
