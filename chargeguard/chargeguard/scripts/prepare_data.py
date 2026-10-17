"""Download the Shenzhen EV charging dataset and convert it to data/ev30.csv.gz.

    python -m chargeguard.scripts.prepare_data

Source: Yang, Tian, Mai, Chung (Hong Kong Polytechnic University) with Shenzhen Auto Electric
Power Plant Co. (Autosun), Mendeley Data doi:10.17632/c7gg94tmvz.3, CC BY 4.0. The file used is
processed_data_longer_than_30.xlsx (charging records longer than 30 minutes), 8 sheets.
"""
from __future__ import annotations

import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
URL = ("https://data.mendeley.com/public-files/datasets/c7gg94tmvz/files/"
       "0bd6856e-41e5-45a4-ab75-e9f1273dff85/file_downloaded")


def download(dest: Path, tries: int = 4) -> None:
    for i in range(tries):
        urllib.request.urlretrieve(URL, dest)
        if zipfile.is_zipfile(dest):  # Mendeley sometimes answers with a JSON error page
            return
        time.sleep(10)
    raise RuntimeError("Mendeley did not return the spreadsheet; try again later")


def main() -> None:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    xlsx, out = data / "ev30.xlsx", data / "ev30.csv.gz"
    if out.exists():
        print(f"{out} already exists")
        return
    if not xlsx.exists():
        print("downloading (103 MB)...")
        download(xlsx)
    t = time.time()
    sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"{len(df):,} readings from {len(sheets)} sheets -> {out} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
