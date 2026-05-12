"""Download all public datasets used by GeoProspectNet.

All datasets are free and public. URLs and fallbacks come from CLAUDE.md Section 1.
The script is resumable — files that already exist are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DATASETS = {
    "labels/operating_plants.zip": {
        "url": "https://www.nrel.gov/gis/assets/data/geothermal-operating-plants.zip",
        "unzip_to": "labels/operating_plants",
        "description": "NREL operating geothermal power plants",
    },
    "labels/developing_plants.zip": {
        "url": "https://www.nrel.gov/gis/assets/data/geothermal-developing-plants.zip",
        "unzip_to": "labels/developing_plants",
        "description": "NREL developing geothermal plants",
    },
    "geophysics/bouguer_gravity.tif": {
        "url": "https://mrdata.usgs.gov/gravity/bouguer/USgrv_bou_SDD_geog.tif",
        "description": "USGS Bouguer gravity anomaly",
    },
    "geophysics/isostatic_gravity.tif": {
        "url": "https://mrdata.usgs.gov/gravity/isostatic/USgrv_iso_SDD_geog.tif",
        "description": "USGS isostatic residual gravity anomaly",
    },
    "geology/qfaults.zip": {
        "url": "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip",
        "unzip_to": "geology/qfaults",
        "description": "USGS Quaternary fault and fold database",
    },
}

MANUAL_INSTRUCTIONS = """
The following datasets require manual / GEE / login-based download.
Place the resulting files at the indicated paths under data/raw/.

1. USGS Identified Hydrothermal Systems (2008 Assessment)
   - Page: https://gdr.openei.org/submissions/194
   - Expected: data/raw/labels/usgs_identified_systems.csv

2. SMU Heat Flow Database
   - Page: https://gdr.openei.org/submissions/1704
   - Expected: data/raw/geophysics/smu_heatflow/*.csv

3. USGS / NOAA Magnetic Anomaly
   - Page: https://mrdata.usgs.gov/magnetic/map-us.html
   - NOAA fallback EMAG2v3: https://www.ngdc.noaa.gov/geomag/emag2.html
   - Expected: data/raw/geophysics/magnetic_anomaly.tif

4. USGS GEOTHERM Geochemistry
   - Page: https://gdr.openei.org/submissions/194
   - Expected: data/raw/geochemistry/geotherm.csv

5. NOAA Thermal Springs
   - Page: https://www.ngdc.noaa.gov/hazard/thermal.shtml
   - Expected: data/raw/geochemistry/thermal_springs.csv

6. Smithsonian Global Volcanism Program — Holocene Volcanoes
   - Page: https://volcano.si.edu/volcanolist_holocene.cfm  (click "Download List")
   - Expected: data/raw/geology/holocene_volcanoes.csv

7. SRTM Elevation (western US)
   - Run:  pip install elevation
            eio clip -o data/raw/geology/srtm_elevation.tif --bounds -125 31 -103 49
   - Or use the Python `elevation` package directly.

8. Landsat thermal anomaly (OPTIONAL — modality can be skipped)
   - Run the GEE script in CLAUDE.md Section 1.4.1
   - Expected: data/raw/satellite/thermal_anomaly.tif

Any missing file is handled gracefully at processing time: the affected modality
mask is set to False for cells that have no coverage.
"""


def _download(url: str, dst: Path, chunk: int = 1 << 16) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst.name} already exists")
        return True
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dst.name
            ) as bar:
                for buf in r.iter_content(chunk):
                    f.write(buf)
                    bar.update(len(buf))
            tmp.rename(dst)
        return True
    except Exception as e:
        print(f"[fail] {url}: {e}", file=sys.stderr)
        return False


def _unzip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst_dir)
    print(f"[unzip] {zip_path.name} -> {dst_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-manual", action="store_true",
                        help="Print manual-download instructions and exit")
    args = parser.parse_args()

    if args.print_manual:
        print(MANUAL_INSTRUCTIONS)
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for rel_path, meta in DATASETS.items():
        dst = RAW / rel_path
        ok = _download(meta["url"], dst)
        if ok and "unzip_to" in meta and dst.suffix == ".zip":
            _unzip(dst, RAW / meta["unzip_to"])
        n_ok += int(ok)

    print(f"\nDownloaded {n_ok}/{len(DATASETS)} auto-downloadable datasets.")
    print("\nNext: run with --print-manual for the remaining manual steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
