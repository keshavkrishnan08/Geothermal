# Data availability

Every input is public. This file lists source, version, retrieval date, licence,
and a brief description for each dataset used by GeoProspectNet.

## Geophysics

### Gravity & magnetic anomaly grids
- **Source:** USGS Gravity and Magnetic Atlas of North America (GMNA), v2002
- **DOI / URL:** https://mrdata.usgs.gov/magnetic/show-magnetic.php
- **Resolution:** 2 km regional, resampled to 4 km in our grid
- **Licence:** US Public Domain
- **Pipeline:** `src/data/ingest_geophysics.py::download_gmna`

### Surface heat flow
- **Source:** SMU Geothermal Lab Heat Flow Database, v2023
- **URL:** http://geothermal.smu.edu/gtda/
- **Licence:** Free for academic use; commercial use requires permission
- **Pipeline:** `src/data/ingest_geophysics.py::sample_heat_flow`

### Elevation (SRTM)
- **Source:** NASA Shuttle Radar Topography Mission, 90 m v3
- **URL:** https://lpdaac.usgs.gov/products/srtmgl3v003/
- **Licence:** No restrictions
- **Pipeline:** `src/data/ingest_geophysics.py::sample_srtm`

## Geochemistry

### Spring chemistry + geothermometers
- **Source:** USGS GEOTHERM database, continuous updates (we use the 2024 snapshot)
- **URL:** https://www.sciencebase.gov/catalog/item/580ad0a4e4b0a48de6df9d65
- **Licence:** US Public Domain
- **Features used:** spring temperature, chalcedony geothermometer (T_chalc), Na-K geothermometer (T_NaK), Mg-K geothermometer (T_MgK), pH, conductivity, dissolved silica, sulfate, chloride
- **Pipeline:** `src/data/ingest_geochemistry.py`

## Geology

### Lithology — SGMC
- **Source:** USGS State Geologic Map Compilation, Data Series 1052, v6.5 (2017)
- **DOI:** 10.3133/ds1052
- **Licence:** US Public Domain
- **Features used:** unit lithology one-hot encodings collapsed into nine categories
  (volcanic, intrusive, sedimentary, metamorphic, unconsolidated, alluvial, lacustrine, evaporite, other)
- **Pipeline:** `src/data/ingest_geology.py`

### Faults
- **Source:** Mordensky et al. (2023) ScienceBase release
- **URL:** https://www.sciencebase.gov/catalog/item/63c3aaadd34ec1b9ad6c6f30
- **Licence:** US Public Domain
- **Features used:** fault density (km/km²) and mean slip rate (mm/yr) per 4 km cell
- **Pipeline:** `src/data/ingest_geology.py::add_mordensky_faults`

## Labels

### Known positives — USGS Williams (2008)
- **Source:** *Assessment of moderate- and high-temperature geothermal resources of the United States*
- **DOI:** 10.3133/fs20083082
- **N:** 345 entries → 280 unique fields after deduplication → 1 370 grid cells within 5 km of any field
- **Pipeline:** `src/data/build_labels.py::_load_williams_2008`

### Temporal hold-out — Nevada post-2008 permits
- **Source:** Nevada Division of Minerals geothermal permit database (search dates 2008–2025)
- **URL:** http://minerals.nv.gov/Programs/Geothermal/
- **N:** 22 cells corresponding to 19 unique post-2008 permits including:
  - Ormat Dixie Valley LLC 52-15 (permit 1668)
  - Ormat Nevada, Inc 24A-23 (permit 1670), 82A-27 (permit 1673), 21-31 (permit 1590)
  - Zanskar Geothermal & Minerals ROS-11-27, ROS-28-15 (permit 1678), ROS-68-27 (permit 1679)
  - Opus 7 Geothermal AGR 1 TG (permit 1701)
  - Nevada Bureau of Mines and Geology 15-16 (permit 1615), 10-47 (permit 1616), 3-77 (permit 1618),
    1-88 (permit 1619), 34-85 (permit 1621), 11-28 (permit 1624), 3-51 (permit 1625), 10-54 (permit 1626)
  - Additional non-operator-attributed permits 1602, 1607, 1641, 1690, 1569
- **Pipeline:** `data/raw/labels/nv_permits_holdout.csv` (committed)
- **Note:** Coordinates of the Zanskar Big Blind permits are public via the Nevada Division of Minerals permit pages.

## Negative-control cohorts

- **NC1 Colorado Plateau interior:** grid cells in 36–40°N, −111° to −108°W that are > 200 km from any USGS field
- **NC3 random deep:** grid cells > 100 km from any USGS field AND lacking geochemistry coverage (proxy for no nearby springs)
- **Random western US:** uniform random sample of 100 cells across the study region

All cohorts are reconstructed deterministically by `src/evaluation/negative_controls.py` (seed = 42).

## Processed arrays

The 7 GB of processed-array files (`data/processed/*.npy`) live in this repo at:
- `geophysics_patches.npy`  shape (N, 6, 32, 32)
- `geochemistry_features.npy`  shape (N, 9)
- `geology_features.npy`  shape (N, 11)
- `modality_masks.npy`  shape (N, 3)  — per-modality coverage masks
- `labels.npy`  shape (N,)  — 0/1
- `train_mask.npy`  shape (N,)  — bool
- `grid_coordinates.csv`  N rows of (cell_id, row, col, lat, lon, province)

For external mirroring (Zenodo DOI to be issued upon publication), bundle the entire `data/processed/` directory.
