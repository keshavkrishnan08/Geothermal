# GeoProspectNet

**Continental-scale geothermal discovery via multi-modal contrastive deep learning across the western United States.**

This repository accompanies the manuscript *"Continental-scale geothermal discovery via multi-modal contrastive deep learning identifies 4.3 GW of hidden hydrothermal resource in the western United States."*

## Headline numbers

- **235 470 cells** scored at 4 km resolution across the western US
- **1 370 known positives** from the USGS Williams 2008 inventory
- **22 post-2008 blind-test sites** — 100 % land in the top 10 % of the continental grid (mean 96ᵗʰ percentile), including all three Zanskar Big Blind permits
- **33 consensus discoveries** robust to seed and model variant
- **6 607 MWe** cumulative recoverable resource (Monte Carlo P50; range 2.5 – 17.1 GW at P10–P90), **1.74× current US installed geothermal capacity**
- Tree-based ML baselines achieve 1.00 AUROC on the test split but **0 %** capture on the temporal hold-out — illustrating why temporal validation is essential

## Quick reproduce

```bash
git clone git@github.com:keshavkrishnan08/Geothermal.git Geothermal2
cd Geothermal2
conda env create -f environment.yml && conda activate geoprospectnet

# These four files are too large for GitHub and were excluded — regenerate them via:
# 1. data/processed/geophysics_patches.npy (5.4 GB) :
python -m src.data.ingest_geophysics --config configs/cpu_max.yaml
# 2. data/raw/geology/sgmc/USGS_SGMC_Geodatabase (740 MB) — pull via:
python -m src.data.ingest_geology --config configs/cpu_max.yaml
# 3. data/raw/geology/qfaults/SHP/Qfaults_US_Database.dbf (194 MB) :
#    auto-downloaded by ingest_geology when missing
# 4. data/raw/labels/mordensky2023_full.csv (135 MB) — fetch with:
curl -L 'https://www.sciencebase.gov/catalog/file/get/63c3aaadd34ec1b9ad6c6f30' -o data/raw/labels/mordensky2023_full.csv

# Then run the full pipeline:
jupyter lab notebooks/kaggle_full_pipeline.ipynb     # or nature_energy_pipeline.ipynb
```

For Kaggle: `notebooks/kaggle_full_pipeline.ipynb` clones this repo automatically and runs every experiment in one 9-hour GPU session.

The notebook runs end-to-end on CPU in ~3 hours. Pre-computed checkpoints, continental scores, discovery tables and figures are committed to `outputs/`; the notebook short-circuits any step you don't want to re-run.

## Repository layout

```
configs/                Training configs (cpu_calibrated → cpu_max)
data/
├── raw/                Original public downloads (gravity, magnetic, heat flow, ...)
├── metadata/           Curated metadata tables (known fields, splits)
└── processed/          Per-cell arrays consumed by training
src/
├── data/               Data ingestion + label / split construction
├── models/             GeoProspectNet architecture
├── training/           Training loops, losses, separation sweep
├── evaluation/         Negative controls, discovery, baselines, calibration, MWe
└── visualization/      Figure generation
outputs/
├── checkpoints/        Trained models (one per config + seed)
├── results/            Tables: separation sweep, baselines, consensus + MWe, ...
├── maps/               Continental GeoTIFFs (prospectivity + uncertainty)
└── figures/            Publication-quality PNG + PDF
notebooks/
└── nature_energy_pipeline.ipynb   End-to-end reproduction
manuscript/
└── manuscript.md       Working draft (Markdown for now; LaTeX export pending)
```

## Key results

### Cohort validation (cpu_max variant, continental percentile rank)
| Cohort | n | Mean percentile | In top 10 % |
|---|---:|---:|---:|
| Known USGS positives | 1 370 | 88.5 | 50 % |
| Post-2008 hold-out | 22 | **95.7** | **100 %** |
| Random western US | 100 | 53.3 | 10 % |
| NC1 Colorado Plateau | 100 | 43.2 | 0 % |
| NC3 random deep | 100 | **5.3** | 0 % |

### Baselines vs GeoProspectNet
| Method | AUROC | Hold-out mean pct | Hold-out top-10 % capture |
|---|---:|---:|---:|
| Heat-flow rule | 0.86 | 93.1 | 82 % |
| Logistic regression | 1.00 | 95.3 | 91 % |
| Random Forest | 1.00 | 59.4 | **0 %** |
| XGBoost | 1.00 | 61.1 | **0 %** |
| **GeoProspectNet** | **1.00** | **98.2** | **100 %** |

See `outputs/results/baselines.csv` for full metrics.

## Data sources

See `DATA.md` for source, version, retrieval date, and licence for every dataset.

## Citation

```
@article{krishnan2026geoprospectnet,
  author  = {Krishnan, Keshav},
  title   = {Continental-scale geothermal discovery via multi-modal contrastive deep learning identifies 4.3 GW of hidden hydrothermal resource in the western United States},
  year    = {2026},
  journal = {<submitted>},
}
```

## Licence

Code released under MIT. Data are under their original licences — see `DATA.md`.

## Contact

keshavkrishnanbusiness@gmail.com
