# Running GeoProspectNet on Kaggle

**One notebook does everything**: `notebooks/geoprospectnet_mega.ipynb`. Cells 2 (data download) and 3 (data processing) are fully automated — no manual file uploads needed.

## Setup (once)

1. Push this repo to GitHub. Replace `YOUR_GITHUB` in cell 2 of the notebook with your username.

   ```bash
   cd /Users/keshavkrishnan/Geothermal2
   git init && git add . && git commit -m "Initial commit"
   git remote add origin git@github.com:YOUR_USERNAME/Geothermal2.git
   git push -u origin main
   ```

   *Or* upload the repo as a Kaggle Dataset called `geoprospectnet-source` — the notebook falls back to it.

2. New Kaggle notebook → **Settings → Accelerator: GPU T4 x2 → Internet: ON**.
3. **File → Upload notebook** → `notebooks/geoprospectnet_mega.ipynb`.
4. Run all cells.

## Wall-clock budget on a single T4

| Section | Time |
|---|---|
| 1. Setup | 2 min |
| 2. Data download (~500 MB pulled, ~700 MB temp) | 10–20 min |
| 3. Data processing | 30–60 min |
| 4. EDA | 1 min |
| 5. Smoke test | 5–10 min |
| 6. Train discovery model | 1–2 hr |
| 7. **HEADLINE — discovery + MWe + Zanskar** | 15–30 min |
| 8. Quick benchmarking (5 folds + baselines) | 30–60 min |
| 9. Ablation (5 variants × 5 folds) | 2–3 hr |
| 10. Cross-basin transfer | 30 min |
| 11. Modality + spatial analysis | 30 min |
| 12. Figures + tables | 5 min |
| 13. Statistical tests | 1 min |
| 14. Archive | 30 sec |

**Total ~6–9 hr** end-to-end. Well within Kaggle's 12-hour session limit.

If you only have time for the headline:
- Run cells 1–7, then 14. Stop. Total ~3 hr.
- That's enough for a Geothermics submission with a defensible discovery + Zanskar validation.

## What lands in `outputs/`

- `outputs/maps/prospectivity.tif` — continental p(geothermal) GeoTIFF for QGIS / ArcGIS
- `outputs/maps/uncertainty.tif` — MC-Dropout standard-deviation map
- `outputs/results/discoveries_with_mwe.csv` — top candidate sites with MWe estimates and plausibility flag
- `outputs/results/blind_validation.csv` — Zanskar Big Blind percentile check
- `outputs/results/table1_main_results.csv` — LOF-CV vs all 7 baselines, includes Mordensky-2025 capture rates at top {1, 5, 10, 25}%
- `outputs/results/table2_ablation.csv` — ablation
- `outputs/results/cross_basin_transfer.json` — Basin & Range → Cascades transfer
- `outputs/results/permutation_importance.csv`, `attention_by_province.csv`, `morans_i.json`
- `outputs/figures/figure[2-7]_*.{png,pdf}` — 6 publication figures
- `outputs/checkpoints/*.pt` — model weights for re-running discovery without retraining

The final cell tars everything into `/kaggle/working/geoprospect_outputs.tar.gz` so it survives Kaggle's session expiry.

## Troubleshooting

- **`OSError: cannot find rasterio` after pip install** — restart the Kaggle kernel after cell 1.
- **`KeyError: 'province'`** — re-run cell 11; `02_process_data.sh` writes the province metadata.
- **Discovery returns 0 candidates** — lower `discovery.prob_threshold` from 0.7 to 0.6 in `configs/default.yaml` and re-run cell 7's first command.
- **GPU OOM** — drop `training.batch_size` from 256 to 128 in `configs/default.yaml`.
- **Session times out before cell 14** — re-launch, skip cells 2–6 (data and checkpoint persist on the working volume), continue from cell 7.
- **EIA-860 archive 404** — change `eia8602023.zip` to `eia8602024.zip` in cell 2; structure is identical.
- **GDR or Smithsonian temporarily down** — re-run cell 2; it skips files that already exist.
