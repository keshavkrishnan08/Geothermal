# Positioning + experimental setup revisions

After the deep literature review (Mordensky 2025, Brown 2023, Smith 2024 PIGNN, SatCLIP/SeisCLIP, Zanskar Big Blind Dec 2025, GHFDB 2024), the project's experimental design and positioning need to shift. Paper text is unchanged for now — this doc captures what the experiments need to do and why.

## What the lit review changed

1. **Mordensky et al. 2025** is the killer benchmark. Same region (Great Basin), same year, USGS-imprimatur. Headline metric: 85% of operating power plants captured in the top 10% of predicted favorability; 15/28 in the top 1%. We have to report this metric or reviewers reject on benchmark-comparability grounds.

2. **Brown et al. 2023** ran ML on Nevada PFA features with Bayesian NN + PCA + k-means. Closest supervised-DL precursor. Must cite + benchmark.

3. **Smith et al. 2024 (PIGNN, Geothermal Energy)** is the deep-learning-on-CONUS competitor. Physics-informed graph network for thermal Earth model. Must cite as parallel deep approach.

4. **SatCLIP / GeoCLIP / SeisCLIP** (2023–2025) ALREADY use CLIP-style alignment in adjacent Earth-science settings. We cannot claim "first contrastive in Earth science." Defensible scope: "first geothermal-prospectivity application of cross-modal contrastive alignment."

5. **Zanskar Big Blind, December 2025**. First commercially confirmed AI-driven blind geothermal discovery in 30+ years (Nevada, 250°F at ~2,700 ft). If we ignore it, Nature Energy and Joule reject for being out-of-touch with the field.

6. **Global Heat Flow Database 2024 (Fuchs et al.)**. Strictly better than SMU. Must use it as primary source.

## Defensible novelty claim (revised)

> "First geothermal-prospectivity framework with cross-modal contrastive alignment, attention-based fusion, and explicit spatial-continuity priors. We benchmark directly against Mordensky et al.\ (2025) capture rates on the Great Basin and discuss our discoveries against the contemporaneous Zanskar commercial validation."

Drop everything broader.

## Required experimental additions

### E1. Mordensky 2025 head-to-head benchmark (PRIORITY)
- **Metric**: capture rate at top {1%, 5%, 10%, 25%} of predicted favorability across all 247K cells.
- **Status**: implemented in `src/evaluation/metrics.py::capture_rate_at_top_pct` and integrated into `train.py::metrics_at_threshold`. Reported automatically in `outputs/results/lof_cv_results.json` and aggregated into `table1_main_results.csv`.
- **Reporting**: target ≥85% capture at top 10% to claim parity-or-better with Mordensky 2025. Stretch: ≥90%.

### E2. GHFDB 2024 as primary heat-flow source (PRIORITY)
- **What changes**: replace SMU CSV input in `process_geophysics.py::_interp_heatflow` with GHFDB 2024 download (Fuchs et al., GFZ Potsdam DOI, ESSD 2025-341).
- **How**: add a `data/raw/geophysics/ghfdb2024/` directory; the `_interp_heatflow` function already reads any CSV with `lat`/`lon`/`hf`-like columns, so the dataset drop-in works. Treat SMU as legacy / cross-check.
- **Status**: not yet implemented. Just add a download instruction in `download.py` manual section.

### E3. Cross-basin transfer experiment (PRIORITY)
- **Why**: addresses a gap nobody else has filled — "does prospectivity learnt on Great Basin transfer to Cascades or Snake River Plain?"
- **How**: pretrain on Great Basin (lat 36–42, lon -120 to -114), evaluate top-percentile capture on Cascades + Snake River Plain held-out positives.
- **Status**: needs a new evaluation script `src/evaluation/cross_basin.py`. Not blocking the main results.

### E4. InSAR Sentinel-1 deformation as 5th modality (OPTIONAL — strong reach)
- **Why**: existing literature uses InSAR only for monitoring, not as a prospectivity input. Genuine novelty.
- **How**: ASF HyP3 service generates interferograms on demand. Compute median descending-track LOS velocity over 2016-2024 per cell. Feed to a 5th encoder (small CNN on a 32×32 deformation patch).
- **Status**: deferred. Significant data-pull effort. Mention as "future work" if not run; add as a 5th modality if Kaggle compute allows.

### E5. Engage the Zanskar Big Blind site (PRIORITY)
- **What**: the Zanskar discovery is in Nevada. If we exclude its known coordinates from training and our model independently ranks that location in the top 1%, that is the strongest possible blind-validation result. Direct comparison to a contemporaneous AI-driven commercial discovery.
- **How**: identify Zanskar Big Blind public coordinates (Nevada, near Black Rock Desert per public reports). Add to a `data/metadata/zanskar_validation.csv`. Add an evaluation script that loads predictions, finds the cell containing Zanskar coords, reports the predicted percentile.
- **Status**: easy add. Adds a striking sentence to the discussion.

### E6. Top-N MWe estimation per discovery (PRIORITY for Nature Energy reach)
- **Why**: turns "5 candidate sites" into "5 candidate sites with X ± Y MWe potential." Required for Nature Energy / Joule.
- **How**: USGS Williams 2008 volumetric method: `MW_e = (V * ρ * c_p * (T_res − T_ref) * RF * CE) / (lifetime * 86400 * 365 * conversion)`. Use geothermometer-derived T_res, assume reservoir thickness 250 m, recovery factor 0.10, conversion efficiency 0.10, lifetime 30 yr. Report mean ± 1σ across MC-Dropout passes.
- **Status**: not implemented. Add to `discovery.py` after the cluster aggregation step.

## Required positioning shifts

| Claim in current draft | Revised claim |
|---|---|
| "First multi-modal contrastive alignment for subsurface exploration" | "First geothermal-prospectivity application of cross-modal contrastive alignment, building on the SatCLIP/SeisCLIP lineage" |
| "First continental ML geothermal discovery system with spatial geological constraints" | "First geothermal prospectivity framework combining contrastive alignment, attention fusion, and explicit spatial priors; benchmarked against Mordensky 2025" |
| "Identifies previously unrecognised prospective regions" | "Identifies high-confidence candidate sites with quantified MWe potential; we benchmark our top-N predictions against the December 2025 Zanskar commercial discovery" |
| Implicit "novel deep learning" framing | Explicit "we use established contrastive alignment, applied for the first time to this domain, and quantify the gain over a strong supervised tree-ensemble baseline (Mordensky 2025)" |

## Venue strategy update

| Venue | Fit before lit review | Fit after experimental additions |
|---|---|---|
| Geothermics | 8/10 | 9/10 — add E1 + E2 + E5 |
| Energy Reports | 6.5/10 | 7.5/10 — add E1 + E2 |
| Joule | 6/10 | 8/10 — add E1 + E5 + E6 |
| Nature Energy | 4/10 | 7/10 — add E1 + E5 + E6 + ideally E4 |

## Implementation status

| ID | Task | Status | File |
|---|---|---|---|
| E1 | Mordensky 2025 capture-rate metric | **done** | `src/evaluation/metrics.py::capture_rate_at_top_pct`, integrated in `train.py::metrics_at_threshold` and `cross_validation.py` |
| E2 | GHFDB 2024 heat-flow source | doc only | drop GHFDB CSVs into `data/raw/geophysics/smu_heatflow/` — `_interp_heatflow` reads any lat/lon/hf-shaped CSV |
| E3 | Cross-basin transfer experiment | **done** | `src/evaluation/cross_basin.py` |
| E4 | InSAR Sentinel-1 5th modality | deferred | future work |
| E5 | Zanskar Big Blind blind validation | **done** | `data/metadata/zanskar_validation.csv` + `src/evaluation/blind_validation.py` |
| E6 | MWe estimation per discovery | **done** | `src/evaluation/mwe_estimation.py` |

## Pipeline ordering (discovery-first)

`scripts/run_all.sh` is now ordered to put discovery in the headline position:

1. **Phase A (data):** download + process — same as before.
2. **Phase B (training):** single discovery model on all labelled data.
3. **Phase C (HEADLINE — discovery):** continental MC-Dropout inference → ranked candidates → MWe estimates → Zanskar blind validation. This is the paper's primary contribution.
4. **Phase D (benchmarking):** LOF-CV against baselines including Mordensky-style capture rates. Supporting evidence for the discovery quality.
5. **Phase E (architectural justification):** ablation, cross-basin transfer, modality analysis. Justifies the architectural choices.
6. **Phase F (figures):** all plots + LaTeX tables.

## Run order on Kaggle

1. **First session (CPU notebook, ~2 hr):** Phase A — data download + processing.
2. **Second session (GPU, ~2 hr):** Phase B + Phase C — train discovery model + headline outputs. **At this point you have the paper's main result.**
3. **Third session (GPU, ~6-10 hr):** Phase D — benchmarking + LOF-CV (you can skip if Phase C looks promising and you want to ship faster).
4. **Fourth session (GPU, ~3 hr):** Phase E + Phase F — ablation, transfer, figures.

## What does NOT change

- Architecture (encoders, contrastive head, fusion, spatial smoothing) — already in place.
- LOF-cluster-out CV — already in place.
- Focal loss + oversampling balance — already corrected.
- Reproducibility scaffolding (configs, scripts, checkpoints) — already in place.
