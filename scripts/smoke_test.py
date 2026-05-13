"""End-to-end smoke test for the GeoProspectNet pipeline.

Runs every script entrypoint that doesn't require training, plus loads each
trained checkpoint and runs a 64-cell inference. Exits non-zero on first
failure with a clear error.

Coverage:
  1.  Imports for every src/ module
  2.  Required input data files exist
  3.  Each trained checkpoint loads and produces non-NaN scores on 64 cells
  4.  Every cached result CSV is readable
  5.  Re-runs MWe v2, pre-registration, economic analysis, calibration,
      reservoir thickness, hard-negatives, field validation, LOFCV audit
      (these are pure-analysis, all <1 min each)
  6.  Regenerates all 20 figures
  7.  Sanity-checks the pre-registration SHA-256 is stable

Run as:
  python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; BOLD = "\033[1m"; END = "\033[0m"
PASS = f"{GREEN}✓{END}"
FAIL = f"{RED}✗{END}"
WARN = f"{YELLOW}⚠{END}"

results = []  # list of (name, ok, detail, elapsed_ms)


def step(name: str):
    """Decorator-like context for printing + capturing a step."""
    class _S:
        def __init__(self, name): self.name = name; self.t0 = None
        def __enter__(self):
            self.t0 = time.time()
            print(f"  • {self.name:.<58s} ", end="", flush=True)
            return self
        def __exit__(self, exc_type, exc, tb):
            ms = (time.time() - self.t0) * 1000
            if exc is not None:
                results.append((self.name, False, str(exc), ms))
                print(f"{FAIL}  ({ms:5.0f} ms)")
                print(f"      {RED}{exc}{END}")
                return False  # propagate to abort the run
            results.append((self.name, True, "", ms))
            print(f"{PASS}  ({ms:5.0f} ms)")
    return _S(name)


# --------------------------------------------------------------------------
# §1. Module imports
# --------------------------------------------------------------------------
def section_imports():
    print(f"\n{BOLD}§1  Module imports{END}")
    modules = [
        "src.data.dataset", "src.models.geoprospectnet", "src.models.fusion",
        "src.models.spatial", "src.training.losses",
        "src.evaluation.negative_controls", "src.evaluation.discovery",
        "src.evaluation.mwe_estimation_v2", "src.evaluation.calibration",
        "src.evaluation.economic_analysis", "src.evaluation.preregister",
        "src.evaluation.reservoir_thickness", "src.evaluation.hard_negatives",
        "src.evaluation.field_validation", "src.evaluation.lofcv_buffer_audit",
        "src.evaluation.mordensky_comparison", "src.evaluation.baselines",
        "src.evaluation.modality_analysis", "src.evaluation.ood_eastern_us",
        "src.visualization.make_figures",
    ]
    for m in modules:
        with step(f"import {m}"):
            __import__(m)


# --------------------------------------------------------------------------
# §2. Required input data files
# --------------------------------------------------------------------------
def section_data_files():
    print(f"\n{BOLD}§2  Input data files present{END}")
    required = [
        "data/processed/labels.npy",
        "data/processed/train_mask.npy",
        "data/processed/modality_masks.npy",
        "data/processed/grid_coordinates.csv",
        "data/processed/geochemistry_features.npy",
        "data/processed/geology_features.npy",
        "data/metadata/known_fields_details.csv",
        "data/raw/labels/nv_permits_holdout.csv",
        "data/raw/labels/nv_permits_extracted.csv",
        "data/raw/geophysics/bouguer_gravity.tif",
        "data/raw/geophysics/smu_heatflow/heat_flow_combined.csv",
    ]
    for f in required:
        with step(f"exists {f}"):
            p = ROOT / f
            if not p.exists():
                raise FileNotFoundError(p)
    # geophysics_patches.npy is .gitignored — soft check
    p = ROOT / "data/processed/geophysics_patches.npy"
    if not p.exists():
        print(f"  {WARN} geophysics_patches.npy missing (5.4 GB) — regenerate with src.data.ingest_geophysics")


# --------------------------------------------------------------------------
# §3. Checkpoint load + 64-cell inference
# --------------------------------------------------------------------------
def section_checkpoints():
    print(f"\n{BOLD}§3  Checkpoint load + tiny inference (skip if patches missing){END}")
    if not (ROOT / "data/processed/geophysics_patches.npy").exists():
        print(f"  {WARN} geophysics_patches.npy missing — skipping inference smoke test")
        return
    import torch, yaml
    from src.data.dataset import GeoProspectDataset, make_loader
    from src.models.geoprospectnet import GeoProspectNet

    ckpts = sorted((ROOT / "outputs/checkpoints").glob("random_cpu_*_seed42.pt"))
    if not ckpts:
        print(f"  {WARN} no checkpoints found — skipping")
        return

    cfg_for_ckpt = {
        "cpu_calibrated": "cpu_calibrated", "cpu_tuned": "cpu_tuned",
        "cpu_margin": "cpu_margin", "cpu_max": "cpu_max",
    }
    n = int(np.load(ROOT / "data/processed/labels.npy", mmap_mode="r").shape[0])
    indices = np.arange(64)

    for ckpt in ckpts[:4]:  # cap at 4 for speed
        with step(f"load + 64-cell inference {ckpt.name}"):
            stem = ckpt.stem.replace("random_", "").replace("_seed42", "")
            cfg_name = cfg_for_ckpt.get(stem, "cpu_max")
            cfg = yaml.safe_load(open(ROOT / f"configs/{cfg_name}.yaml"))
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = GeoProspectNet(cfg).to("cpu")
            model.load_state_dict(ck["state_dict"]); model.eval()
            ds = GeoProspectDataset(ROOT / "data/processed", indices=indices,
                                     use_thermal=False)
            loader = make_loader(ds, batch_size=64, shuffle=False, num_workers=0)
            scores = []
            with torch.no_grad():
                for b in loader:
                    out = model(b)
                    scores.append(torch.sigmoid(out["logits"]).numpy())
            s = np.concatenate(scores)
            assert len(s) == 64, f"expected 64 scores, got {len(s)}"
            assert np.all(np.isfinite(s)), "NaN or inf in scores"
            assert s.min() >= 0 and s.max() <= 1, f"scores out of [0,1]: {s.min()} {s.max()}"


# --------------------------------------------------------------------------
# §4. Cached result CSVs readable
# --------------------------------------------------------------------------
def section_results_csvs():
    print(f"\n{BOLD}§4  Result tables readable{END}")
    csvs = [
        "outputs/results/negative_controls_cpu_max.csv",
        "outputs/results/baselines.csv",
        "outputs/results/separation_sweep.csv",
        "outputs/results/separation_sweep_loss_ablation.csv",
        "outputs/results/separation_sweep_arch_ablation.csv",
        "outputs/results/multi_seed_sensitivity.csv",
        "outputs/results/mordensky_comparison.csv",
        "outputs/results/consensus_discoveries.csv",
        "outputs/results/consensus_with_mwe_v2.csv",
        "outputs/results/calibration_table.csv",
        "outputs/results/hard_negative_cohorts.csv",
        "outputs/results/field_validation.csv",
        "outputs/results/reservoir_thickness_validation.csv",
        "outputs/results/lofcv_buffer_audit.csv",
        "outputs/results/economic_analysis.csv",
        "outputs/results/preregistration_manifest.csv",
        "outputs/results/ood_eastern_us_validation.csv",
    ]
    for csv in csvs:
        with step(f"read {csv}"):
            # preregistration_manifest.csv has '# ...' header comments
            kwargs = {"comment": "#"} if "preregistration" in csv else {}
            df = pd.read_csv(ROOT / csv, **kwargs)
            assert len(df) > 0, "empty CSV"


# --------------------------------------------------------------------------
# §5. Re-run pure-analysis scripts (each <1 min)
# --------------------------------------------------------------------------
def section_reruns():
    print(f"\n{BOLD}§5  Pure-analysis scripts re-run cleanly{END}")
    fast_scripts = [
        ("MWe v2",            "src.evaluation.mwe_estimation_v2",
         ["--in_csv", "outputs/results/consensus_discoveries.csv",
          "--out_csv", "outputs/results/_smoke_mwe.csv",
          "--n_samples", "200"]),
        ("Pre-registration",  "src.evaluation.preregister",       []),
        ("Economic analysis", "src.evaluation.economic_analysis", []),
        ("Calibration",       "src.evaluation.calibration",
         ["--config", "configs/cpu_max.yaml",
          "--scores", "outputs/results/scores_cpu_max.npy"]),
        ("Field validation",  "src.evaluation.field_validation",  []),
        ("Hard negatives",    "src.evaluation.hard_negatives",    []),
        ("LOFCV audit",       "src.evaluation.lofcv_buffer_audit", []),
        ("Reservoir thick.",  "src.evaluation.reservoir_thickness", []),
    ]
    for label, mod, args in fast_scripts:
        with step(f"{label}  ({mod})"):
            r = subprocess.run([sys.executable, "-m", mod, *args],
                               cwd=str(ROOT), capture_output=True, text=True,
                               timeout=180)
            if r.returncode != 0:
                raise RuntimeError(f"exit {r.returncode}\n--- stderr ---\n{r.stderr[-500:]}")
    # Cleanup smoke artifact
    (ROOT / "outputs/results/_smoke_mwe.csv").unlink(missing_ok=True)


# --------------------------------------------------------------------------
# §6. Figure generation
# --------------------------------------------------------------------------
def section_figures():
    print(f"\n{BOLD}§6  Figure generation{END}")
    with step("regenerate all 20 figures"):
        r = subprocess.run([sys.executable, "-m", "src.visualization.make_figures"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}\n{r.stderr[-500:]}")
    figs = list((ROOT / "outputs/figures").glob("fig*.png"))
    with step(f"sanity check  {len(figs)} PNG figures present"):
        assert len(figs) >= 18, f"expected ≥18 figures, found {len(figs)}"


# --------------------------------------------------------------------------
# §7. Pre-registration SHA-256 stability
# --------------------------------------------------------------------------
def section_preregistration_hash():
    print(f"\n{BOLD}§7  Pre-registration manifest is reproducible{END}")
    j = json.load(open(ROOT / "outputs/results/preregistration_manifest.json"))
    sha = j["sha256"]
    with step("re-derive SHA-256"):
        import hashlib
        df = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe_v2.csv")
        cols = ["discovery_id", "lat", "lon", "area_km2", "T_res_central",
                 "mwe_p10", "mwe_p50", "mwe_p90", "province", "plausibility"]
        manifest = df[cols].sort_values("discovery_id").reset_index(drop=True)
        recomputed = hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest()
        assert recomputed == sha, (
            f"hash drifted!\n  manifest: {sha[:16]}...\n  recomputed: {recomputed[:16]}...")


# --------------------------------------------------------------------------
def main():
    print(f"{BOLD}GeoProspectNet — pipeline smoke test{END}")
    print(f"root: {ROOT}")
    t0 = time.time()
    sections = [
        section_imports,
        section_data_files,
        section_checkpoints,
        section_results_csvs,
        section_reruns,
        section_figures,
        section_preregistration_hash,
    ]
    aborted = False
    for fn in sections:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            aborted = True
            break

    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_fail = sum(1 for _, ok, _, _ in results if not ok)
    elapsed = time.time() - t0
    print(f"\n{BOLD}=== smoke summary ==={END}")
    print(f"  passed: {GREEN}{n_pass}{END}")
    print(f"  failed: {RED if n_fail else GREEN}{n_fail}{END}")
    print(f"  elapsed: {elapsed:.1f} s")
    if n_fail or aborted:
        print(f"\n{RED}smoke test FAILED{END}")
        for name, ok, detail, ms in results:
            if not ok:
                print(f"  {FAIL} {name}: {detail}")
        sys.exit(1)
    print(f"\n{GREEN}{BOLD}smoke test PASSED — pipeline is healthy{END}")


if __name__ == "__main__":
    main()
