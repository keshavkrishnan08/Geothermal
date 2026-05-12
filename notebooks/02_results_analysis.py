"""Kaggle/Jupyter notebook: results analysis (Phase 2).

Loads everything produced by the evaluation suite and prints the headline
numbers used in the paper.
"""
# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "outputs" / "results"

# %%
# Table 1
t1 = res / "table1_main_results.csv"
if t1.exists():
    print("=== Table 1 (LOF-CV main results) ===")
    print(pd.read_csv(t1).to_string(index=False))

# %%
# Ablation
t2 = res / "table2_ablation.csv"
if t2.exists():
    print("\n=== Table 2 (Ablation) ===")
    print(pd.read_csv(t2).to_string(index=False))

# %%
# Discoveries
t3 = res / "table3_discoveries.csv"
if t3.exists():
    df = pd.read_csv(t3)
    print(f"\n=== Table 3 (Discoveries — {len(df)} sites) ===")
    print(df.head(15).to_string(index=False))
    print(f"\nPlausible share: {(df['plausibility']=='plausible').mean():.1%}")

# %%
# Statistical tests
st = res / "statistical_tests.csv"
if st.exists():
    print("\n=== Paired t-tests (Holm-Bonferroni) ===")
    print(pd.read_csv(st).to_string(index=False))

# %%
# Permutation importance
pi = res / "permutation_importance.csv"
if pi.exists():
    print("\n=== Permutation importance ===")
    print(pd.read_csv(pi).to_string(index=False))

# %%
# Moran's I
mi = res / "morans_i.json"
if mi.exists():
    print("\n=== Moran's I ===")
    print(json.load(open(mi)))
