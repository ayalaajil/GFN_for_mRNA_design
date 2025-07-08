"""
Systematic weight-sweep over [w_gc, w_mfe, w_cai] on a regular grid.
For each (w_gc, w_mfe, w_cai), samples sequences via evaluate(), computes per-objective means and hypervolume, then plots the hypervolume heatmap.
"""


import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt


import pygmo as pg

from env import CodonDesignEnv
from evaluate import evaluate, load_trained_model

# --- CONFIGURATION ---
N_SAMPLES = 50           # # of sequences per weight
GRID_SIZE = 20           # defines a GRID_SIZE x GRID_SIZE scan
REF_POINT = [1.0, 0.0, 1.0]  


OUTPUT_DIR = "weight_sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

env, sampler = load_trained_model("trained_gflownet.pth")

# --- GRID SWEEP ---
grid = np.linspace(0, 1, GRID_SIZE)
records = []

for w1 in grid:
    for w2 in grid:
        if w1 + w2 > 1.0:
            continue
        w3 = 1.0 - (w1 + w2)
        weights = torch.tensor([w1, w2, w3], dtype=torch.float32)

        # sample
        samples, gc_list, mfe_list, cai_list = evaluate(
            env, sampler, weights=weights, n_samples=N_SAMPLES
        )

        # compute means
        mean_gc  = float(np.mean(gc_list))
        mean_mfe = float(np.mean(mfe_list))
        mean_cai = float(np.mean(cai_list))

        # prepare points for hypervolume (flip MFE to max)
        points = np.vstack([
            gc_list,
            -np.array(mfe_list),
            cai_list
        ]).T

        # compute hypervolume
        hv_calc = pg.hypervolume(points)
        hv = hv_calc.compute(REF_POINT)

        records.append({
            "w_gc": w1, "w_mfe": w2, "w_cai": w3,
            "mean_gc": mean_gc, "mean_mfe": mean_mfe,
            "mean_cai": mean_cai, "hypervolume": hv
        })
        print(f"w=({w1:.2f},{w2:.2f},{w3:.2f}) -> HV={hv:.4f}")


df = pd.DataFrame(records)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_DIR, f"weight_sweep_{ts}.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")


# --- PLOT HEATMAP of Hypervolume ---

pivot = df.pivot(index="w_gc", columns="w_mfe", values="hypervolume")

plt.figure(figsize=(6,5))

plt.imshow(
    pivot.values,
    origin="lower",
    extent=[pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max()],
    aspect="auto"
)

plt.colorbar(label="Hypervolume")
plt.xlabel("w_mfe")
plt.ylabel("w_gc")
plt.title("Hypervolume over weight grid (w_cai=1−w_gc−w_mfe)")

heatmap_path = os.path.join(OUTPUT_DIR, f"hypervolume_heatmap_{ts}.png")
plt.tight_layout()
plt.savefig(heatmap_path)
print(f"Heatmap saved to {heatmap_path}")
plt.show()

