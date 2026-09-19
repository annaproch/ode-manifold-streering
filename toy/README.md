# Toys for the manifold-constrained ODESteer plan

Synthetic CPU checks of the steering mechanism used in [`../experiment-plan.md`](../experiment-plan.md) §9. They say nothing about any LLM.

| File | What it does |
|---|---|
| `ring_toy.py` | Hue ring + lightness axis in a 3-D subspace of a 256-D space; ODESteer's own `NormedPolyClassifier`; all ring conditions (A, B, C, D, E1, E2 and its ablations) |
| `run_all.py` | Regenerates the ring tables (§9.1, §9.3) → `results/tables.md`, `results/runs.jsonl` |
| `geometries.py` | Horseshoe and Y-branch curves; B, E2 and E3 (graph score) → `results/geometries.md` (§9.4) |

## Run

```bash
toy/setup.sh                                   # fetches ODESteer at a pinned commit into toy/third_party, makes toy/.venv
cd toy
.venv/bin/python run_all.py --worlds 3 --seeds 2      # ~4 min on 5 CPU cores
.venv/bin/python geometries.py --worlds 3 --seeds 2   # ~3 min
```

ODESteer is fetched, not vendored: its repository has no license file. Set `ODESTEER_SRC` to use another checkout.

A replicate = one world (fresh data, noise, random sketch, classifier, KDE; seeded by `--worlds` index) × one set of 32 noisy starts (seeded by `--seeds` index). Tables report mean ± SD over replicates.
