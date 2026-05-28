"""
Smoke test: load -> sample -> baseline LCCDE -> BGWO bi -> BGWO+SHAP tri
            -> metrics -> save one plot.

Runs on a tiny config (~3K rows, BGWO pop=4 iter=3, 1 seed) using the
local CIC-IDS2017 sample CSV from the baseline repo. Goal: prove the
whole pipeline wires together end-to-end. Performance numbers are NOT
expected to be meaningful.
"""
from __future__ import annotations
import sys
from pathlib import Path
import traceback
import json

# Make src/ importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import smoke_config, seed_everything
from src.evaluation import run_one
from src.plots import (
    plot_per_class_f1, plot_bgwo_convergence, plot_pareto,
)


def main() -> int:
    cfg = smoke_config()

    # Point loader at the baseline repo's bundled CSV.
    baseline_csv = (
        ROOT.parent
        / "Intrusion-Detection-System-Using-Machine-Learning-main"
        / "data"
        / "CICIDS2017_sample.csv"
    )
    if not baseline_csv.exists():
        print(f"[smoke] FAIL: cannot find {baseline_csv}", flush=True)
        return 2
    cfg.local_cicids_csv = baseline_csv
    cfg.ensure_dirs()

    print("[smoke] config:")
    print(cfg.to_json())

    summaries = {}
    for method in ("none", "filter", "bgwo_bi", "bgwo_shap"):
        print(f"\n=========== smoke: {method} ===========")
        try:
            r = run_one(cfg, method=method, seed=0)
        except Exception:
            traceback.print_exc()
            return 3
        summaries[method] = {
            "macro_f1": round(r.macro_f1, 4),
            "accuracy": round(r.accuracy, 4),
            "n_selected": r.n_features_selected,
            "n_total":    r.n_features_total,
            "latency_ms_per_flow": round(r.latency_ms_per_flow, 4),
            "fidelity":  None if r.fidelity is None else round(r.fidelity, 4),
        }
        # Save one plot per method.
        try:
            plot_per_class_f1(
                r.per_class_f1, {c: str(c) for c in r.per_class_f1},
                cfg.results_dir, f"smoke_{method}_per_class_f1",
            )
            if r.fitness_history:
                plot_bgwo_convergence(
                    r.fitness_history, cfg.results_dir,
                    f"smoke_{method}_bgwo_convergence",
                )
            if r.pareto_points:
                plot_pareto(
                    r.pareto_points, cfg.results_dir,
                    f"smoke_{method}_pareto",
                )
        except Exception as e:
            print(f"[smoke] plotting warning for {method}: {e}")

    print("\n========== SMOKE SUMMARY ==========")
    print(json.dumps(summaries, indent=2))
    print(f"\n[smoke] plots + run JSONs saved under: {cfg.results_dir}")
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
