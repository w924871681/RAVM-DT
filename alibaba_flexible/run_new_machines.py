"""Run the published E3 methods on a newly selected Alibaba machine pool.

This is a prospective evaluation, not a replay of the paper's frozen data.
Invocation performs substantial training; importing or --help does not.
"""
import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import shutil
import sys
import types
from pathlib import Path

from fit_source_models import TRAINING, load_selection, read_bound_array, sha

PAPER = TRAINING.parent
STREAMS = {"clean": {"kind": "clean", "alpha": 0.0, "location": "both"},
           "mixed": {"kind": "mixed", "alpha": 0.1, "location": "both"}}
ARMS = [
    {"id": "JOINT__l3__m0", "method": "JOINT", "generator": "U_MSE", "threshold": 5.6036475532487204e-5, "weight": .3},
    {"id": "RECENT_MEAN__g12", "method": "RECENT_MEAN", "generator": "U_MSE", "threshold": .000420273566493654},
    {"id": "MWU_MEAN__g04", "method": "MWU_MEAN", "generator": "U_MSE", "threshold": 2.8018237766243602e-5},
    {"id": "DUAL_MEAN__g12", "method": "DUAL_MEAN", "generator": "U_MSE", "threshold": .000420273566493654},
    {"id": "RAVM_ORIGINAL", "method": "RAVM_ORIGINAL", "generator": "U_MSE", "threshold": 0.},
    {"id": "RAVM_D2B_V02", "method": "RAVM_D2B_V02", "generator": "U_MSE", "threshold": 0.},
    *[{"id": ident, "method": ident, "legacy_native_method": native,
       "generator": "NATIVE_LEGACY", "threshold": 0.} for ident, native in (
        ("ALWAYS_FT", "Always-FT"), ("ER", "ER"), ("EWC", "EWC"),
        ("HUBER_FT", "Huber-FT"), ("VALIDATION_COMMIT", "ValidationCommit"), ("FROZEN", "Frozen"))],
]
LEGACY_CONFIG = {"B": 5, "beta": .1, "kappa_rec": .5, "kappa_hist": 0.,
                 "delta_imp": 2.8018237766243602e-5, "delta_ret": 0.,
                 "validation_retain_all": True, "ewc_lambda": 10., "huber_delta": .1,
                 "ewc_gamma": .9, "ewc_fisher_examples": 32, "clear_factor": .75,
                 "clear_capacity": 128, "clear_ewc_lambda": 200.}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def make_gate():
    from cmc_original_gate_adapter import ORIGINAL_ENGINE_SHA256, OriginalCMCDecision
    import core.v2_engine as v2
    engine = TRAINING / "code/main/runtime/core/v2_engine.py"
    if sha(engine) != ORIGINAL_ENGINE_SHA256:
        raise ValueError("Original gate engine identity mismatch")

    class ReaderGate:
        def __init__(self, *_args):
            self._paired_score = v2._paired_score

        def evaluate(self, candidate, parent, recent_blocks, prior_history_blocks, horizon=1):
            recent, history = tuple(recent_blocks), tuple(prior_history_blocks)[-12:]
            if not recent or not history:
                raise ValueError("Recent and historical blocks are required")
            ur, mr, ar, _, _ = self._paired_score(candidate, parent, recent, horizon, 0.)
            uh, mh, ah, _, _ = self._paired_score(candidate, parent, history, horizon, .5)
            finite = all(math.isfinite(float(x)) for x in (ur, uh, mr, ar, mh, ah))
            recent_pass = finite and ur <= -2.8018237766243602e-5
            history_pass = finite and uh <= 0.
            reason = ("NONFINITE_RETAIN" if not finite else "ACCEPT" if recent_pass and history_pass
                      else "RECENT_AND_HISTORY_FAIL" if not recent_pass and not history_pass
                      else "RECENT_FAIL" if not recent_pass else "HISTORY_VETO")
            return OriginalCMCDecision(float(ur), float(uh), float(mr), float(ar),
                                       float(mh), float(ah), len(recent), len(history),
                                       bool(recent_pass and history_pass), reason)
    return ReaderGate


def bootstrap(output, science_lock):
    import run as existing
    control = output / "control"
    control.mkdir()
    write_json(control / "SCIENCE_EXECUTION.lock.json", science_lock)
    common = types.ModuleType("common")
    common.ROOT, common.PRIOR, common.STORAGE = control, TRAINING / "code/main", output
    common.read, common.sha, common.atomic = existing.read, existing.sha, existing.atomic
    sys.modules["common"] = common
    sys.path[:0] = [str(TRAINING / "code/meta"), str(TRAINING / "code/main"),
                    str(TRAINING / "code/main/runtime"), str(PAPER)]
    bridge = importlib.import_module("bridge")
    bridge.OriginalCMCGate = make_gate()
    import core.v2_engine as v2
    v2.windows = bridge.windows
    from legacy_reporting_precision import install
    install(bridge)
    # The auditor asks d3_runner for the arms of the current phase.
    runner = types.ModuleType("d3_runner")
    runner.arms = lambda _phase: ARMS
    sys.modules["d3_runner"] = runner
    return bridge


def summarize(rows, output):
    import numpy as np
    fields = ("center", "stream", "arm", "mean_service_mse", "harm_fraction", "mean_downside", "commit_fraction", "stages")
    per_machine = []
    with (output / "per_machine.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows):
            values = rows[key]
            n = len(values)
            record = dict(zip(fields, (*key,
                sum(row["deployed_service_mse"] for row in values) / n,
                sum(bool(row["harmful"]) for row in values) / n,
                sum(row["downside"] for row in values) / n,
                sum(bool(row["commit"]) for row in values) / n, n)))
            writer.writerow(record)
            per_machine.append(record)
    columns = ("stream", "arm", "machines", "mean_service_mse", "harm_fraction", "mean_downside", "commit_fraction")
    with (output / "method_stream.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stream, arm in sorted({(r["stream"], r["arm"]) for r in per_machine}):
            group = [r for r in per_machine if r["stream"] == stream and r["arm"] == arm]
            result = {"stream": stream, "arm": arm, "machines": len(group)}
            for key in ("mean_service_mse", "harm_fraction", "mean_downside", "commit_fraction"):
                result[key] = sum(r[key] for r in group) / len(group)
            writer.writerow(result)
    lookup = {(r["center"], r["stream"], r["arm"]): r for r in per_machine}
    centers = sorted({r["center"] for r in per_machine})
    joint, baseline = "JOINT__l3__m0", "DUAL_MEAN__g12"
    if len(centers) >= 2 and all((c, stream, arm) in lookup for c in centers
                                 for stream in ("clean", "mixed") for arm in (joint, baseline)):
        differences = np.array([
            sum(lookup[c, stream, joint]["mean_service_mse"] -
                lookup[c, stream, baseline]["mean_service_mse"] for stream in ("clean", "mixed")) / 2
            for c in centers], dtype=float)
        rng = np.random.default_rng(20261022)
        samples = differences[rng.integers(len(centers), size=(4000, len(centers)))].mean(axis=1)
        write_json(output / "joint_vs_dual_balanced_mse.json", {
            "machine_count": len(centers), "mean_difference_joint_minus_dual": float(differences.mean()),
            "bootstrap_percentile_95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
            "bootstrap_seed": 20261022, "bootstrap_replicates": 4000,
            "unit": "machine", "interpretation": "Exploratory; does not establish equivalence or historical values"})


def evaluate(selection_dir, sources_dir, initials_dir, output, centers, seeds, streams, arms):
    import torch
    from core.models import Forecast, DEVICE, state_hash

    selection_dir = selection_dir.resolve(strict=True)
    sources_dir = sources_dir.resolve(strict=True)
    initials_dir = initials_dir.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    selection = load_selection(selection_dir)
    selection_sha = sha(selection_dir / "selection.json")
    source_info = json.loads((sources_dir / "source_models.json").read_text(encoding="utf-8"))
    init_info = json.loads((initials_dir / "initial_models.json").read_text(encoding="utf-8"))
    if (source_info.get("status") != "NEW_SELECTION_SOURCE_MODELS_TRAINED" or
            init_info.get("status") != "NEW_SELECTION_INITIAL_MODELS_TRAINED"):
        raise ValueError("Source or initial model manifest is incomplete")
    if source_info["selection_sha256"] != selection_sha or init_info["selection_sha256"] != selection_sha:
        raise ValueError("Model manifests refer to a different selection")
    if init_info["source_manifest_sha256"] != sha(sources_dir / "source_models.json"):
        raise ValueError("Initial models refer to a different source manifest")
    eval_rows = {r["center_id"]: r for r in selection["machines"] if r["pool"] == "evaluation"}
    model_rows = {r["seed"]: r for r in source_info["models"]}
    initial_rows = {(r["center"], r["seed"]): r for r in init_info["models"]}
    if len(model_rows) != len(source_info["models"]) or len(initial_rows) != len(init_info["models"]):
        raise ValueError("Duplicate source or initial model identity")
    if not set(centers) <= eval_rows.keys() or not set(seeds) <= model_rows.keys():
        raise ValueError("Requested center or seed is missing")
    if any((c, s) not in initial_rows for c, s in itertools.product(centers, seeds)):
        raise ValueError("Requested initial model is missing")
    all_arms = {a["id"]: a for a in ARMS}
    selected_arms = [all_arms[ident] for ident in arms]
    output.mkdir(parents=True)
    root = output / "new_split"
    (root / "data/attack_masks").mkdir(parents=True)
    (root / "initial").mkdir()
    (root / "reference").mkdir()
    (root / "runtime/core").mkdir(parents=True)
    shutil.copy2(TRAINING / "code/main/runtime/core/engine.py", root / "runtime/core/engine.py")
    write_json(root / "reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json", {"frozen_config": LEGACY_CONFIG})
    write_json(root / "RUN_MANIFEST.json", {"centers": {"E3_NEW": centers}, "seeds": {"E3_NEW": seeds}})
    lock = {"status": "NEW_MACHINE_EVALUATION", "selection_sha256": selection_sha,
            "source_manifest_sha256": sha(sources_dir / "source_models.json"),
            "initial_manifest_sha256": sha(initials_dir / "initial_models.json"),
            "arms": selected_arms, "streams": streams, "centers": centers, "seeds": seeds,
            "original_paper_split": False}
    write_json(root / "EXPERIMENT_LOCK.json", lock)
    for center in centers:
        read_bound_array(selection_dir, eval_rows[center])
        shutil.copy2(selection_dir / "data" / f"center_{center}.npz", root / "data" / f"center_{center}.npz")
    sources = {}
    for seed in seeds:
        source = model_rows[seed]
        path = sources_dir / source["checkpoint"]
        if sha(path) != source["sha256"]:
            raise ValueError("Source checkpoint hash mismatch")
        sources[str(seed)] = {"path": str(path), "sha256": source["sha256"], "state_hash": source["state_hash"]}
        for center in centers:
            row = initial_rows[(center, seed)]
            path = initials_dir / f"{center}_{seed}.pt"
            side = path.with_suffix(".json")
            data = json.loads(side.read_text(encoding="utf-8"))
            if sha(path) != row["sha256"] or data != {k: row[k] for k in data} or data["source_sha256"] != source["sha256"]:
                raise ValueError("Initial model or source binding mismatch")
            model = Forecast("GRU", H=1).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            if state_hash(model) != data["state_hash"]:
                raise ValueError("Initial model state mismatch")
            shutil.copy2(path, root / "initial" / path.name)
            shutil.copy2(side, root / "initial" / side.name)
    bridge = bootstrap(output, lock)
    spec = {"group": "E3_NEW", "phase": "E3_NEW", "root": str(root), "domain": "alibaba",
            "features": 24, "architecture": "GRU", "horizon": 1, "stages": 24,
            "centers": centers, "seeds": seeds, "sources": sources, "arms": selected_arms,
            "stream_ids": streams, "stream_specs": {k: STREAMS[k] for k in streams}}
    grouped = {}
    with bridge.context(spec):
        bridge.torch.set_num_threads(1)
        for center, seed, stream, arm in itertools.product(centers, seeds, streams, selected_arms):
            fn = bridge.run_legacy if arm["id"] in bridge.LEGACY_NAMES else bridge.run_trajectory
            values = fn("E3_NEW", center, seed, stream, arm, "NEW_ALIBABA_MACHINE_SELECTION")
            td = root / "science" / "E3_NEW" / str(center) / str(seed) / stream / arm["id"]
            audit = bridge.audit_trajectory("E3_NEW", td)
            if audit.get("status") != "VALIDATED":
                raise RuntimeError(f"Trajectory audit failed: {td}")
            grouped.setdefault((center, stream, arm["id"]), []).extend(values)
    summarize(grouped, output)
    is_full = len(centers) == 40 and len(seeds) == 3 and len(streams) == 2 and len(arms) == 12
    write_json(output / "execution_status.json", {"status": "COMPLETE" if is_full else "PARTIAL",
               "trajectories": len(centers) * len(seeds) * len(streams) * len(arms),
               "is_paper_historical_result": False, "selection_sha256": selection_sha})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "sources", "initials", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--center", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--stream", choices=list(STREAMS), action="append")
    parser.add_argument("--arm", choices=[a["id"] for a in ARMS], action="append")
    args = parser.parse_args()
    centers = args.center or list(range(20060, 20100))
    seeds = args.seed or [7311, 7312, 7313]
    streams = args.stream or list(STREAMS)
    arms = args.arm or [a["id"] for a in ARMS]
    for items in (centers, seeds, streams, arms):
        if len(items) != len(set(items)):
            parser.error("Repeated centers, seeds, streams or arms are not allowed")
    evaluate(args.selection, args.sources, args.initials, args.output, centers, seeds, streams, arms)


if __name__ == "__main__":
    main()
