"""Fit new Alibaba source models from an alternative local machine selection.

This command is for a reader to invoke. It does not run when imported.
Weights are produced locally, not included in the repository.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

TRAINING = (Path(__file__).resolve().parent.parent / "g04_论文版本" / "代码" /
            "实验代码" / "Paper2_RAVM_DT" / "training")
sys.path[:0] = [str(TRAINING), str(TRAINING / "code/main/runtime")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selection(folder):
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    machines = selection["machines"]
    ids = [row["center_id"] for row in machines]
    hashes = [row["machine_id_hash"] for row in machines]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("Machine selection contains duplicate center or machine identities")
    source = [row["center_id"] for row in machines if row["pool"] == "source"]
    evaluation = [row["center_id"] for row in machines if row["pool"] == "evaluation"]
    if source != list(range(20000, 20020)) or evaluation != list(range(20060, 20100)):
        raise ValueError("Expected 20 source and 40 evaluation centers in registered order")
    return selection


def read_bound_array(folder, row):
    import numpy as np
    path = folder / "data" / f"center_{row['center_id']}.npz"
    if sha(path) != row["array_sha256"]:
        raise ValueError(f"Selected array changed: {path.name}")
    with np.load(path, allow_pickle=False) as z:
        x, y, ts = z["x"], z["y"], z["ts"]
        meta = json.loads(str(z["meta"]))
    if any(meta[key] != row[key] for key in ("pool", "center_id", "machine_id_hash")):
        raise ValueError(f"Selection metadata mismatch: {path.name}")
    if (x.ndim != 2 or x.shape[1] != 24 or len(x) != len(y) or len(y) != len(ts)
            or len(y) < 16760 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.all(np.diff(ts) > 0)):
        raise ValueError(f"Invalid time-series arrays: {path.name}")
    return x, y, ts


def fit(selection_dir, output, seeds):
    import numpy as np
    import torch
    from core.models import Forecast, DEVICE, tensors, state_hash
    from core.protocol import Segment, windows
    from run import atomic, selected_functions

    selection_dir = selection_dir.resolve(strict=True)
    output = output.resolve()
    selection = load_selection(selection_dir)
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    (output / "logs").mkdir()

    class SourceTimeline:
        def __init__(self, center):
            self.center = center

        def source(self):
            row = next(item for item in selection["machines"] if item["center_id"] == self.center)
            x, y, ts = read_bound_array(selection_dir, row)
            return Segment(x[:1000], y[:1000], ts[:1000], "source")

    env = dict(ROOT=output, POOLS={"source": list(range(20000, 20020))},
               Timeline=SourceTimeline, windows=windows, Forecast=Forecast,
               DEVICE=DEVICE, np=np, torch=torch, tensors=tensors,
               state_hash=state_hash, time=time, save_json=atomic)
    selected_functions(TRAINING / "code/recipes/historical_pretrain.py", ["pretrain"], env)
    results = []
    for seed in seeds:
        env["pretrain"](seed, "GRU", 1, False, "alibaba")
        path = checkpoint_dir / f"alibaba_source_GRU_H1_{seed}.pt"
        model = Forecast("GRU", H=1).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        results.append({"seed": seed, "checkpoint": str(path.relative_to(output)),
                        "sha256": sha(path), "state_hash": state_hash(model)})
    atomic(output / "source_models.json", {
        "status": "NEW_SELECTION_SOURCE_MODELS_TRAINED", "models": results,
        "selection_sha256": sha(selection_dir / "selection.json"),
        "selection_seed": selection["selection_seed"],
        "source_code_sha256": sha(TRAINING / "code/recipes/historical_pretrain.py"),
        "python": sys.version, "torch": str(torch.__version__),
        "historical_paper_weights": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    seeds = args.seeds or [7311, 7312, 7313]
    if len(seeds) != len(set(seeds)):
        parser.error("model seeds must be unique")
    fit(args.selection, args.output, seeds)


if __name__ == "__main__":
    main()
