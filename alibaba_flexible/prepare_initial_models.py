"""Create per-evaluation-machine warm-up models for a new selection.

This is an explicit reader-invoked training step. No historical model files
are copied, and results must not be labeled as the paper's frozen weights.
"""
import argparse
import json
import sys
from pathlib import Path

from fit_source_models import TRAINING, load_selection, read_bound_array, sha


def prepare(selection_dir, sources_dir, output):
    import torch
    from core.models import Forecast, DEVICE, adapt, tensors, state_hash
    from core.protocol import Segment, windows
    from run import atomic

    selection_dir = selection_dir.resolve(strict=True)
    sources_dir = sources_dir.resolve(strict=True)
    output = output.resolve()
    selection = load_selection(selection_dir)
    source_info = json.loads((sources_dir / "source_models.json").read_text(encoding="utf-8"))
    if source_info.get("status") != "NEW_SELECTION_SOURCE_MODELS_TRAINED":
        raise ValueError("Source model manifest is not complete")
    seeds = [row["seed"] for row in source_info["models"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Source model seeds are empty or duplicated")
    if source_info["selection_sha256"] != sha(selection_dir / "selection.json"):
        raise ValueError("Source models refer to a different machine selection")
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    output.mkdir(parents=True)
    rows = []
    evaluation = [row for row in selection["machines"] if row["pool"] == "evaluation"]
    for source in source_info["models"]:
        seed = source["seed"]
        source_path = sources_dir / source["checkpoint"]
        if sha(source_path) != source["sha256"]:
            raise ValueError("Source checkpoint hash mismatch")
        for row in evaluation:
            center = row["center_id"]
            x, y, ts = read_bound_array(selection_dir, row)
            warm = Segment(x[:800], y[:800], ts[:800], "warmup")
            model = Forecast("GRU", H=1).to(DEVICE)
            model.load_state_dict(torch.load(source_path, map_location=DEVICE, weights_only=True))
            if state_hash(model) != source["state_hash"]:
                raise ValueError("Source model state mismatch")
            initial = adapt(model, tensors(windows(warm, H=1)), steps=50, seed=seed + center)
            path = output / f"{center}_{seed}.pt"
            torch.save(initial.state_dict(), path)
            sidecar = {"sha256": sha(path), "state_hash": state_hash(initial),
                       "source_sha256": source["sha256"], "warmup_hash": warm.hash,
                       "warmup_steps": 50, "warmup_seed": seed + center}
            atomic(path.with_suffix(".json"), sidecar)
            rows.append({"center": center, "seed": seed, **sidecar})
    atomic(output / "initial_models.json", {
        "status": "NEW_SELECTION_INITIAL_MODELS_TRAINED", "models": rows,
        "selection_sha256": sha(selection_dir / "selection.json"),
        "source_manifest_sha256": sha(sources_dir / "source_models.json"),
        "python": sys.version, "torch": str(torch.__version__),
        "historical_paper_weights": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.selection, args.sources, args.output)


if __name__ == "__main__":
    main()
