"""Prepare the g01 sensitivity plan for published non-Alibaba groups only.

This is a subset of the original historical plan. It never claims a full
17-group execution, and training starts only if the reader passes --run.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "配置与必要输入"
REPO = HERE / "实验代码/Paper2_RAVM_DT"
VERSION = HERE.parent.resolve()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.source_python.is_file():
        raise FileNotFoundError("The source-environment Python is missing")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    template = json.loads((INPUTS / "PLAN_TEMPLATE.json").read_text(encoding="utf-8-sig"))
    bindings = []
    replaced_manifest = None
    for row in template["bindings"]:
        target = (INPUTS / row["path"]).resolve()
        if not target.is_relative_to(VERSION):
            raise ValueError("Binding outside the version directory")
        if target.name == "E3.json" and target.parent.name == "groups":
            continue
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha(target)
        if target.name == "ASSET_MANIFEST.json" and target.parent.name == "training":
            manifest = json.loads(target.read_text(encoding="utf-8-sig"))
            if manifest.get("schema") != "synthetic-and-e4-subset-manifest-v1" or manifest.get("groups") != 16:
                raise ValueError("Unexpected published 16-group manifest")
            replaced_manifest = {"historical_sha256": row["sha256"], "subset_sha256": actual}
        elif actual != row["sha256"]:
            raise ValueError("Original scientific binding changed: " + str(target))
        bindings.append({"path": str(target), "sha256": actual})
    if replaced_manifest is None:
        raise ValueError("Subset asset manifest binding is missing")
    groups = [g for g in template["groups"] if g != "E3"]
    tasks = [t for t in template["tasks"] if t["group"] != "E3"]
    ordinary = [t for t in tasks if not t["group"].startswith("E5_COST_")]
    costs = [t for t in tasks if t["group"].startswith("E5_COST_")]
    if len(groups) != 16 or len(ordinary) != 1854 or len(costs) != 3:
        raise ValueError("Unexpected non-E3 task count")
    template.update(
        status="PUBLIC_NON_E3_SUBSET_NOT_FULL_PAPER", bindings=bindings,
        groups=groups, tasks=tasks, counts=[c for c in template["counts"] if c["group"] != "E3"],
        ordinary_total=len(ordinary), cost_total=len(costs),
        historical_lock=str((INPUTS / "g04_lock.json").resolve()),
        observed_lock=str((INPUTS / "g01_lock.json").resolve()),
        reuse_sources=str((INPUTS / "sources").resolve()),
        reuse_initials=str((INPUTS / "prepared").resolve()),
        source_python=str(args.source_python.resolve()),
        published_subset_manifest_binding=replaced_manifest,
        excluded_groups=["E3"], full_independent_delivery_still_required=True,
    )
    output.mkdir(parents=True)
    planfile, taskfile = output / "PORTABLE_BRANCH_PLAN.json", output / "ORDINARY_TASKS.json"
    planfile.write_text(json.dumps(template, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    taskfile.write_text(json.dumps(ordinary, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PREPARED_NON_E3_ONLY", "ordinary_tasks": len(ordinary),
                      "cost_tasks": len(costs), "training_started": args.run}))
    if args.run:
        subprocess.run([sys.executable, "-X", "utf8", "-B", str(REPO / "reproduce_selection_branch.py"),
                        "--plan", str(planfile), "--tasks", str(taskfile),
                        "--output", str(output / "new_result")], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
