"""Prepare a new Alibaba machine split; this does not run the experiments.

This is a new selection protocol, not a reconstruction of the paper's split.
It may overlap previously exposed machines. No historical machine identifier
list, original dataset, or trained weights are included.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_SHA256 = "3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a"
MIN_OBSERVATIONS = 16760
MAX_OBSERVATIONS = 17000
POOLS = (("source", 20, 20000), ("development", 20, 20020),
         ("calibration", 20, 20040), ("evaluation", 40, 20060))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_key(machine_id, seed):
    return hashlib.sha256(f"{seed}:{machine_id}".encode("utf-8")).hexdigest(), machine_id


def frames(archive, columns, chunk_size):
    # Read the archive as a stream; member paths are never extracted to disk.
    with tarfile.open(archive, "r:gz") as tar:
        matches = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "machine_usage.csv"]
        if len(matches) != 1:
            raise ValueError("Archive must contain exactly one machine_usage.csv")
        with tar.extractfile(matches[0]) as source:
            yield from pd.read_csv(source, header=None, usecols=columns,
                                   dtype={0: "string"}, chunksize=chunk_size)


def normalize(frame, min_observations=MIN_OBSERVATIONS):
    # Keep the first occurrence of a timestamp, as the existing E3 recipe does.
    frame = frame.drop_duplicates(1).sort_values(1, kind="stable")
    cpu = pd.to_numeric(frame[2], errors="coerce")
    frame = frame[cpu.between(0, 100) & frame[1].notna()].iloc[:MAX_OBSERVATIONS]
    if len(frame) < min_observations:
        return None
    times = pd.to_numeric(frame[1], errors="raise").to_numpy()
    if not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all():
        raise ValueError("timestamps must be finite integer seconds")
    raw = frame.iloc[:, 2:9].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    valid = np.isfinite(raw) & (raw >= 0) & (raw <= 100)
    valid[:, 3] = False  # mkpi has an undocumented scale
    x = np.zeros((len(frame), 12), dtype="float32")
    mask = np.zeros_like(x)
    x[:, :7] = np.where(valid, raw / 100, 0)
    mask[:, :7] = valid
    return np.concatenate([x, mask], axis=1), raw[:, 0] / 100, times.astype("int64")


def pool_for(position):
    for name, count, first_center in POOLS:
        if position < count:
            return name, first_center + position
        position -= count
    raise ValueError("too many machines")


def prepare(archive, output, seed, batch_size, chunk_size, exclude_hashes):
    archive = archive.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing data are not overwritten")
    digest = file_sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the registered Alibaba archive")
    identifiers = set()
    for chunk in frames(archive, [0], chunk_size):
        identifiers.update(chunk[0].dropna().astype(str).unique())
    candidates = sorted((value for value in identifiers if hashlib.sha256(
        value.encode("utf-8")).hexdigest() not in exclude_hashes),
        key=lambda value: rank_key(value, seed))
    required = sum(pool[1] for pool in POOLS)
    if len(candidates) < required:
        raise ValueError(f"Only {len(candidates)} machine IDs found; {required} are required")
    output.mkdir(parents=True)
    data_dir = output / "data"
    data_dir.mkdir()
    selected = []
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset:offset + batch_size]
        wanted = set(batch)
        collected = {machine: [] for machine in batch}
        for chunk in frames(archive, list(range(9)), chunk_size):
            subset = chunk[chunk[0].isin(wanted)]
            for machine, group in subset.groupby(0, sort=False):
                collected[str(machine)].append(group)
        for machine in batch:
            if not collected[machine]:
                continue
            arrays = normalize(pd.concat(collected[machine], ignore_index=True))
            if arrays is None:
                continue
            pool, center = pool_for(len(selected))
            meta = {"center_id": center, "pool": pool, "domain": "alibaba",
                    "machine_id_hash": hashlib.sha256(machine.encode("utf-8")).hexdigest(),
                    "selection_seed": seed,
                    "selection_protocol": "new_selection_not_verified_disjoint_from_history"}
            x, y, ts = arrays
            path = data_dir / f"center_{center}.npz"
            np.savez_compressed(path, x=x, y=y, ts=ts, meta=json.dumps(meta, sort_keys=True))
            selected.append({**meta, "array_sha256": file_sha256(path)})
            if len(selected) == required:
                break
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"Only {len(selected)} eligible machines found; output is incomplete")
    (output / "selection.json").write_text(json.dumps({
        "status": "INPUTS_PREPARED_ONLY", "archive_sha256": digest,
        "selection_seed": seed, "ranking": "SHA256(<seed>:<machine-id>)",
        "excluded_machine_hashes_count": len(exclude_hashes),
        "selection_scope": "new split; prior exposure is not automatically ruled out",
        "min_observations": MIN_OBSERVATIONS, "max_observations": MAX_OBSERVATIONS,
        "machines": selected, "trained_models": False, "experiments_run": False,
    }, indent=2), encoding="utf-8")
    print(f"Prepared {len(selected)} machine arrays in {output}; no experiments ran.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=20261104, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--chunk-size", default=100000, type=int)
    parser.add_argument("--exclude-hashes", type=Path,
                        help="Optional local UTF-8 file, one SHA-256 machine-ID hash per line")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0:
        parser.error("batch and chunk sizes must be positive")
    excluded = set()
    if args.exclude_hashes:
        excluded = {line.strip().lower() for line in args.exclude_hashes.read_text(
            encoding="utf-8").splitlines() if line.strip()}
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
               for value in excluded):
            parser.error("every excluded machine hash must be a 64-digit SHA-256 hex string")
    prepare(args.archive, args.output, args.seed, args.batch_size, args.chunk_size, excluded)


if __name__ == "__main__":
    main()
