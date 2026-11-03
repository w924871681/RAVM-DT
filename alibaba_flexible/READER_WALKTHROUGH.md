# Reader walkthrough: E3 with a new Alibaba machine selection

This workflow creates a **new machine split**. It is an additional evaluation,
not an exact rerun of the paper's original machine identities or numerical
results. Prior exposure cannot be excluded without a complete exclusion list.
The code below was prepared without rerunning experiments for this release; a
complete run on official data remains to be independently validated.

## 1. Download and prepare environments

Obtain `machine_usage.tar.gz` using the [official Alibaba Cluster Trace 2018
instructions](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)
and follow the provider's terms. The repository does not redistribute the
archive, original historical E3 arrays, or trained E3 model weights.

Use the separate environments described in `../ENVIRONMENT_GUIDE.md`. The
recorded source-fitting environment is Python 3.11.15 with PyTorch 2.5.1+cu121;
the recorded lifecycle environment is Python 3.13.5 with PyTorch 2.13.0+cu126.
The environment installation is not independently verified on a clean system.

The following Windows PowerShell examples assume the current directory is
`alibaba_flexible` within a full Git checkout. Replace the paths with real paths
on the reader's machine. Output directories must not exist beforehand.

```powershell
$srcpy = 'C:\path\to\source\Scripts\python.exe'
$runpy = 'C:\path\to\lifecycle\Scripts\python.exe'
$archive = 'C:\path\to\machine_usage.tar.gz'
$base = 'D:\ravmdt_e3_new'
& $srcpy .\preflight.py --role source --archive $archive
& $runpy .\preflight.py --role lifecycle --archive $archive
```

Optional `--check-archive-sha256` reads the entire archive and verifies the
registered SHA-256. The selection program also verifies that SHA-256 before
preparation. The preflight only checks source syntax, package metadata and the
paths explicitly passed to it. A pass does not imply training or scientific
validation. Investigate errors or version warnings before continuing.

## 2. Select machines and generate input arrays

```powershell
& $srcpy .\select_and_prepare.py --archive $archive --output "$base\selection" --seed 20261104
& $srcpy .\preflight.py --role source --selection "$base\selection"
```

This command scans the large archive and produces 100 local arrays in four
pools: 20 source, 20 development, 20 calibration and 40 evaluation machines.
The optional `--exclude-hashes` accepts a locally supplied exclusion list, but
no verified historical exclusion list is included. The current lifecycle entry
uses the already fixed paper g04 settings; creating development/calibration
pools is **not** a rerun of historical development selection.

## 3. Generate model weights locally

```powershell
& $srcpy .\fit_source_models.py --selection "$base\selection" --output "$base\sources"
& $srcpy .\preflight.py --role source --selection "$base\selection" --sources "$base\sources"
& $srcpy .\prepare_initial_models.py --selection "$base\selection" --sources "$base\sources" --output "$base\initials"
& $runpy .\preflight.py --role lifecycle --selection "$base\selection" --sources "$base\sources" --initials "$base\initials"
```

The source and initialization commands perform training. They generate new
checkpoints locally; they are not the paper's historical checkpoint bytes.
The preflight checks file presence and manifest bindings; the later evaluation
also checks input and checkpoint hashes when used. Keep generated artifacts
local unless their redistribution is allowed by the provider's terms.

## 4. Evaluate and interpret

```powershell
& $runpy .\run_new_machines.py --selection "$base\selection" --sources "$base\sources" --initials "$base\initials" --output "$base\evaluation"
Get-Content "$base\evaluation\execution_status.json"
```

The default requests 40 machines, 3 seeds, 2 streams and 12 methods, or 2,880
trajectories. Selected subsets using `--center`, `--seed`, `--stream` and
`--arm` are marked `PARTIAL`. `COMPLETE` means this new-selection loop and its
implemented per-trajectory audits completed; it does not mean that the paper's
original numbers, blind-holdout status, figures, or full independent
reproduction have been established. Compare outputs as a separate experiment.
This calculation can require substantial time, storage and compute resources.

For failures preserve the error output, `selection.json`, generated manifests,
software versions, and relevant execution status locally. Check provider rights
before sharing any data-derived content.
