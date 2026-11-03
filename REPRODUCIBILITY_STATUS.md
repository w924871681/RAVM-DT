# Reproduction inputs and current limitations

This repository is a **partial release**. The published historical results can be inspected, but the public files do not yet support end-to-end reproduction. The notes below describe the existing implementation; they are not a claim that the public workflow has passed in a clean environment.

## Data provided by third parties

Obtain the following datasets directly from their providers, comply with their terms, and retain the required attribution. Raw data are not distributed in this repository.

- [Alibaba Cluster Trace 2018](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md): follow the provider’s download process and obtain `machine_usage.tar.gz`. The provider lists SHA-256 `3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a` for this archive.
- [QUANT sensor systems data](https://zenodo.org/records/10775692): obtain `QUANT_SensorSystems_hourly.csv`. The file used locally has SHA-256 `a8fa398aacd7bb5c5c320df6675a8dccb146d4e44d0c6f392b4d5acf76fc4d1f`. The provider identifies the sensor file’s underlying terms as Open Government Licence v3.0, with source acknowledgment.
- [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction): obtain `energydata_complete.csv` and follow the dataset’s CC BY 4.0 attribution terms.
- [ThermBuild](https://doi.org/10.24406/fordatis/445): consult the provider’s data record and licence before use.

Check SHA-256 locally after download, for example with `sha256sum <file>` or PowerShell `Get-FileHash -Algorithm SHA256 <file>`. A correct raw-file hash identifies the download; it does not supply missing experiment configurations or establish that all experiments have been reproduced.

## Alibaba E3 reconstruction dependency

The existing `training/prepare_inputs.py` selects registered machines using `training/source_inputs/alibaba_pool_manifest.json`, then creates numerical arrays from the official archive. That machine mapping is not in the current public release. Replacing the registered machines with arbitrary machines changes the experimental inputs and cannot be presented as the same reproduction.

A prior local check reconstructed 20 source and 40 target arrays matching registered SHA-256 values. That check used the local mapping and is not a public end-to-end test. The present repository does not yet provide an authorised, complete public substitute for the mapping and all required assets.

The existing source-fitting recipe uses the selected source arrays to fit source models; `historical_alibaba_initial.py` then constructs per-center initial models from a source checkpoint and warm-up data. These scripts require missing configurations, identity records and inputs. Initial-identity metadata alone is not a trained model. Alibaba-derived data and model weights are not redistributed here while their redistribution rights remain unresolved.

An [alternative-machine E3 candidate](alibaba_flexible/README.md) is now supplied. It uses a new selection and locally generated weights rather than the missing historical Alibaba weights. Its interfaces have been checked with synthetic or mocked inputs; it has not been run end to end on the official archive, and no claim of complete reproduction or new-machine generalization has been established.

## Runtime and acceptance boundary

The historical records specify a source-fitting environment of Python 3.11.15 and PyTorch 2.5.1+cu121, and a lifecycle environment of Python 3.13.5 and PyTorch 2.13.0+cu126. The existing records describe the original Windows/NVIDIA setup; a complete fresh installation, compatibility on other systems and end-to-end execution of this public subset have not been established. Do not assume that a later run produces bit-identical checkpoints merely from these version numbers.

The current execution code verifies the full asset manifest and stages inputs before source-to-final execution. Consequently, obtaining the raw archives alone is insufficient. Further work must supply a rights-cleared machine-selection mechanism, the required configuration and identity files, legally available or reconstructible inputs, model-generation instructions and a validated installation path. Until these dependencies are addressed, the public repository cannot support a complete independent run or be described as fully reproducible.

The paper’s historical g04 results, its historical timing-based choice, and the later g01 sensitivity analysis are distinct; consult the main README before interpreting them. Saved summary tables and plots are not evidence that a reader has rerun the experiments.
