# Official data acquisition for the public subset

Original provider archives are not included. Follow each provider's current access, attribution and licence terms. Obtaining the data does not by itself verify a complete experiment.

## Alibaba Cluster Trace 2018

1. Follow the survey/download instructions on the [official Alibaba Cluster Trace v2018 page](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md).
2. Obtain `machine_usage.tar.gz`. The provider lists SHA-256 `3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a`.
3. The [alternative-machine E3 candidate](alibaba_flexible/README.md) selects and prepares a new split locally. It has not been validated end to end on the official archive. The original E3 machine mapping and historical E3 weights are not distributed.

## UCI Appliances Energy Prediction

1. Download the dataset from the [official UCI record](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), extract `energydata_complete.csv`, and retain the provider's CC BY 4.0 attribution.
2. The study's CSV SHA-256 is `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`.
3. The original E4 group definitions refer to `training/raw/uci/energydata_complete.csv`. The published E4 subset also includes processed arrays; original raw data can be prepared locally with `training/prepare_inputs.py uci --csv <path> --output <new-directory>`.

## ThermBuild

1. Obtain the measured raw files from [Fraunhofer Fordatis](https://fordatis.fraunhofer.de/handle/fordatis/486?locale=en): `ThermBuild_measure_raw.zip` and `ThermBuild_measure_Temp_raw.zip`. Retain the dataset's CC BY-SA 4.0 attribution and share-alike terms.
2. Extract the BSE1 and BSE2 raw and room-temperature CSV files. The expected SHA-256 values are:
   * `age1_Use1001_setB0_masshea_size100_SoProt0_roo3_BSE1_vent1_155_rot000_wetReal_raw.csv`: `881740b3c8bf32ea9ef34c5555b042ffac778ad94fe7fe37be2c722fe3f675aa`
   * `age1_Use1001_setB0_masshea_size100_SoProt0_roo3_BSE1_vent1_155_rot000_wetReal_roomTemp.csv`: `3da56be46ce59756ff4621650d25a65ede3b98d29faaf48520360466cf8520d9`
   * `age1_Use1002_setB0_masshea_size100_SoProt0_roo3_BSE2_vent1_155_rot000_wetReal_raw.csv`: `171416313c19d9236bac748daaefc7eac10f8ede398031327777fc0835c356d9`
   * `age1_Use1002_setB0_masshea_size100_SoProt0_roo3_BSE2_vent1_155_rot000_wetReal_roomTemp.csv`: `786149c6f2cc86db91f50eef8e4e04af9121af4b7f186df4ea8e2e3ee96b35ea`
3. The E4 configurations refer to these files under `training/raw/thermbuild/`. The published E4 subset includes processed arrays. To prepare from raw files in a separate directory, see `training/prepare_inputs.py thermbuild --raw-dir <directory-containing-the-four-CSVs> --output <new-directory>`.

## QUANT sensor dataset

For the external data-quality reference, obtain `QUANT_SensorSystems_hourly.csv` from the [official Zenodo record](https://zenodo.org/records/10775692). The study file SHA-256 is `a8fa398aacd7bb5c5c320df6675a8dccb146d4e44d0c6f392b4d5acf76fc4d1f`. The record describes source attribution and the Open Government Licence v3.0 for the underlying sensor file.

Use `sha256sum <file>` or PowerShell `Get-FileHash -Algorithm SHA256 <file>` to check a downloaded file. The instructions identify intended inputs; preparation scripts and full results have not been validated on a fresh public installation.
