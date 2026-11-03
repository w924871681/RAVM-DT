# RAVM-DT: code, results and figures

Materials for manuscript CMC 92902. `g04_论文版本` follows the historical
selection used in the manuscript. `g01_另一种选择` presents a post-hoc
sensitivity comparison using the same fixed test inputs. When development
errors tied, measured maintenance time selected g04 historically and g01 in a
later run; the later timing choice is not an independent reproduction of g04.

## Contents and scope

Each version contains Python source code (`代码`), figure PDFs (`图`), and
author-generated aggregate tables and plotting inputs (`输出结果`). The figure
scripts generate Figures 3–8 from saved inputs; Figures 1 and 2 are artwork.

This is a **partial release**. Configuration JSON, datasets, model weights and
checkpoints are not included. Consequently, the original frozen experiment
runners are not currently runnable end-to-end. The alternative-machine E3
candidate is supplied separately and has not been validated end to end. The historical experiment
results and comparisons do not demonstrate that this public snapshot has been
fully reproduced in a clean environment.

The historical g04 matrix has 17,118 completed trajectories with scientific
comparisons. Development calibration used an evidence-supported combination
that reused the historical timing-selection component. The g01 sensitivity
analysis compares 2,094 ordinary trajectories and three cost repetitions.

For data access, model-generation dependencies and the remaining gaps, see
[Reproduction inputs and current limitations](REPRODUCIBILITY_STATUS.md).
An [alternative-machine E3 candidate](alibaba_flexible/README.md) provides
selection, training and evaluation entry points, but it has not been validated
end to end on the official dataset.

## Raw data access

Raw datasets are obtained separately from their respective providers and are
not distributed here. Follow each provider's current access and attribution
terms.

* [Alibaba Cluster Trace 2018](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md):
  obtain `machine_usage.tar.gz`. Its documented SHA-256 is
  `3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a`.
* [QUANT sensor data](https://zenodo.org/records/10775692): obtain
  `QUANT_SensorSystems_hourly.csv`. The study file has SHA-256
  `a8fa398aacd7bb5c5c320df6675a8dccb146d4e44d0c6f392b4d5acf76fc4d1f`.
* [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
  and [ThermBuild](https://doi.org/10.24406/fordatis/445) retain the provider's
  attribution and licence conditions.

The Alibaba-derived inputs and model checkpoints are not cleared for public
redistribution, and a complete public reconstruction workflow is not yet
available. `LICENSE-vendor.txt` retains the MIT notice from the original vendor
code; it does not license this entire mixed repository or third-party data.
