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

This is a **partial release**. Inputs, configurations and source and initial
checkpoints are supplied for Main, E1, E2, E4 and E5 in both versions; see
[subset scope](SYNTHETIC_RELEASE_SCOPE.md) and [data attribution and licences](DATA_THIRDPARTY_ATTRIBUTION.md).
The new manifest covers these groups only. Development-calibration files and
historical references are also provided; see [their scope](DEVCAL_RELEASE_SCOPE.md).
Saved trajectory references for the 16 non-E3 groups are provided; see [reference scope](HISTORICAL_REFERENCE_SCOPE.md). Original data archives and Alibaba E3 historical assets are not included. No full clean environment or source-to-final execution has been validated for this release.
The alternative-machine E3 candidate has not been validated end to end.

The [g01 non-E3 reuse subset](g01_另一种选择/代码/G01_PUBLIC_SUBSET.md) includes source and initialization assets and a scoped preparation entry. This entry has not been validated by a complete public rerun.



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
