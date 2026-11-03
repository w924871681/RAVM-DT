# Recorded environments and installation entry

The historical source-fitting environment was Windows/Python 3.11.15, PyTorch 2.5.1+cu121 and NumPy 2.4.6. The lifecycle environment was Windows/Python 3.13.5, PyTorch 2.13.0+cu126, NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, cryptography 44.0.1 and PyYAML 6.0.2. These are recorded identities, not a guarantee of cross-device numerical identity. The source setup also installs pandas 2.3.3 and SciPy 1.17.1; the lifecycle setup includes matplotlib 3.10.0 and psutil 7.2.2.

PyTorch's official version and index instructions are at https://pytorch.org/get-started/previous-versions/. On Windows, after installing both exact Python interpreters and an appropriate NVIDIA driver, the author's `setup_environments.py` in either version's `代码/实验代码/Paper2_RAVM_DT` directory creates fresh environments when explicitly invoked:

```text
<python> setup_environments.py --python311 <source-python-3.11.15> --python313 <lifecycle-python-3.13.5> --output <new-environment-directory>
```

The command requires internet access and substantial disk space. Do not install into an existing environment. Its complete fresh-install path was not re-executed or independently validated for this release; network, platform and hardware limitations may require adjustment and must be reported. The g01 non-E3 preparation entry is described in `g01_另一种选择/代码/G01_PUBLIC_SUBSET.md`. Use the official data links and dataset licences in DATA_THIRDPARTY_ATTRIBUTION.md and REPRODUCIBILITY_STATUS.md.
