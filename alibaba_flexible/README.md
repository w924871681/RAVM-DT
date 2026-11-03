# Preparing an alternative Alibaba machine split

This is a **new, unvalidated end-to-end candidate** for evaluating E3 methods
on an alternative machine split. It contains separate entry points for input
preparation, source fitting, initialization and lifecycle evaluation. It does
not replace the original frozen runner or recover the original paper numbers.
Only static checks and small mocked or synthetic interface tests have been
performed; the four commands have not been run together on the official data.
Their practical success and scientific equivalence are not yet established.

## Data and setup

Obtain `machine_usage.tar.gz` directly via the [Alibaba Cluster Trace 2018
instructions](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)
and follow the provider's terms. The program checks the archive against the
provider's registered SHA-256. Use Python with NumPy and pandas installed;
these dependencies were used for a small synthetic input check, not validated
as a full fresh-environment installation.

```bash
python select_and_prepare.py --archive /path/to/machine_usage.tar.gz --output /path/to/new_output --seed 20261104
```

Choose an output directory that does not already exist. This may read the
large archive repeatedly and use considerable time, memory and storage; it
does not extract the archive or run any training. The optional `--batch-size`
and `--chunk-size` options trade memory for additional reads.

## Selection and output

The script ranks machine identifiers using SHA-256 of `<seed>:<machine-id>`;
it scans candidates in that order, keeping those with at least 16,760 valid
observations. It assigns 20 source, 20 development, 20 calibration and 40
evaluation machines to disjoint pools. The preprocessing follows the existing
E3 recipe: deduplicate timestamps, sort, retain at most 17,000 valid CPU
observations, normalize the known percentage fields and mask missing values
and the undocumented-scale `mkpi` field. It writes `data/center_<id>.npz`
arrays and a local `selection.json` with the selected machine-ID hashes,
pool assignments, seed and source archive hash.

This selection differs from the historical protocol: it does **not** contain
or automatically exclude previously exposed machines. To exclude a known set,
the reader may supply a local UTF-8 text file with one SHA-256 machine-ID hash
per line via `--exclude-hashes /path/to/hashes.txt`. No historical exclusion
list is distributed here. Unless the exclusion list is complete and verified,
the new split must not be described as a blind holdout. Different seeds can
select different machines, and there is no guarantee a new selection is
disjoint from the paper's machines.

The paper's original pool was selected after excluding earlier exposed machine
hashes, using the documented seed 7311 and splitting 20/20/20/40. Its reported
40 evaluation machines later became exposed in prior work. A new split must be
reported separately; the paper's exact figure and table values are not expected
to be reproduced by choosing different machines.

## Candidate source fitting and evaluation

The original records describe two Python environments: source fitting and
warm-up used Python 3.11.15 / PyTorch 2.5.1+cu121; the lifecycle used Python
3.13.5 / PyTorch 2.13.0+cu126. The relevant Python environments also need the
NumPy, pandas, SciPy and cryptography dependencies used by the existing source
files. These version records are not a verified fresh-install recipe. Hardware,
driver and numerical differences can affect model states and timings.

With those environments separately prepared, a reader may attempt the following
from this directory. Replace the placeholders; each output directory must be a
new path. These commands actually perform training and can take substantial
time and storage. They have **not** been executed as part of this release.

```text
<source-python> fit_source_models.py --selection <new_output> --output <sources>
<source-python> prepare_initial_models.py --selection <new_output> --sources <sources> --output <initials>
<lifecycle-python> run_new_machines.py --selection <new_output> --sources <sources> --initials <initials> --output <evaluation>
```

The default requests three seeds, both clean and mixed streams and all twelve
paper E3 methods across 40 evaluation machines. Optional repeated `--center`,
`--seed`, `--stream` and `--arm` flags restrict the scope, and the execution
status is then marked `PARTIAL`. Generated weights and records belong to the
new selection; they are saved locally. The evaluation attempts to audit each
completed trajectory and writes `per_machine.csv`, `method_stream.csv`, and,
when the paired methods and both streams are present for at least two machines,
an exploratory paired bootstrap summary. It uses the paper's fixed g04 method
settings; it does not independently repeat the
historical development-calibration selection.

Experiment output can contain generated keys, machine hashes, checkpoints and
other data-derived files. Treat that output according to the providers' terms
and inspect it before considering any redistribution.

Do not describe a successful command launch, input preparation, mocked test or
partial run as full reproduction. Complete operation on official data,
end-to-end scientific validation, summary/figure correspondence and results
on genuinely unseen machines remain to be established independently.

For step-by-step reader commands and read-only checks, see [READER_WALKTHROUGH.md](READER_WALKTHROUGH.md).
