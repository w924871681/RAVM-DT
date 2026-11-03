# Explicit reader PATH_B commands

Use the two recorded stacks in ENVIRONMENT_LOCK.json. Supply your local interpreter paths to $SourcePython and $MainPython; no author directory is needed. Both optimization commands below are for a reader's future run, NOT for the submission packaging checks. Choose fresh output paths outside this immutable bundle.

```powershell
$SourcePython = 'PATH_TO_SOURCE_PYTHON'
$MainPython = 'PATH_TO_MAIN_PYTHON'
& $MainPython run.py verify
& $SourcePython source_training.py --domain synthetic --architecture GRU --horizon 1 --seed 7311 --output ../READER_OUTPUT/source_fit
& $MainPython train_from_source.py --group Main --center 63000 --seed 7311 --stream mixed --arm JOINT__l3__m0 --source-checkpoint ../READER_OUTPUT/source_fit/checkpoints/source_GRU_H1_7311.pt --source-receipt ../READER_OUTPUT/source_fit/SOURCE_TRAINING_RECEIPT.json --output ../READER_OUTPUT/main_path_b
```

The matching acceptance receipt covers this synthetic case, not other domains, grids or all hardware. Prefix arrays, initial models and core training/gate functions retain accepted identities. Remote clean-environment installation has not been newly validated.
