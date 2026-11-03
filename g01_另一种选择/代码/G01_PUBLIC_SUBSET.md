# g01 public non-E3 sensitivity subset

This directory provides the historical plan and the reviewed source/initial reuse materials for the synthetic and E4 groups. Alibaba E3 data, source models, initial models and E3 prepared group are not redistributed. The historical PLAN_TEMPLATE still documents the complete local plan; the original `运行g01.py` expects all 17 groups and the original full asset manifest, and is not runnable unchanged in this public subset. Its historical status fields are not a new run receipt.

To prepare a separate non-E3 plan from this version directory, run:

```text
<lifecycle-python> 代码/run_g01_non_e3.py --source-python <source-python> --output <new-plan-dir>
```

The wrapper checks the original scientific bindings, explicitly replaces only the historical full-manifest binding with the published 16-group subset manifest, and constructs 1,854 non-E3 ordinary tasks plus three separately timed cost tasks. Add `--run` to launch ordinary training; it is not started by default. The original `运行g01成本.py` accepts the generated plan for the three isolated cost tasks. A complete clean-machine run of this public subset has not been validated. The g01 selection and timings are a post-hoc sensitivity comparison, not a new blind Final or a reproduction of the paper's historical g04 timing selection. UCI/ThermBuild material retains the attribution and terms in the repository's DATA_THIRDPARTY_ATTRIBUTION.md.
