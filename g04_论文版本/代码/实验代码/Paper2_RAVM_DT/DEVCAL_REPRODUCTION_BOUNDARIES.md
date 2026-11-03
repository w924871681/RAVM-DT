# Development calibration reproduction

The accepted route combines the newly reproduced science with the original sealed
timing-dependent selection component. It does not claim that a fresh wall-clock
measurement reproduces the historical tie-breaking identity.

- New execution: 1,584 trajectories / 38,016 stages passed the original numerical
  and causal checks. Two original summaries (66 and 198 rows, 5,082 checked cells)
  passed the existing 1e-7 comparison; timing and storage fields retain the
  original comparison exclusions.
- New measured selection: MWU_MEAN__g01. The historical selection is
  MWU_MEAN__g04. Eight configurations have exactly equal primary MSE; the original
  second ordering field is measured maintenance time. Both observed choices are
  produced by the unchanged original rule on their respective measurements.
- Reused component: the original sealed selection, byte for byte. Its 38,016
  measured stage costs aggregate exactly to all 66 original totals. The original
  selector AST reproduces all discrete choice fields and GO checks exactly.
  Historical role-chain hashes bind selection before subsequent phase inputs.
  All 462 non-time selection inputs from the new run match the historical inputs
  exactly. New timings are never replaced with old timings in new results.
- The fresh-run exact-selection failure remains preserved. No trajectories,
  models, inference or accepted numerical comparisons were rerun for this repair.

Current evidence is registered in SUPERVISION_STATE.json under
`active_paths.paper2_devcal_compositional_acceptance`. The standalone archive must
include that directory's original selector, source and reference manifests,
historical measured costs, selection seal and role events, both selection locks,
new science/table receipts and explicit component provenance. Historical timing
reuse is not a new performance measurement. This acceptance does not constitute
whole-paper or independent ZIP acceptance.
