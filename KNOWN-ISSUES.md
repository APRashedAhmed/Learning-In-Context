# Known issues

## Checkpoint loading can silently use the wrong architecture (2026-08-21)

Found during the molab delegation-contract capture (see
`heliopolis/PerAnkh/projects/laboratory/plans/2026-08-21-molab-extraction-delegation-contract.md`);
deposited here per operator direction for repo-side handling.

- `get_model_config_for_id()`
  (`src/learning_in_context/config/model_config.py:285-303`) covers only 5
  hardcoded model IDs (`TIMESCALES_MODEL_CONFIGS`, lines 256–282: SAN-4378,
  4401, 4566, 4567, 4568) out of the ~653 checkpoints on disk. Any other
  model ID silently receives `ModelConfig()` defaults (`recurrent_size=16`,
  `recurrent_num_layers=1`) — wrong for e.g. the 1024-unit models.
- `load_model_from_checkpoint` prefers
  `checkpoint['hyper_parameters']['model_config']` (lines 64–77) and falls
  back to the ID table only when that key is absent — but whether checkpoints
  in general carry a usable `hyper_parameters.model_config` is unverified.
- If the strict `load_state_dict` fails, it retries with `strict=False`
  (lines 98–104), so a partially-wrong model can load without any error.

Net effect: an extraction sweep can "succeed" while running some models with
the wrong architecture. Suggested direction: make architecture params an
explicit, verified input of each run (or validate loaded architecture against
the checkpoint's tensor shapes and fail loudly), and drop the `strict=False`
fallback.

## Cached ideal-observer predictions are not reproducible (2026-08-31)

`data/cache/model_states/participant_dataset/ibo/ibo-01.npz` cannot be
regenerated from any observer implementation in this repo, so the panels it
feeds — figure 2's `cwc_hazard_rate` and `cwc_contingency` — rest on an
artifact of unverified provenance.

- No task in `dodo.py`, and no script under `scripts/`, produces the file:
  it entered the cache from outside the pipeline (mtime 2025-08-01).
- Its `counts` keys — `alpha_hz`, `beta_hz`, `alpha_cont`, `beta_cont` — are
  `IdealCountingObserverV2`'s beta-prior names, so it was written by that
  observer's lineage. But re-running the committed `IdealCountingObserverV2`
  over `participant_dataset/samples.npy` reproduces neither half of the file:
  the stored predictions differ from the recomputed beliefs by up to 1.0
  (only a quarter of entries agree to 1e-4), and the stored counts are large
  integers (1, 419, 829 for the first three trials) where the committed
  observer accumulates fractional expected counts (2.5, 6.5, 11.5). The cache
  was therefore written by a lineage of the observer that the committed code
  no longer implements; which revision produced it is unestablished.
- The symptom is visible in the panel: mean CWC per hazard-rate condition is
  non-monotone in grayzone position (Low reads roughly -0.98, -0.70, -0.91
  across positions 0, 1, 2). A longer occlusion gives the colour more
  opportunity to change, so the observer's confidence in a "stay" response
  must not *recover* at the longest position. The contingency panel, which
  collapses over grayzone position, looks unaffected.

Pending re-verification once the px1 empirical run regenerates the observer
predictions. Until then treat figure 2's hazard CWC panel as provisional: its
grammar (conditions, ordering, colour ramp, mean traces) is settled, but the
position-2 values are not.
