# Copilot Instructions for similarity

## Project intent
- This repository predicts peptide spectra and computes pairwise spectral similarity scores.
- Keep changes focused on scientific correctness first, then throughput and memory behavior.

## Architecture snapshot
- Main flow: CLI -> `Config` parsing -> `Experiment` fixtures -> prediction/grouping -> array/dataframe output.
- Core modules:
  - `similarity/prediction.py`: peptide table generation/loading, subset offsets, shared-memory arrays, Koina prediction.
  - `similarity/grouping.py`: candidate pair generation via cKDTree and pair scoring via C extension.
  - `similarity/experiment.py`: fixture orchestration and resource cleanup.
  - `similarity/cli.py`: command-line wiring and subset runner behavior.

## Non-negotiable correctness invariants
- Subset processing must be equivalent to a full single-run result after concatenating all subsets.
- `subset_offsets` must produce exactly `config.subsets` ranges and must not drop tail rows.
- Overlap handling between subsets must avoid both missed boundary pairs and duplicate pairs.
- In grouping, boundary inclusion must keep the first non-overlap index in scope.
- `score_array` records are structured as `(i, j, score)` with consistent global peptide indices.

## Data and dtype conventions
- Internal peptide sequences are bytes in runtime arrays/dataframes; decode to ASCII only for user-facing serialization.
- Keep `m/z`, `irt`, and `ccs` numeric arrays in shared memory as float32 where currently used.
- Preserve structured dtype for scores: `[("i", int32), ("j", int32), ("score", float32)]`.

## Performance and implementation guidance
- Prefer NumPy vectorization and existing C-backed routines over Python loops on hot paths.
- Do not replace `similarity._match_peaks` usage unless explicitly requested.
- Treat multiprocessing queue behavior, worker lifecycle, and shared-memory cleanup as sensitive.
- Any change in batching, offsets, or tolerance filtering should include attention to duplicate and missed-pair risk.

## Logging and diagnostics
- Use module-level `logger = logging.getLogger(__name__)`.
- Keep info logs concise for progress; use debug logs for detailed internals.
- Preserve useful context in errors, especially for subset boundary calculations and cache/prediction paths.

## Configuration and CLI expectations
- Keep `Config` dataclass and CLI argument mapping aligned.
- If adding config fields, update argparse integration in `BaseConfig` behavior.
- Do not silently change semantics of `--subsets`, `--subset`, `--batch-size`, cache options, or score thresholds.

## Testing requirements for code changes
- For peak matching or score math changes, run at least:
  - `tests/test_match_peaks.py`
- For subset/grouping/prediction changes, run at least:
  - `tests/test_experiment.py` (especially `SubsetTest` cases)
- If behavior changes intentionally, update or add targeted regression tests rather than broad rewrites.
- For test runs, use the configured Python interpreter. Check that the system Python is not used.

## Change style
- Make minimal, localized edits.
- Preserve public APIs and file formats unless the task requires a breaking change.
- Avoid introducing new dependencies unless clearly justified.
- Add short comments only for non-obvious logic, especially around subset overlap boundaries.
