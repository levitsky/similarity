# Similarity

Predict mass spectra of peptides using [Koina](https://github.com/wilhelm-lab/koina) and score pairwise spectral similarity.

The project supports:

- Single-input mode: one peptide list/table scored against itself.
- Dual-input mode: one peptide list/table scored against another.
- Reuse of precomputed peptide tables and predicted spectra.
- Optional caching of model predictions.
- Two output styles: a rich TSV dataframe and a compact NumPy array.
- Subset processing for large jobs, including parallel execution on workstations or clusters.

## Installation

Install in your Python environment:

```bash
pip install git+https://github.com/levitsky/similarity.git
```

Main CLI entry points:

- `run_single` (alias: `run`)
- `run_dual`
- `time_scoring`

## Input Format

For `--input-file`, provide a plain text file with one peptide sequence per line.

## Start Here: Minimal Example

Run a complete single-input workflow from peptide list to TSV scores:

```bash
run_single \
	--input-file tests/test_peptides.txt \
	--output-file scores.tsv
```

This is the simplest starting point. The rest of this README adds performance-oriented options for larger runs.

## Single vs Dual Input Mode

### Single-input mode (`run_single`)

Scores one peptide table against itself.

```bash
run_single \
	--input-file peptides.txt \
	--output-file scores.tsv
```

### Dual-input mode (`run_dual`)

Scores peptides from input 1 against peptides from input 2.

```bash
run_dual \
	--input-file-1 peptides_A.txt \
	--input-file-2 peptides_B.txt \
	--output-file scores_A_vs_B.tsv
```

Dual mode uses suffixed file arguments (`-1`, `-2`) for input/load/save paths.

## Save and Reuse Intermediate Files

For repeated experiments, save peptide tables and spectra once, then reload them.

### Save peptide table and spectra

```bash
run_single \
	--input-file peptides.txt \
	--peptide-file peptides.tsv \
	--spectrum-file spectra.npy \
	--array-file scores.npy
```

### Load existing peptide table and spectra

```bash
run_single \
	--load-peptide-table peptides.tsv \
	--load-spectrum-file spectra.npy \
	--array-file scores_reused.npy
```

Dual mode has equivalent options with `-1` and `-2` suffixes:

- Save: `--peptide-file-1`, `--peptide-file-2`, `--spectrum-file-1`, `--spectrum-file-2`
- Load: `--load-peptide-table-1`, `--load-peptide-table-2`, `--load-spectrum-file-1`, `--load-spectrum-file-2`

## Caching

Caching stores model predictions. This allows reusing predicted values for specific peptides, even if your input file
is not identical to the previous run.

Set cache backend with `--cache`:

- `NONE` (default)
- `DISKCACHE`
- `REDIS`

Use `--cache-properties` if you also want to cache precursor-level properties. Otherwise, only MS2 spectrum predictions are cached.

### Diskcache example

```bash
run_single \
	--input-file peptides.txt \
	--cache DISKCACHE \
	--cache-dir .cache/similarity \
	--output-file scores.tsv
```
**Note**: you need to install `diskcache` package to use it.

### Redis cache example

```bash
run_single \
	--input-file peptides.txt \
	--cache REDIS \
	--host localhost \
	--port 6379 \
	--db 0 \
	--output-file scores.tsv
```
**Note**: you need to install `redis` package, as well as Redis itself, and manage the Redis service.


## Output Options: DataFrame vs Array

You can write one or both output formats:

- `--output-file`: TSV dataframe (human-readable, richer metadata)
- `--array-file`: raw NumPy structured array (compact and memory-efficient)

### DataFrame output (`--output-file`)

Best for inspection and downstream tabular analysis.
Includes peptide sequences, charge, m/z, iRT, and score columns.

### Array output (`--array-file`)

Best for very large runs. The saved `.npy` array stores only core records `(i, j, score)` and has lower overhead than a dataframe.
This compact representation allows larger result sets to fit in memory when loaded or concatenated for post-processing.

## Subset Processing for Large Datasets

Use subsets when a full run is too large for memory or when distributing work.

- `--subsets N`: total number of subsets.
- `--subset K`: which subset to run (`1..N`).

### Important behavior

- In single-input mode, the peptide table is split into overlapping m/z-safe ranges.
- In dual-input mode, only input/table 1 is split; input/table 2 is used in full for each subset.
  **It is advised to use the bigger input as input 1.**
- To run all subsets via the built-in runner (`--subset 0`), you must provide `--peptide-file` or `--load-peptide-table`.

### Step 1: Create reusable prerequisites once

```bash
run_single \
	--input-file peptides.txt \
	--peptide-file peptides.tsv \
	--spectrum-file spectra.npy
```

### Step 2a: Run subsets consecutively (shell loop)

```bash
N=8
for K in $(seq 1 "$N"); do
	run_single \
		--subsets "$N" \
		--subset "$K" \
		--load-peptide-table peptides.tsv \
		--load-spectrum-file spectra.npy \
		--array-file "scores_subset_${K}.npy"
done
```

### Step 2b: Run subsets in parallel with GNU parallel

```bash
N=8
seq 1 "$N" | parallel -j 8 '
	run_single \
		--subsets '"$N"' \
		--subset {} \
		--load-peptide-table peptides.tsv \
		--load-spectrum-file spectra.npy \
		--array-file scores_subset_{}.npy
'
```

### Step 2c: Run subsets on a SLURM cluster (job array)

Create `run_subset.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=similarity-subsets
#SBATCH --array=1-8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/subset_%A_%a.out

source similarity_env/bin/activate

run_single \
	--subsets 8 \
	--subset ${SLURM_ARRAY_TASK_ID} \
	--load-peptide-table peptides.tsv \
	--load-spectrum-file spectra.npy \
	--array-file scores_subset_${SLURM_ARRAY_TASK_ID}.npy
```

Submit:

```bash
sbatch run_subset.slurm
```

### Built-in sequential all-subsets runner

You can also ask `run_single` to process all subsets in one command by setting `--subset 0` with `--subsets > 1`.

When using `--array-file` or `--output-file` with this mode, include a `{}` placeholder so each subset writes to a unique file.

```bash
run_single \
	--input-file peptides.txt \
	--subsets 8 \
	--subset 0 \
	--peptide-file peptides.tsv \
	--spectrum-file spectra.npy \
	--array-file scores_subset_{}.npy
```

## Experiment Config Options (`Config` fields)

CLI flags map directly to `Config` fields in kebab-case, for example:

- `Config.precursor_mz_tolerance` -> `--precursor-mz-tolerance`
- `Config.model_intensity` -> `--model-intensity`

Below are the main experiment options and their defaults.

### Acquisition and precursor setup

- `collision_energy` (default `30`): collision energy passed to prediction models.
- `fragmentation_type` (default `HCD`): fragmentation regime (`HCD` or `CID`).
- `min_charge` / `max_charge` (default `2` / `2`): precursor charge-state range to generate and score.
- `min_length` / `max_length` (default `5` / `30`): peptide length limits.

### Prediction model selection

- `model_intensity` (default `Prosit_2025_intensity_40PTM`): MS2 intensity model.
- `model_irt` (default `Prosit_2025_irt_40PTM`): RT/iRT prediction model.
- `model_ccs` (default `None`): optional CCS model. If enabled, CCS is included in tolerance filtering.
- `koina_host` (default `koina.wilhelmlab.org:443`): Koina server endpoint.

### Pair candidate tolerances

- `precursor_mz_tolerance` (default `10.0`): precursor m/z tolerance.
- `precursor_mz_unit` (default `PPM`): unit for precursor m/z tolerance (`PPM` or `Th`).
- `isotope_error` (default `1`): allowed isotope offset during precursor matching.
- `irt_tolerance` (default `5.0`): absolute iRT tolerance.
- `ccs_rtolerance` (default `0.02`): relative CCS tolerance (used only when `model_ccs` is set).

### Peak matching and spectral preprocessing

- `fragment_mz_tolerance` (default `10.0`): fragment m/z matching tolerance.
- `fragment_mz_unit` (default `PPM`): unit for fragment tolerance (`PPM` or `Th`).
- `resolution` (default `30000`): instrument resolving power used when merging close predicted peaks.
- `mass_analyzer` (default `Orbitrap`): analyzer model for resolution scaling (`Orbitrap`, `TOF`, `FTICR`).
- `max_peaks` (default `50`): maximum number of peaks retained per predicted spectrum.

### Sequence chemistry and PTM controls

- `nonstandard_aminoacids` (default `False`): allow non-standard amino acids.
- `ptms` (default `False`): enable PTM-aware handling.
- `fixed_mods` (default `None`): fixed modification rules.
- `variable_mods` (default `None`): variable modification rules.

When using modifications, ensure model choice is compatible with your peptidoform space.

### Performance and execution behavior

- `workers` (default `cpu_count()`): number of worker processes used in grouping/scoring.
- `batch_size` (default `1000`): batch size used for grouping.
- `score_threshold` (default `0.0`): minimum similarity score to keep.
- `spectrum_collection` (default `SHAREDARRAY`): spectrum storage backend (`SHAREDARRAY` or `CACHED`).

Typical strategy: start with moderate `workers` and tune with your RAM and I/O limits.

### Subset controls

- `subsets` (default `1`): number of subsets to split input 1 into.
- `subset` (default `0`): subset index to run (`1..N`) or `0` for built-in all-subsets sequential mode.

For distributed workloads, prefer running explicit subset indices in parallel (GNU parallel or SLURM array jobs).

## Practical Tips

- For iterative analyses, save and reload peptide tables and spectra.
- Prefer `--array-file` for very large outputs; convert to dataframe only when needed.
- Keep `--batch-size`, `--workers`, and subset count tuned to your RAM and CPU availability.

The optimal value of `--workers` depends on the CPU and I/O speeds, but in practice it can be as low as 5-8. To use many cores efficiently, consider running multiple subsets in parallel, if RAM permits.