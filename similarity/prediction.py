from typing import TYPE_CHECKING, Any, cast
from collections.abc import Sequence, Callable
import pandas as pd
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from koinapy import Koina
from ._match_peaks import merge_close_peaks_sorted
from .utils.abc import Fixture, IndexType
from .utils.config import PROTON_MASS, MassAnalyzerType
from .utils.cache.common import SpectrumCache
from pyteomics import cmass, auxiliary as aux, proforma, parser
import logging
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

if TYPE_CHECKING:
    from pathlib import Path
    import numpy as np
    from numpy.typing import DTypeLike
    from .experiment import Experiment
    from .utils.abc import Cache, SpectrumCollection
    from .utils.config import Config

logger = logging.getLogger(__name__)


class PredictedSpectrumCollection(Fixture):
    batch_size: int = 10000

    @staticmethod
    def _merge_close_peaks(
        mz: "np.ndarray", intensities: "np.ndarray", config: "Config"
    ) -> int:
        return merge_close_peaks_sorted(
            mz,
            intensities,
            float(config.resolution),
            config.mass_analyzer.value,
        )

    @staticmethod
    def preprocess_predictions(
        result: dict[str, list["np.ndarray"]], config: "Config"
    ) -> dict[str, list["np.ndarray"]]:
        for mz, intensities in zip(result["mz"], result["intensities"]):
            order = np.argsort(mz, kind="mergesort")
            mz[:] = mz[order]
            intensities[:] = intensities[order]
            npeaks = PredictedSpectrumCollection._merge_close_peaks(
                mz, intensities, config
            )

            np.sqrt(
                intensities[:npeaks],
                out=intensities[:npeaks],
                where=intensities[:npeaks] > 0,
            )
            if npeaks < mz.size:
                mz[npeaks:] = -1.0
                intensities[npeaks:] = -1.0
        return result

    def evaluate(self, experiment: "Experiment") -> "SpectrumCollection":
        index = cast("SpectrumCache | None", experiment.cache[IndexType.INTENSITY])
        collection = experiment.config.spectrum_collection.value(
            experiment, self.suffix
        )
        if index is not None:
            collection.fill_from_cache(index)
        cached = collection.spectra_available
        if cached.all():
            logger.info("All spectra are cached, skipping prediction")
            return collection
        prediction_inputs = collection.peptides.loc[~cached]
        model = Koina(
            experiment.config.model_intensity.name, experiment.config.koina_host
        )

        bsize = self.batch_size
        nbatches = (prediction_inputs.shape[0] + bsize - 1) // bsize
        logger.info(
            "Predicting spectra for %d peptides in %d batches of size %d...",
            prediction_inputs.shape[0],
            nbatches,
            bsize,
        )
        with logging_redirect_tqdm():
            for i in trange(
                nbatches, desc=experiment.config.model_intensity.name, unit="batch"
            ):
                logger.debug("Predicting batch %d of %d ...", i + 1, nbatches)
                batch_inputs = prediction_inputs.iloc[i * bsize : (i + 1) * bsize]
                result: dict[str, list["np.ndarray"]] = model.predict(batch_inputs, df_output=False, mode="async", disable_progress_bar=True)  # type: ignore
                result = self.preprocess_predictions(result, experiment.config)

                collection.fill_from_predictions(batch_inputs, result)
                if index is not None:
                    index.save_predictions(batch_inputs, result)

        if index is not None:
            index.finalize()
        assert collection.is_ready()
        return collection


class Offsets(Fixture):
    def __set__(self, instance: "Experiment", value: Sequence[tuple[int, int]]):  # type: ignore
        self._data[instance] = value

    def evaluate(self, experiment: "Experiment") -> Any:
        raise RuntimeError(
            "Offsets are not supposed to be evaluated directly, they are set by the MzIrtDataFrame when calculating subset offsets."
        )


class MzIrtDataFrame(Fixture):
    _shared_memory: dict["Experiment", dict[str, SharedMemory]]
    batch_size: int = 50000

    def __init__(self):
        super().__init__()
        self._shared_memory = {}

    def close(self, experiment: "Experiment"):
        shm_dict = self._shared_memory.pop(experiment, {})
        for name, shm in shm_dict.items():
            logger.debug(
                "Closing %s shared memory for experiment %d, name %s",
                self.name,
                id(experiment),
                name,
            )
            shm.close()
            shm.unlink()

    def get_predictions(
        self,
        name: str,
        index: "Cache | None",
        inputs: pd.DataFrame,
        output: np.ndarray,
        experiment: "Experiment",
    ):
        logger.debug(
            "Getting %s predictions. Target array shape: %s", name, output.shape
        )
        if index is None:
            output[:] = np.nan
            ncached = 0
            mask = np.ones_like(output, dtype=bool)
            logger.info(
                "Skipping index lookup for %s prediction, no cache configured", name
            )
        else:
            index.fill_from_cache(inputs, output)
            mask = np.isnan(output)
            ncached = (~mask).sum()
            logger.info(
                "%d of %d peptides are cached for %s prediction",
                ncached,
                inputs.shape[0],
                name,
            )

        if ncached < inputs.shape[0]:
            logger.info(
                "Predicting %s for %d peptides...", name, inputs.shape[0] - ncached
            )
            model = Koina(
                getattr(experiment.config, f"model_{name}").name,
                experiment.config.koina_host,
            )
            masked = inputs.loc[mask]
            bsize = self.batch_size
            idx = np.where(mask)[0]
            nbatches = (idx.shape[0] + bsize - 1) // bsize
            logger.info(
                "Predicting %s in %d batches of size %d...",
                name,
                nbatches,
                bsize,
            )
            with logging_redirect_tqdm():
                for i in trange(
                    nbatches,
                    desc=f"{getattr(experiment.config, f'model_{name}').name}",
                    unit="batch",
                ):
                    batch_idx = idx[i * bsize : (i + 1) * bsize]
                    batch_masked = masked.loc[batch_idx]
                    result = model.predict(
                        batch_masked,
                        df_output=False,
                        mode="async",
                        disable_progress_bar=True,
                    )
                    output[batch_idx] = result[name].reshape(-1)
                    if index is not None:
                        index.save_predictions(batch_masked, result)  # type: ignore

            if np.isnan(output).any():
                logger.error(
                    "Some %s values are still missing after prediction. Output:\n%s",
                    name,
                    output,
                )
            if index is not None:
                index.finalize()
        else:
            logger.info("All %s values are cached, skipping prediction", name)

    def shared_array(
        self,
        experiment: "Experiment",
        name: str,
        shape: tuple[int, ...],
        dtype: "DTypeLike",
    ) -> np.ndarray:
        self._shared_memory.setdefault(experiment, {})
        shm_dict = self._shared_memory[experiment]
        if name in shm_dict:
            shm = shm_dict[name]
        else:
            size = np.prod(shape, dtype=int) * np.dtype(dtype).itemsize
            logger.info(
                "Allocating %.2f MB of shared memory for %s with dtype %s",
                size / 2**20,
                name,
                dtype,
            )
            shm = SharedMemory(
                name=f"{name}{self.suffix}-{id(experiment)}",
                create=True,
                size=size,
            )
            shm_dict[name] = shm
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def load_peptide_table(
        self, fpath: "str | Path", experiment: "Experiment"
    ) -> pd.DataFrame:
        """Load peptide table from previously saved file into shared memory."""
        df = pd.read_csv(fpath, sep="\t")
        required_columns = {"peptide_sequences", "precursor_charges", "irt", "m/z"}
        if experiment.config.model_ccs is not None:
            required_columns.add("ccs")
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in peptide table {fpath}: {sorted(missing)}"
            )

        if df["peptide_sequences"].isna().any():
            raise ValueError("Column peptide_sequences contains missing values")

        if self.is_first and experiment.config.subsets > 1:
            logger.info(
                "Subset processing mode. Assuming peptide table is sorted by m/z and contains the complete peptide set."
            )
            offsets = self.subset_offsets(
                experiment, cast(np.ndarray, df["m/z"].values)
            )

            start, end = offsets[experiment.config.subset - 1]
            logger.info(
                "Loading subset %d (rows %d to %d) from peptide table %s with m/z range %.4f - %.4f",
                experiment.config.subset,
                start + 1,
                end,
                fpath,
                df["m/z"].iloc[start],
                df["m/z"].iloc[end - 1],
            )
            df = df.iloc[start:end]

        seq = np.char.encode(
            df["peptide_sequences"].to_numpy(dtype=str, copy=False), "ascii"
        )
        seq_array = self.shared_array(
            experiment,
            "peptide_sequences",
            shape=(df.shape[0],),
            dtype=seq.dtype,
        )
        seq_array[:] = seq
        df["peptide_sequences"] = seq_array

        charge_array = self.shared_array(
            experiment,
            "precursor_charges",
            shape=(df.shape[0],),
            dtype=np.uint8,
        )
        charge_array[:] = df["precursor_charges"].to_numpy(dtype=np.uint8, copy=False)
        df["precursor_charges"] = charge_array

        mzrt_shape = (df.shape[0], 3 if "ccs" in df.columns else 2)
        mzrt = self.shared_array(experiment, "mzrt", shape=mzrt_shape, dtype=np.float32)
        mzrt[:, 0] = df["m/z"].to_numpy(copy=False)
        mzrt[:, 1] = df["irt"].to_numpy(copy=False)
        df["m/z"] = mzrt[:, 0]
        df["irt"] = mzrt[:, 1]

        if "ccs" in df.columns:
            mzrt[:, 2] = df["ccs"].to_numpy(copy=False)
            df["ccs"] = mzrt[:, 2]

        return df

    def subset_offsets(
        self, experiment: "Experiment", mz_array: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Returns the offsets for each subset. When the whole dataset is too big to be processed at once,
        Experiment Config can have subsets > 1, which will split the dataset into that many subsets.
        This function calculates the offsets for each subset.
        """
        c = experiment.config
        values = mz_array
        dim = "m/z"
        nvalues = len(values)
        if c.subsets > nvalues:
            raise ValueError(
                f"Number of subsets ({c.subsets}) cannot exceed number of peptides ({nvalues})"
            )

        bsize = nvalues // c.subsets
        max_overlap = max(1, bsize // 5)
        # Nominal boundaries without overlap: exactly c.subsets chunks, full coverage.
        ends = [((k + 1) * nvalues) // c.subsets for k in range(c.subsets)]
        offsets: list[tuple[int, int]] = []
        previous_start = 0
        for k, end in enumerate(ends):
            if k == 0:
                start = 0
            else:
                end_of_previous = ends[k - 1]
                start = end_of_previous

                if not self.suffix:  # only apply overlap logic in single-input mode
                    while (
                        values[end_of_previous] - values[start - 1]
                        <= c.absolute_mz_error(values[end_of_previous])
                        + c.isotope_error * PROTON_MASS / c.min_charge
                    ):
                        start -= 1
                        if (
                            start <= previous_start
                            or end_of_previous - start >= max_overlap
                        ):
                            logger.error(
                                "Subset size is too small to accommodate the %s tolerance. "
                                "Please decrease the number of subsets.",
                                dim,
                            )
                            raise ValueError("Subset size is too small")
            offsets.append((start, end))
            previous_start = start

        logger.debug("Calculated subset offsets: %s ... %s", offsets[:3], offsets[-3:])
        experiment.offsets = offsets
        return offsets

    def has_ptms(self, experiment: "Experiment") -> bool:
        return bool(
            experiment.config.ptms
            or experiment.config.variable_mods
            or experiment.config.fixed_mods
        )

    def generate_sequences(self, experiment: "Experiment") -> np.ndarray:
        attr = self.get(experiment, "input_file")
        input_file = cast("str | Path", attr)
        logger.debug("Generating peptide sequences from %s as %s", input_file, attr)
        seq = np.unique(np.loadtxt(input_file, dtype=bytes))

        logger.info("Loaded %d unique peptide sequences from %s", len(seq), input_file)
        if experiment.config.ptms:
            length = lambda s: len(proforma.ProForma.parse(s.decode("ascii")))
        else:
            length = len
        len_in_bounds = np.array(
            [
                experiment.config.min_length
                <= length(s)
                <= experiment.config.max_length
                for s in seq
            ],
            dtype=bool,
        )
        n_out_of_bounds = (~len_in_bounds).sum()
        if n_out_of_bounds > 0:
            logger.warning(
                "%d peptide sequences are out of length bounds [%d, %d] and will be skipped",
                n_out_of_bounds,
                experiment.config.min_length,
                experiment.config.max_length,
            )
            logger.debug(
                "Out of bounds sequences:\n%s",
                "\n".join(
                    list(seq[~len_in_bounds][:10].astype(str))
                    + (["..."] if n_out_of_bounds > 10 else [])
                ),
            )
        seq = seq[len_in_bounds]

        ptms = self.has_ptms(experiment)
        std_amino_acids = set(parser.std_amino_acids)
        if experiment.config.variable_mods or experiment.config.fixed_mods:
            logger.info(
                "Expanding peptide sequences with variable and fixed modifications"
            )
            variable_rules = [
                proforma.TagParser(item)()[0]
                for item in (experiment.config.variable_mods or [])
            ]
            fixed_rules = [
                proforma.TagParser(item)()[0]
                for item in (experiment.config.fixed_mods or [])
            ]
            logger.debug(
                "Variable modifications %s parsed as %s",
                experiment.config.variable_mods,
                variable_rules,
            )
            logger.debug(
                "Fixed modifications %s parsed as %s",
                experiment.config.fixed_mods,
                fixed_rules,
            )
            expanded = []
            skipped = 0
            for peptide in seq:
                peptide = peptide.decode("ascii")
                try:
                    base = proforma.ProForma.parse(peptide)
                except aux.PyteomicsError as e:
                    logger.error(
                        "Failed to parse peptide sequence %s: %s", peptide, str(e)
                    )
                    continue

                if not experiment.config.nonstandard_aminoacids and any(
                    aa not in std_amino_acids for aa, _ in base.sequence
                ):
                    logger.debug(
                        "Peptide sequence %s contains non-standard amino acids and will be skipped",
                        peptide,
                    )
                    skipped += 1
                    continue
                for peptidoform in proforma.proteoforms(
                    base,
                    fixed_modifications=fixed_rules,
                    variable_modifications=variable_rules,
                    include_unmodified=True,
                    expand_rules=True,
                    strip=True,
                ):
                    sequence = str(peptidoform)
                    # logger.debug("Expanded peptide %s to %s", peptide, sequence)
                    expanded.append(sequence.encode("ascii"))
            seq = np.array(expanded, dtype=bytes)
            logger.info(
                "Expanded to %d peptide sequences after applying modifications",
                len(seq),
            )
            if skipped:
                logger.warning(
                    "%d peptide sequences were skipped due to non-standard amino acids",
                    skipped,
                )

        elif not experiment.config.nonstandard_aminoacids and experiment.config.ptms:
            skipped = np.zeros(len(seq), dtype=bool)
            for i, peptide in enumerate(seq):
                peptide = peptide.decode("ascii")
                try:
                    parsed = proforma.ProForma.parse(peptide)
                except aux.PyteomicsError as e:
                    logger.error(
                        "Failed to parse peptide sequence %s: %s", peptide, str(e)
                    )
                    continue

                if any(aa not in std_amino_acids for aa, _ in parsed.sequence):
                    logger.debug(
                        "Peptide sequence %s contains non-standard amino acids and will be skipped",
                        peptide,
                    )
                    skipped[i] = True
            if skipped.any():
                logger.warning(
                    "%d peptide sequences were skipped due to non-standard amino acids",
                    skipped.sum(),
                )
                seq = seq[~skipped]

        elif not experiment.config.nonstandard_aminoacids and not ptms:
            common_aa = set(map(lambda s: bytes(s, "ascii")[0], std_amino_acids))
            unsupported = np.array([bool(set(s) - common_aa) for s in seq])
            logger.debug("Unsupported mask:\n%s", unsupported[:10])
            if unsupported.any():
                logger.warning(
                    "Found %d unsupported peptide sequences, they will be skipped",
                    unsupported.sum(),
                )
                logger.debug(
                    "Unsupported sequences:\n%s",
                    "\n".join(
                        list(seq[unsupported][:10].astype(str))
                        + (["..."] if unsupported.sum() > 10 else [])
                    ),
                )
            seq = seq[~unsupported]
        return seq

    def mz_calculator(self, experiment: "Experiment") -> Callable[[bytes, int], float]:
        if self.has_ptms(experiment):
            return lambda peptide, charge: proforma.ProForma.parse(
                peptide.decode("ascii")
            ).mz(charge=charge)
        else:
            return lambda peptide, charge: cmass.fast_mass(
                peptide.decode("ascii"), charge=charge
            )

    def mass_calculator(self, experiment: "Experiment") -> Callable[[bytes], float]:
        if self.has_ptms(experiment):
            return lambda peptide: proforma.ProForma.parse(peptide.decode("ascii")).mass
        else:
            return lambda peptide: cmass.fast_mass(peptide.decode("ascii"))

    def mz_array(
        self,
        experiment: "Experiment",
        peptides: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate m/z values for all peptides and charge states.
        If multiple charges are configured, the output will be longer than the input.
        """
        logger.info("Calculating m/z values for all peptides and charge states...")
        if out is None:
            out = np.empty(
                len(peptides)
                * (experiment.config.max_charge - experiment.config.min_charge + 1),
                dtype=np.float32,
            )
        if experiment.config.min_charge == experiment.config.max_charge:
            mz_calc = self.mz_calculator(experiment)
            npeptides = len(peptides)
            out[:] = [
                mz_calc(peptide, experiment.config.min_charge) for peptide in peptides
            ]
        else:
            mass_calc = self.mass_calculator(experiment)
            npeptides = len(peptides)
            mass_array = np.array(
                [mass_calc(peptide) for peptide in peptides], dtype=np.float32
            )

            for i, charge in enumerate(
                range(experiment.config.min_charge, experiment.config.max_charge + 1)
            ):
                out[i * npeptides : (i + 1) * npeptides] = (
                    mass_array + charge * PROTON_MASS
                ) / charge
        logger.debug(
            "%d m/z values calculated, samples: %s and %s", len(out), out[:5], out[-5:]
        )
        return out

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        if experiment.config.subsets > 1 and experiment.config.subset == 0:
            logger.error(
                "Subset number is not set. Please set subset to a value between 1 and %d.",
                experiment.config.subsets,
            )
            raise ValueError("Subset number is not set")
        peptide_table = self.get(experiment, "peptide_table")
        if peptide_table is not None:
            logger.info("Loading peptide table from %s", peptide_table)
            return self.load_peptide_table(peptide_table, experiment)

        seq = self.generate_sequences(experiment)
        if seq.size == 0:
            logger.error("No valid peptide sequences found in input file, aborting")
            return pd.DataFrame()

        ncharges = experiment.config.max_charge - experiment.config.min_charge + 1
        npeptides = len(seq)
        nprecursors = npeptides * ncharges
        logger.info("%d total precursors to analyze", nprecursors)

        # calculate m/z values for all peptides and charge states
        # use them to figure out subsets
        mz_array = self.mz_array(experiment, seq)
        sort_idx = np.argsort(mz_array)
        mz_array = mz_array[sort_idx]

        # if processing the first (or only) input, apply the offset logic
        if self.is_first and experiment.config.subsets > 1:
            offsets = self.subset_offsets(experiment, mz_array)
            logger.debug(
                "Configured %d subsets with offsets %s",
                experiment.config.subsets,
                offsets,
            )
            idx_start, idx_end = offsets[experiment.config.subset - 1]
            logger.info(
                "Processing subset %d of %d (precursors %d to %d of %d) with m/z range %.4f - %.4f",
                experiment.config.subset,
                experiment.config.subsets,
                idx_start + 1,
                idx_end,
                len(mz_array),
                mz_array[idx_start],
                mz_array[idx_end - 1],
            )
            logger.debug(
                "Calculated precursor offsets for all subsets: %s ... %s",
                experiment.offsets[:3],
                experiment.offsets[-3:],
            )
        else:
            logger.info("Processing all %d precursors for %s", nprecursors, self)
            idx_start, idx_end = 0, nprecursors

        subset_size = idx_end - idx_start
        seq = seq[sort_idx[idx_start:idx_end] % npeptides]
        mz_array = mz_array[idx_start:idx_end]
        logger.debug("Subset peptide sequences: %s ... %s", seq[:3], seq[-3:])

        seq_array = self.shared_array(
            experiment, "peptide_sequences", shape=(subset_size,), dtype=seq.dtype
        )
        seq_array[:] = seq

        mzrt_shape = (subset_size, 3 if experiment.config.model_ccs is not None else 2)
        mzrt = self.shared_array(experiment, "mzrt", shape=mzrt_shape, dtype=np.float32)
        mzrt[:] = np.nan
        mzrt[:, 0] = mz_array

        # irt prediction
        if (
            experiment.config.cache_conf
            and experiment.config.cache_conf.cache_properties
        ):
            index = experiment.cache[IndexType.IRT]
        else:
            index = None

        if ncharges == 1:
            # sequences do not repeat in the input
            self.get_predictions(
                "irt",
                index,
                pd.DataFrame({"peptide_sequences": seq}, copy=False),
                mzrt[:, 1],
                experiment,
            )
        else:
            # sequences may repeat but we might have a subset of all precursors
            seq_unique, inverse = np.unique(seq, return_inverse=True)
            irt_array = np.empty(seq_unique.shape[0], dtype=np.float32)
            self.get_predictions(
                "irt",
                index,
                pd.DataFrame({"peptide_sequences": seq_unique}, copy=False),
                irt_array,
                experiment,
            )
            mzrt[:, 1] = irt_array[inverse]

        logger.debug("Predicted iRT values: %s ... %s", mzrt[:5, 1], mzrt[-5:, 1])

        charge_array = self.shared_array(
            experiment, "precursor_charges", shape=(subset_size,), dtype=np.uint8
        )

        charge_array[:] = np.arange(
            experiment.config.min_charge,
            experiment.config.max_charge + 1,
            dtype=np.uint8,
        )[sort_idx[idx_start:idx_end] // npeptides]

        peptide_data = {
            "peptide_sequences": seq_array,
            "precursor_charges": charge_array,
            "irt": mzrt[:, 1],
            "m/z": mzrt[:, 0],
        }

        if experiment.config.model_ccs is not None:
            if (
                experiment.config.cache_conf
                and experiment.config.cache_conf.cache_properties
            ):
                index = experiment.cache[IndexType.CCS]
            else:
                index = None
            self.get_predictions(
                "ccs",
                index,
                pd.DataFrame(
                    {"peptide_sequences": seq_array, "precursor_charges": charge_array},
                    copy=False,
                ),
                mzrt[:, 2],
                experiment,
            )
            peptide_data["ccs"] = mzrt[:, 2]

        logger.debug("Creating peptide DataFrame%s with columns:", self.suffix)
        for col, values in peptide_data.items():
            logger.debug(
                "  %s: dtype=%s, shape=%s, samples=%s ... %s",
                col,
                values.dtype,
                values.shape,
                values[:5],
                values[-5:],
            )
        df = pd.DataFrame(
            peptide_data,
            copy=False,
            index=range(idx_start, idx_end),
        )
        # the rest of the columns are not in shared memory
        df["collision_energies"] = experiment.config.collision_energy
        if experiment.config.fragmentation_type is not None:
            df["fragmentation_types"] = experiment.config.fragmentation_type.value
        return df
