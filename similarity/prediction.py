from typing import TYPE_CHECKING, cast
from collections.abc import Sequence, Callable
import pandas as pd
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from koinapy import Koina
from .utils.abc import Fixture, IndexType
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

logger = logging.getLogger(__name__)


class PredictedSpectrumCollection(Fixture):
    batch_size: int = 10000

    @staticmethod
    def preprocess_predictions(
        result: dict[str, list["np.ndarray"]],
    ) -> dict[str, list["np.ndarray"]]:
        for mz, intensities in zip(result["mz"], result["intensities"]):
            np.sqrt(intensities, out=intensities, where=intensities > 0)
            order = np.argsort(mz, kind="mergesort")
            mz[:] = mz[order]
            intensities[:] = intensities[order]
        return result

    def evaluate(self, experiment: "Experiment") -> "SpectrumCollection":
        index = experiment.cache[IndexType.INTENSITY]
        collection = experiment.config.spectrum_collection.value(experiment)
        if index is not None:
            collection.fill_from_cache(experiment, index)
        cached = collection.spectra_available
        if cached.all():
            logger.info("All spectra are cached, skipping prediction")
            return collection
        prediction_inputs = experiment.peptides.loc[~cached]
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
                result = self.preprocess_predictions(result)

                collection.fill_from_predictions(batch_inputs, result)
                if index is not None:
                    index.save_predictions(batch_inputs, result)

        if index is not None:
            index.finalize()
        assert collection.is_ready()
        return collection


class MzIrtDataFrame(Fixture):
    _shared_memory: dict["Experiment", dict[str, SharedMemory]] = {}
    batch_size: int = 50000

    @classmethod
    def close(cls, experiment: "Experiment"):
        shm_dict = cls._shared_memory.pop(experiment, {})
        for name, shm in shm_dict.items():
            logger.debug(
                "Closing shared memory for experiment %d, name %s", id(experiment), name
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

    @classmethod
    def shared_array(
        cls,
        experiment: "Experiment",
        name: str,
        shape: tuple[int, ...],
        dtype: "DTypeLike",
    ) -> np.ndarray:
        cls._shared_memory.setdefault(experiment, {})
        shm_dict = cls._shared_memory[experiment]
        if name in shm_dict:
            shm = shm_dict[name]
        else:
            size = np.prod(shape, dtype=int) * np.dtype(dtype).itemsize
            logger.info(
                "Allocating %.2f MB of shared memory for %s", size / 2**20, name
            )
            shm = SharedMemory(
                name=f"{name}-{id(experiment)}",
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

        if experiment.config.subsets > 1:
            logger.info(
                "Subset processing mode. Assuming peptide table is sorted by m/z and contains the complete peptide set."
            )
            offsets = self.subset_offsets(experiment, df["m/z"])
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

        mzrt_shape = (df.shape[0], 3 if experiment.config.model_ccs is not None else 2)
        mzrt = self.shared_array(experiment, "mzrt", shape=mzrt_shape, dtype=np.float32)
        mzrt[:, 0] = df["m/z"].to_numpy(copy=False)
        mzrt[:, 1] = df["irt"].to_numpy(copy=False)
        df["m/z"] = mzrt[:, 0]
        df["irt"] = mzrt[:, 1]

        if experiment.config.model_ccs is not None:
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
        dim, tol = "m/z", c.mz_tolerance  # slicing axis and tolerance for batch overlap
        bsize = len(mz_array) // c.subsets
        values = mz_array
        offsets = [(0, bsize)]
        while offsets[-1][1] < len(values):
            end_of_batch = next_offset = offsets[-1][1]
            if next_offset >= len(values):
                break
            while values[end_of_batch] - values[next_offset - 1] <= tol:
                next_offset -= 1
                if (
                    next_offset <= offsets[-1][0]
                    or end_of_batch - next_offset >= bsize // 5
                ):
                    logger.error(
                        "Subset size is too small to accommodate the %s tolerance. "
                        "Please decrease the number of subsets.",
                        dim,
                    )
                    raise ValueError("Subset size is too small")
            offsets.append((next_offset, min(next_offset + bsize, len(values))))
        logger.debug("Calculated subset offsets: %s ... %s", offsets[:3], offsets[-3:])
        return offsets

    def has_ptms(self, experiment: "Experiment") -> bool:
        return bool(
            experiment.config.ptms
            or experiment.config.variable_mods
            or experiment.config.fixed_mods
        )

    def generate_sequences(self, experiment: "Experiment") -> np.ndarray:
        input_file = cast("str | Path", experiment.config.input_file)
        seq = np.unique(np.loadtxt(input_file, dtype=bytes))

        logger.info("Loaded %d unique peptide sequences from %s", len(seq), input_file)

        ptms = self.has_ptms(experiment)
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
            for peptide in seq:
                peptide = peptide.decode("ascii")
                try:
                    base = proforma.ProForma.parse(peptide)
                except aux.PyteomicsError as e:
                    logger.error(
                        "Failed to parse peptide sequence %s: %s", peptide, str(e)
                    )
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

        if not experiment.config.nonstandard_aminoacids and not ptms:
            common_aa = set(map(lambda s: bytes(s, "ascii")[0], parser.std_amino_acids))
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

    def mz_array(self, experiment: "Experiment", peptides: np.ndarray) -> np.ndarray:
        """
        Calculate m/z values for all peptides and charge states.
        If multiple charges are configured, the output will be longer than the input.
        """
        logger.info("Calculating m/z values for all peptides and charge states...")
        mz_calc = self.mz_calculator(experiment)
        out = []
        for charge in range(
            experiment.config.min_charge, experiment.config.max_charge + 1
        ):
            out.extend(mz_calc(peptide, charge) for peptide in peptides)
        logger.debug(
            "%d m/z values calculated, samples: %s and %s", len(out), out[:5], out[-5:]
        )
        return np.array(out, dtype=np.float32)

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        if experiment.peptide_table is not None:
            logger.info("Loading peptide table from %s", experiment.peptide_table)
            return self.load_peptide_table(experiment.peptide_table, experiment)

        seq = self.generate_sequences(experiment)
        if seq.size == 0:
            logger.error("No valid peptide sequences found in input file, aborting")
            return pd.DataFrame()

        ncharges = experiment.config.max_charge - experiment.config.min_charge + 1
        npeptides = len(seq)
        nprecursors = npeptides * ncharges
        logger.info("%d total precursors to analyze", nprecursors)

        mzarr = self.mz_array(experiment, seq)
        sort_idx = np.argsort(mzarr)
        mzarr = mzarr[sort_idx]
        if experiment.config.subsets > 1:
            offsets = self.subset_offsets(experiment, mzarr)
            logger.info(
                "Dataset will be processed in %d subsets with offsets %s",
                experiment.config.subsets,
                offsets,
            )
            idx_start, idx_end = offsets[experiment.config.subset - 1]
            logger.info(
                "Processing subset %d (precursors %d to %d of %d) with m/z range %.4f - %.4f",
                experiment.config.subset,
                idx_start + 1,
                idx_end,
                len(mzarr),
                mzarr[idx_start],
                mzarr[idx_end - 1],
            )
        else:
            logger.info("Processing all %d precursors", nprecursors)
            idx_start, idx_end = 0, len(mzarr)

        subset_size = idx_end - idx_start

        seq_idx = sort_idx[idx_start:idx_end] % npeptides
        seq_array = self.shared_array(
            experiment, "peptide_sequences", shape=(subset_size,), dtype=seq.dtype
        )
        # for i in range(ncharges):
        #     seq_array[i * npeptides : (i + 1) * npeptides] = seq
        seq_array[:] = seq[seq_idx]

        mzrt_shape = (subset_size, 3 if experiment.config.model_ccs is not None else 2)

        mzrt = self.shared_array(experiment, "mzrt", shape=mzrt_shape, dtype=np.float32)
        mzrt[:] = np.nan
        if (
            experiment.config.cache_conf
            and experiment.config.cache_conf.cache_properties
        ):
            index = experiment.cache[IndexType.IRT]
        else:
            index = None
        # after the addition of subsets, seq_array can have repeated sequences (due to different charges)
        # requesting RT predictions is slightly redundant but is left for simplicity
        # TODO: optimize
        self.get_predictions(
            "irt",
            index,
            pd.DataFrame({"peptide_sequences": seq_array}, copy=False),
            mzrt[:, 1],
            experiment,
        )
        logger.debug("Predicted iRT values:\n%s", mzrt[:10, 1])
        # for i in range(1, ncharges):
        #     mzrt[i * npeptides : (i + 1) * npeptides, 1] = mzrt[:npeptides, 1]

        charge_array = self.shared_array(
            experiment, "precursor_charges", shape=(subset_size,), dtype=np.uint8
        )
        # for charge in range(
        #     experiment.config.min_charge, experiment.config.max_charge + 1
        # ):
        #     charge_array[
        #         (charge - experiment.config.min_charge)
        #         * npeptides : (charge - experiment.config.min_charge + 1)
        #         * npeptides
        #     ] = charge
        logger.debug("Sorted indexes: %s", sort_idx[idx_start:idx_end])
        logger.debug("Sequence indexes: %s", seq_idx)
        logger.debug("Charge indexes: %s", sort_idx[idx_start:idx_end] // npeptides)

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

        mzrt[:, 0] = mzarr[idx_start:idx_end]

        # sort by m/z for better perforamce
        # logger.debug("Sorting peptides by m/z for better performance")
        # idx = np.argsort(mzrt[:, 0])
        # for arr in [seq_array, charge_array, mzrt]:
        #     arr[:] = arr[idx]

        df = pd.DataFrame(peptide_data, copy=False, index=range(idx_start, idx_end))
        # the rest of the columns are not in shared memory
        df["collision_energies"] = experiment.config.collision_energy
        if experiment.config.fragmentation_type is not None:
            df["fragmentation_types"] = experiment.config.fragmentation_type
        return df
