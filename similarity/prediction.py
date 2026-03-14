from typing import TYPE_CHECKING
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
    import numpy as np
    from numpy.typing import DTypeLike
    from .experiment import Experiment
    from .utils.abc import Index, SpectrumCollection

logger = logging.getLogger(__name__)


class PredictedSpectrumCollection(Fixture):
    batch_factor: int = 5

    @staticmethod
    def preprocess_predictions(
        result: dict[str, list["np.ndarray"]],
    ) -> dict[str, list["np.ndarray"]]:
        for arr in result["intensities"]:
            np.sqrt(arr, out=arr, where=arr > 0)
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
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)

        bsize = experiment.config.batch_size * self.batch_factor
        nbatches = (prediction_inputs.shape[0] + bsize - 1) // bsize
        logger.info(
            "Predicting spectra for %d peptides in %d batches of size %d...",
            prediction_inputs.shape[0],
            nbatches,
            bsize,
        )
        with logging_redirect_tqdm():
            for i in trange(
                nbatches, desc=experiment.config.model_intensity, unit="batch"
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
    batch_factor: int = 5

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
        index: "Index | None",
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
                getattr(experiment.config, f"model_{name}"),
                experiment.config.koina_host,
            )
            masked = inputs.loc[mask]
            bsize = experiment.config.batch_size * self.batch_factor
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
                    desc=f"{getattr(experiment.config, f'model_{name}')}",
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

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        input_file = experiment.config.input_file
        seq = np.unique(np.loadtxt(input_file, dtype=bytes))

        logger.info("Loaded %d unique peptide sequences from %s", len(seq), input_file)

        if not experiment.config.nonstandard_aminoacids and not experiment.config.ptms:
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

        if seq.size == 0:
            logger.error("No valid peptide sequences found in input file, aborting")
            return pd.DataFrame()

        ncharges = experiment.config.max_charge - experiment.config.min_charge + 1
        npeptides = len(seq)
        nprecursors = npeptides * ncharges
        logger.info("%d total precursors to analyze", nprecursors)

        seq_array = self.shared_array(
            experiment, "peptide_sequences", shape=(nprecursors,), dtype=seq.dtype
        )
        for i in range(ncharges):
            seq_array[i * npeptides : (i + 1) * npeptides] = seq

        mzrt_shape = (nprecursors, 3 if experiment.config.model_ccs is not None else 2)

        mzrt = self.shared_array(experiment, "mzrt", shape=mzrt_shape, dtype=np.float32)
        mzrt[:] = np.nan
        if experiment.config.cache_properties:
            index = experiment.cache[IndexType.IRT]
        else:
            index = None
        self.get_predictions(
            "irt",
            index,
            pd.DataFrame({"peptide_sequences": seq}, copy=False),
            mzrt[:npeptides, 1],
            experiment,
        )
        logger.debug("Predicted iRT values:\n%s", mzrt[:10, 1])
        for i in range(1, ncharges):
            mzrt[i * npeptides : (i + 1) * npeptides, 1] = mzrt[:npeptides, 1]

        charge_array = self.shared_array(
            experiment, "precursor_charges", shape=(nprecursors,), dtype=np.uint8
        )
        for charge in range(
            experiment.config.min_charge, experiment.config.max_charge + 1
        ):
            charge_array[
                (charge - experiment.config.min_charge)
                * npeptides : (charge - experiment.config.min_charge + 1)
                * npeptides
            ] = charge

        peptide_data = {
            "peptide_sequences": seq_array,
            "precursor_charges": charge_array,
            "irt": mzrt[:, 1],
            "m/z": mzrt[:, 0],
        }

        if experiment.config.model_ccs is not None:
            if experiment.config.cache_properties:
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

        if experiment.config.ptms:
            for i, (peptide, charge) in enumerate(
                zip(
                    peptide_data["peptide_sequences"], peptide_data["precursor_charges"]
                )
            ):
                peptide = peptide.decode("ascii")
                try:
                    mzrt[i, 0] = cmass.fast_mass(peptide, charge=charge)
                except aux.PyteomicsError:
                    mzrt[i, 0] = proforma.ProForma.parse(peptide).mz(charge=charge)

        else:
            for i, (peptide, charge) in enumerate(
                zip(
                    peptide_data["peptide_sequences"], peptide_data["precursor_charges"]
                )
            ):
                peptide = peptide.decode("ascii")
                mzrt[i, 0] = cmass.fast_mass(peptide, charge=charge)

        df = pd.DataFrame(peptide_data, copy=False)
        # the rest of the columns are not in shared memory
        df["collision_energies"] = experiment.config.collision_energy
        if experiment.config.fragmentation_type is not None:
            df["fragmentation_types"] = experiment.config.fragmentation_type
        return df
