import multiprocessing as mp
import numpy as np
from abc import ABC, abstractmethod
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, TypeAlias, cast, Any, Sequence
from pathlib import Path

if TYPE_CHECKING:
    from .config import Config
    from pyteomics.proforma import ProForma
    import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentWorker(ABC, mp.Process):
    def __init__(self, task_queue: mp.Queue, result_queue: mp.Queue, **kwargs):
        super().__init__(name=self.__class__.__name__, daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def run(self) -> None:
        return super().run()


class SingleInputExperimentRunner:
    """
    A class responsible for running an experiment with subsets.
    When `run` is called, it will create an Experiment object for each subset and run them.

    .. note::
        Currently, subsets are processed sequentially, and the `jobs` parameter is not supported.
        In the future, this class may be extended to support parallel execution of subsets using multiprocessing or multithreading.

    If `create_peptide_table` is True, a peptide table for the entire dataset will be created before running the experiments for each subset.
    `array_file`, and `score_df_file` are optional file paths for saving arrays and saving score dataframes, respectively.
    They SHOULD contain a placeholder `{}` for the subset number, e.g. `array_file="score_array_subset_{}.npy"`.
    """

    def __init__(
        self,
        config: "Config",
        input_file: str | Path,
        peptide_table: str | Path,
        jobs: int = 1,  # unsupported for now
        create_peptide_table: bool = True,
        spectrum_file: str | None = None,
        create_spectrum_file: bool = False,
        array_file: str | None = None,
        score_df_file: str | None = None,
    ):
        logger.debug(
            "Initializing ExperimentRunner with config: %s, input_file: %s, peptide_table: %s, jobs: %d, create_peptide_table: %s, spectrum_file: %s, create_spectrum_file: %s, array_file: %s, score_df_file: %s",
            config,
            input_file,
            peptide_table,
            jobs,
            create_peptide_table,
            spectrum_file,
            create_spectrum_file,
            array_file,
            score_df_file,
        )
        self.config = config
        self.input_file = input_file
        # self.jobs = jobs
        self.create_peptide_table = create_peptide_table
        self.peptide_table = peptide_table
        self.spectrum_file = spectrum_file
        self.create_spectrum_file = create_spectrum_file
        self.array_file = array_file
        self.score_df_file = score_df_file

    def run_subset(self, subset: int):
        from ..experiment import SingleInputExperiment

        c = replace(self.config, subset=subset)
        with SingleInputExperiment(
            c, self.input_file, self.peptide_table, self.spectrum_file
        ) as experiment:
            logger.info(
                "Running experiment %d for subset %d of %d",
                id(experiment),
                subset,
                c.subsets,
            )
            if self.array_file:
                array_path = self.array_file.format(subset)
                logger.debug(
                    "Saving score array for subset %d to %s", subset, array_path
                )
                experiment.score_array.dump(array_path)
            if self.score_df_file:
                score_df_path = self.score_df_file.format(subset)
                logger.debug(
                    "Saving score dataframe for subset %d to %s",
                    subset,
                    score_df_path,
                )
                experiment.score_df.to_csv(score_df_path, index=False)

    def create_prerequisites(self):
        from ..experiment import SingleInputExperiment

        c = replace(self.config, subsets=1)
        with SingleInputExperiment(
            c,
            input_file=self.input_file,
            peptide_table=self.peptide_table if not self.create_peptide_table else None,
            spectrum_file=self.spectrum_file if not self.create_spectrum_file else None,
        ) as experiment:
            if self.create_peptide_table:
                logger.info(
                    "Creating full peptide table for the entire dataset at %s",
                    self.peptide_table,
                )
                df = experiment.peptides
                df["peptide_sequences"] = df["peptide_sequences"].str.decode("ascii")
                df.to_csv(self.peptide_table, index=False, sep="\t")
            if self.create_spectrum_file:
                logger.info(
                    "Creating spectrum file for the entire dataset at %s",
                    self.spectrum_file,
                )
                experiment.predicted_spectra.save(cast(str, self.spectrum_file))

    def run(self):
        if self.create_peptide_table or self.create_spectrum_file:
            self.create_prerequisites()

        for subset in range(1, self.config.subsets + 1):
            self.run_subset(subset)


PeptideInput: TypeAlias = "str | Path | Sequence[str | ProForma]"
PeptideTableInput: TypeAlias = "pd.DataFrame | Path | str"
SpectrumInput: TypeAlias = "np.ndarray | Path | str"


def shape_of(value: Any) -> tuple[int, ...] | int | None:
    """Best-effort shape of a container-like argument for debug logging."""
    for attr in ("shape",):
        shape = getattr(value, attr, None)
        if shape is not None:
            return tuple(shape)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def validate_shapes(**kwargs: Any) -> None:
    for name, value in kwargs.items():
        if value is None or isinstance(value, (str, bytes, Path)):
            continue
        shape = shape_of(value)
        if shape is not None:
            logger.debug("Argument %s: shape %s, type %s", name, shape, type(value))


def _normalize_peptide_input(
    input_file: "PeptideInput | None",
    name: str,
) -> "np.ndarray | str | Path | bytes | None":
    """Normalize pre-loaded peptide sequences into an array of ASCII bytes.

    Returns path-like values unchanged so that file-based loading keeps working.
    """
    if input_file is None or isinstance(input_file, (str, bytes, Path)):
        # backward compatibility: a bare string is treated as a file path
        return input_file
    if isinstance(input_file, Sequence):
        input_file = list(input_file)
    if not isinstance(input_file, (list, tuple, set, frozenset)):
        raise TypeError(
            f"{name} must be a container of str or ProForma peptides, got {type(input_file)}"
        )
    if not input_file:
        raise ValueError(f"{name} is an empty container")
    normalized: list[bytes] = []
    for i, peptide in enumerate(input_file):
        try:
            text = str(peptide)
            normalized.append(text.encode("ascii"))
        except (UnicodeEncodeError, ValueError) as e:
            raise ValueError(
                f"{name}[{i}] cannot be encoded as ASCII: {peptide!r}"
            ) from e
        if not normalized[-1]:
            raise ValueError(f"{name}[{i}] is an empty peptide sequence")
    arr = np.array(normalized, dtype="S")
    return np.unique(arr)


def _normalize_peptide_table(
    peptide_table: "PeptideTableInput | None", name: str
) -> "pd.DataFrame | str | Path | None":
    """Validate a DataFrame passed in place of a peptide table file.

    Returns path-like values unchanged so that file-based loading keeps working.
    """
    if peptide_table is None or isinstance(peptide_table, (str, Path)):
        return peptide_table
    if not hasattr(peptide_table, "columns"):
        raise TypeError(
            f"{name} must be a pandas DataFrame, got {type(peptide_table)}"
        )
    if peptide_table.shape[0] == 0:
        raise ValueError(f"{name} is an empty DataFrame")
    logger.debug("Argument %s: shape %s, columns %s", name, peptide_table.shape, list(peptide_table.columns))
    return peptide_table


def _normalize_spectrum_array(
    spectrum_file: "SpectrumInput | None", name: str
) -> "np.ndarray | str | Path | None":
    """Validate an ndarray passed in place of a spectrum file.

    Returns path-like values unchanged so that file-based loading keeps working.
    """
    if spectrum_file is None or isinstance(spectrum_file, (str, Path)):
        return spectrum_file
    if not hasattr(spectrum_file, "shape"):
        raise TypeError(
            f"{name} must be a numpy ndarray, got {type(spectrum_file)}"
        )
    array = np.asarray(spectrum_file)
    if array.ndim != 3:
        raise ValueError(
            f"{name} must be a 3D array of shape (nspectra, 2, npeaks), got shape {array.shape}"
        )
    if array.shape[1] != 2:
        raise ValueError(
            f"{name} must have 2 rows per spectrum (m/z and intensities), got {array.shape}"
        )
    logger.debug("Argument %s: shape %s, dtype %s", name, array.shape, array.dtype)
    return array
