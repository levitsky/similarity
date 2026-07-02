import multiprocessing as mp
from abc import ABC, abstractmethod
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from pathlib import Path

if TYPE_CHECKING:
    from .config import Config

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


class ExperimentRunner:
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
            c, self.input_file, self.peptide_table, self.spectrum_file
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
