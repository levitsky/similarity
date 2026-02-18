from typing import TYPE_CHECKING
import pandas as pd
from koinapy import Koina
from .utils import Fixture, Index
from pyteomics import cmass
import logging

if TYPE_CHECKING:
    import numpy as np
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class PredictedSpectrumCollection(Fixture):

    @staticmethod
    def process_predictions(
        inputs: pd.DataFrame, result: dict[str, "np.ndarray"]
    ) -> dict[str, "np.ndarray"]:
        processed = {}
        for i, peptide in enumerate(inputs["peptide_sequences"].values):
            idx = result["mz"][i] > 0
            processed[peptide] = result["mz"][i][idx], result["intensities"][i][idx]
        return processed

    @staticmethod
    def save_predictions(data: dict[str, "np.ndarray"], index: Index) -> None:
        with index.transact():
            for peptide, (mz, intensities) in data.items():
                index[peptide] = mz, intensities

    def evaluate(self, experiment: "Experiment") -> Index:
        df = experiment.mz_irt_df
        index = Index(experiment=experiment)
        # make sure pairs are calculated
        experiment.pairs
        logger.info("Found cache with %d entries", len(index))
        df["cached"] = df["peptide_sequences"].apply(lambda seq: seq in index)
        logger.info(
            "Dropping %d peptides not in any pairs", (df["in pairs"] == False).sum()
        )
        df = df.loc[df["in pairs"]]
        logger.info("%d of %d spectra are cached", df["cached"].sum(), len(df))
        if df["cached"].all():
            logger.info("All spectra are cached, skipping prediction")
            return index
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)
        prediction_inputs = df.loc[~df["cached"]]
        result = model.predict(prediction_inputs, df_output=False)
        logger.info("Preprocessing %d new predictions...", result["mz"].shape[0])
        data = self.process_predictions(prediction_inputs, result)
        logger.info("Saving predictions to cache...")
        self.save_predictions(data, index)
        logger.info("Caching complete, total cache size is now %d", len(index))
        return index


class MzIrtDataFrame(Fixture):
    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        input_file = experiment.config.input_file
        inputs = pd.read_table(input_file, names=["peptide_sequences"], header=None)
        logger.info("Loaded %d peptide sequences from %s", len(inputs), input_file)
        inputs.drop_duplicates(inplace=True)
        logger.info(
            "After dropping duplicates, %d unique peptide sequences remain", len(inputs)
        )
        unsupported = inputs["peptide_sequences"].str.contains(
            "[^ACDEFGHIKLMNPQRSTVWY]", regex=True
        )
        if unsupported.any():
            logger.warning(
                "Found %d unsupported peptide sequences, these will be skipped: %s",
                unsupported.sum(),
                inputs.loc[unsupported, "peptide_sequences"].tolist(),
            )
        inputs = inputs.loc[~unsupported].reset_index(drop=True)
        model = Koina(experiment.config.model_irt, experiment.config.koina_host)
        df = model.predict(inputs, df_output=True)
        df["m/z"] = df["peptide_sequences"].apply(
            lambda seq: cmass.fast_mass(seq, charge=experiment.config.charge)
        )
        df["precursor_charges"] = experiment.config.charge
        df["collision_energies"] = experiment.config.collision_energy
        return df
