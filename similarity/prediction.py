from typing import TYPE_CHECKING
import pandas as pd
from koinapy import Koina
from .utils import Fixture, SpectrumIndex, RTIndex
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
    def save_predictions(data: dict[str, "np.ndarray"], index: SpectrumIndex) -> None:
        with index.transact():
            for peptide, (mz, intensities) in data.items():
                index[peptide] = mz, intensities

    def evaluate(self, experiment: "Experiment") -> SpectrumIndex:
        df = experiment.mz_irt_df
        index = SpectrumIndex(experiment=experiment)
        # make sure pairs are calculated
        experiment.pairs
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
        logger.info("Caching complete.")
        return index


class MzIrtDataFrame(Fixture):
    @staticmethod
    def save_rt_predictions(df: pd.DataFrame, index: RTIndex) -> None:
        with index.transact():
            for _, row in df.iterrows():
                index[row["peptide_sequences"]] = row["irt"]

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
        index = RTIndex(experiment=experiment)
        inputs["cached"] = inputs["peptide_sequences"].apply(lambda seq: seq in index)
        logger.info(
            "%d of %d peptides are cached for RT prediction",
            inputs["cached"].sum(),
            inputs.shape[0],
        )
        inputs.loc[inputs["cached"], "irt"] = inputs.loc[
            inputs["cached"], "peptide_sequences"
        ].apply(lambda seq: index[seq])
        if not inputs["cached"].all():
            model = Koina(experiment.config.model_irt, experiment.config.koina_host)
            df = model.predict(inputs.loc[~inputs["cached"]], df_output=True)
            logger.info("Predicted RT for %d peptides.", df.shape[0])
            self.save_rt_predictions(df, index)
            logger.info("RT caching complete")
            if inputs["cached"].any():
                df = pd.concat([inputs.loc[inputs["cached"]], df], axis=0)

        else:
            logger.info("All RTs are cached, skipping prediction")
            df = inputs
        df["m/z"] = df["peptide_sequences"].apply(
            lambda seq: cmass.fast_mass(seq, charge=experiment.config.charge)
        )
        df["precursor_charges"] = experiment.config.charge
        df["collision_energies"] = experiment.config.collision_energy
        return df
