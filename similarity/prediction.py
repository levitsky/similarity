from typing import TYPE_CHECKING
import pandas as pd
from koinapy import Koina
from .utils import Fixture, SpectrumIndex, RTIndex, IMIndex
from pyteomics import cmass
import logging

if TYPE_CHECKING:
    import numpy as np
    from .experiment import Experiment
    from .utils import Index

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
    def save_predictions(df: pd.DataFrame, name: str, index: "Index") -> None:
        with index.transact():
            for _, row in df.iterrows():
                index[row["peptide_sequences"]] = row[name]
        logger.info("Saved %d %s predictions to cache.", df.shape[0], name)

    def get_predictions(
        self, name: str, index: "Index", inputs: pd.DataFrame, experiment: "Experiment"
    ) -> None:
        cached = inputs["peptide_sequences"].apply(lambda seq: seq in index)
        logger.info(
            "%d of %d peptides are cached for %s prediction",
            cached.sum(),
            inputs.shape[0],
            name,
        )
        if not cached.all():
            logger.info("Predicting %s for %d peptides...", name, (~cached).sum())
            model = Koina(
                getattr(experiment.config, f"model_{name}"),
                experiment.config.koina_host,
            )
            df = model.predict(inputs.loc[~cached], df_output=True)
            logger.info("Predicted %s for %d peptides.", name, df.shape[0])
            self.save_predictions(df, name, index)
            inputs[name] = df[name]
        else:
            logger.info("All %s values are cached, skipping prediction", name)
        inputs.loc[cached, name] = inputs["peptide_sequences"].apply(
            lambda seq: index[seq]
        )

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        input_file = experiment.config.input_file
        df = pd.read_table(input_file, names=["peptide_sequences"], header=None)
        logger.info("Loaded %d peptide sequences from %s", len(df), input_file)
        df.drop_duplicates(inplace=True)
        logger.info(
            "After dropping duplicates, %d unique peptide sequences remain", len(df)
        )
        if not experiment.config.nonstandard_amino_acids:
            unsupported = df["peptide_sequences"].str.contains(
                "[^ACDEFGHIKLMNPQRSTVWY]", regex=True
            )
            if unsupported.any():
                logger.warning(
                    "Found %d unsupported peptide sequences, these will be skipped: %s",
                    unsupported.sum(),
                    df.loc[unsupported, "peptide_sequences"].tolist(),
                )
            df = df.loc[~unsupported].reset_index(drop=True)

        index = RTIndex(experiment=experiment)
        self.get_predictions("irt", index, df, experiment)

        df["precursor_charges"] = experiment.config.charge
        if experiment.config.model_ccs is not None:
            index = IMIndex(experiment=experiment)
            self.get_predictions("ccs", index, df, experiment)

        df["m/z"] = df["peptide_sequences"].apply(
            lambda seq: cmass.fast_mass(seq, charge=experiment.config.charge)
        )
        df["collision_energies"] = experiment.config.collision_energy
        return df
