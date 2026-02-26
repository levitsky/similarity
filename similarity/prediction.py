from typing import TYPE_CHECKING
import pandas as pd
import numpy as np
from koinapy import Koina
from .utils import Fixture, SpectrumIndex, RTIndex, IMIndex
from pyteomics import cmass, auxiliary as aux, proforma
import threading
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
    ) -> dict[tuple[str, int], "np.ndarray"]:
        processed = {}
        for i, (peptide, charge) in enumerate(
            inputs[["peptide_sequences", "precursor_charges"]].values
        ):
            idx = result["mz"][i] > 0
            processed[(peptide, charge)] = (
                result["mz"][i][idx],
                result["intensities"][i][idx],
            )
        return processed

    @staticmethod
    def save_predictions(
        data: dict[tuple[str, int], "np.ndarray"], index: SpectrumIndex
    ) -> None:
        with index.transact():
            for (peptide, charge), (mz, intensities) in data.items():
                index[(peptide, charge)] = mz, intensities

    def evaluate(self, experiment: "Experiment") -> SpectrumIndex:
        df = experiment.peptides
        index = SpectrumIndex(experiment=experiment)
        # make sure pairs are calculated
        experiment.pairs
        cached = df.apply(
            lambda row: (row["peptide_sequences"], row["precursor_charges"]) in index,
            axis=1,
        )
        logger.info(
            "Dropping %d peptides not in any pairs", (df["in pairs"] == False).sum()
        )
        df = df.loc[df["in pairs"]]
        cached = cached.loc[df.index]
        logger.info("%d of %d spectra are cached", cached.sum(), len(df))
        if cached.all():
            logger.info("All spectra are cached, skipping prediction")
            return index
        model = Koina(experiment.config.model_intensity, experiment.config.koina_host)
        prediction_inputs = df.loc[~cached]
        result = model.predict(prediction_inputs, df_output=False)
        logger.info("Preprocessing %d new predictions...", result["mz"].shape[0])
        data = self.process_predictions(prediction_inputs, result)
        logger.info("Saving predictions to cache...")
        self.save_predictions(data, index)
        logger.info("Caching complete.")
        return index


class MzIrtDataFrame(Fixture):
    @staticmethod
    def _write_to_cache(df: pd.DataFrame, name: str, index: "Index") -> None:
        with index.transact():
            for _, row in df.iterrows():
                index[row["peptide_sequences"]] = row[name]
        logger.info("Saved %d %s predictions to cache", df.shape[0], name)

    @staticmethod
    def save_predictions(df: pd.DataFrame, name: str, index: "Index | None") -> None:
        if index is not None:
            t = threading.Thread(
                target=MzIrtDataFrame._write_to_cache, args=(df, name, index)
            )
            t.start()

    def get_predictions(
        self,
        name: str,
        index: "Index | None",
        inputs: pd.DataFrame,
        experiment: "Experiment",
    ) -> None:
        if index is None:
            inputs[name] = np.nan
            ncached = 0
            logger.info(
                "Skipping index lookup for %s prediction, no cache configured", name
            )
        else:
            inputs[name] = inputs["peptide_sequences"].apply(index.get).astype(float)
            ncached = inputs[name].notna().sum()
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
            df = model.predict(inputs.loc[inputs[name].isna()], df_output=True)
            logger.debug(
                "Predicted %s values (%d rows in total):\n%s",
                name,
                df.shape[0],
                df.head(),
            )
            self.save_queue.append((df, name, index))
            inputs.update(df[[name]])
            if inputs[name].isna().any():
                logger.warning(
                    "Some %s values are still missing after prediction, these will be skipped: %s",
                    name,
                    inputs.loc[inputs[name].isna(), "peptide_sequences"].tolist(),
                )
        else:
            logger.info("All %s values are cached, skipping prediction", name)

    def evaluate(self, experiment: "Experiment") -> pd.DataFrame:
        input_file = experiment.config.input_file
        df = pd.read_table(input_file, names=["peptide_sequences"], header=None)
        logger.info("Loaded %d peptide sequences from %s", len(df), input_file)
        df.drop_duplicates(inplace=True)
        logger.info(
            "After dropping duplicates, %d unique peptide sequences remain", len(df)
        )
        if not experiment.config.nonstandard_aminoacids and not experiment.config.ptms:
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

        self.save_queue = []
        if experiment.config.cache_properties:
            index = RTIndex(experiment=experiment)
        else:
            index = None
        self.get_predictions("irt", index, df, experiment)

        df["precursor_charges"] = experiment.config.min_charge
        dfs = [df]
        if experiment.config.max_charge > experiment.config.min_charge:
            for charge in range(
                experiment.config.min_charge + 1, experiment.config.max_charge + 1
            ):
                df_copy = df.copy()
                df_copy["precursor_charges"] = charge
                dfs.append(df_copy)
            df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
            logger.info(
                "Expanded dataframe to %d rows by adding charge states from %d to %d",
                len(df),
                experiment.config.min_charge,
                experiment.config.max_charge,
            )

        if experiment.config.model_ccs is not None:
            if experiment.config.cache_properties:
                index = IMIndex(experiment=experiment)
            else:
                index = None
            self.get_predictions("ccs", index, df, experiment)

        if experiment.config.ptms:

            def mz(row):
                seq = row["peptide_sequences"]
                charge = row["precursor_charges"]
                try:
                    return cmass.fast_mass(seq, charge=charge)
                except aux.PyteomicsError as e:
                    return proforma.ProForma.parse(seq).mz(charge=charge)

        else:
            mz = lambda row: cmass.fast_mass(
                row["peptide_sequences"], charge=row["precursor_charges"]
            )

        df["m/z"] = df.apply(mz, axis=1)
        df["collision_energies"] = experiment.config.collision_energy
        if experiment.config.fragmentation_type is not None:
            df["fragmentation_types"] = experiment.config.fragmentation_type
        for item in self.save_queue:
            self.save_predictions(*item)
        del self.save_queue
        return df
