from typing import TYPE_CHECKING
import pandas as pd
import numpy as np
from koinapy import Koina
from .utils import Fixture, SpectrumIndex, RTIndex, IMIndex
from pyteomics import cmass, auxiliary as aux, proforma
import logging

if TYPE_CHECKING:
    import numpy as np
    from .experiment import Experiment
    from .utils import Index

logger = logging.getLogger(__name__)


class PredictedSpectrumCollection(Fixture):
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
        index.save_predictions(prediction_inputs, result)
        index.wait()
        return index


class MzIrtDataFrame(Fixture):

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
            index.fill_from_cache(inputs)
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
            mask = inputs[name].isna()
            masked = inputs.loc[mask]
            result = model.predict(masked, df_output=False)
            logger.debug(
                "Predicted %d %s values: %s",
                result[name].size,
                name,
                result[name][:10],
            )
            if index is not None:
                index.save_predictions(masked, result)
                index.finalize()

            inputs.loc[mask, name] = result[name]

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
        return df
