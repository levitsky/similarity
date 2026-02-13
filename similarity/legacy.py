from pathlib import Path
import pandas as pd
from MSCI.Preprocessing.Koina import PeptideProcessor
from MSCI.Preprocessing.Parsing import read_msp_file
from MSCI.Grouping_MS1.Grouping_mw_irt import (
    make_data_compatible,
    find_combinations_kdtree,
)
from MSCI.Similarity.spectral_angle_similarity import joinPeaks, nspectraangle
from matchms.importing import load_from_msp
from tqdm import tqdm
from multiprocessing import cpu_count, Pool, Manager
from functools import partial
import logging

logger = logging.getLogger(__name__)


# def predict_spectra(
#     input_file: str,
#     collision_energy: int = 30,
#     charge: int = 2,
#     model_intensity: str = "Prosit_2020_intensity_HCD",
#     model_irt: str = "Prosit_2019_irt",
# ):
#     pred_file = f"{Path(input_file).with_suffix('')}.msp"

#     processor = PeptideProcessor(
#         input_file=input_file,
#         collision_energy=collision_energy,
#         charge=charge,
#         model_intensity=model_intensity,
#         model_irt=model_irt,
#     )

#     processor.process(pred_file)

#     if Path(pred_file).stat().st_size == 0:
#         raise RuntimeError("Generating spectrum predictions failed")

#     spectra = list(load_from_msp(pred_file))
#     return spectra


# def get_mz_irt_df(pred_file: str):
#     df = read_msp_file(pred_file)
#     return df


# def process_peptide_combinations(mz_irt_df, tolerance1, tolerance2, use_ppm=True):
#     compatible_data = make_data_compatible(mz_irt_df)
#     result_tolerance = find_combinations_kdtree(
#         compatible_data, tolerance1, tolerance2, use_ppm
#     )
#     unique_result_tolerance = list({tuple(sorted(pair)) for pair in result_tolerance})

#     # Create a DataFrame for the results
#     results = []
#     for (index1, mw1, irt1), (index2, mw2, irt2) in unique_result_tolerance:
#         results.append(
#             {
#                 "index1": index1,
#                 "index2": index2,
#                 "peptide 1": mz_irt_df.loc[index1, "Name"],
#                 "peptide 2": mz_irt_df.loc[index2, "Name"],
#                 "m/z  1": mz_irt_df.loc[index1, "MW"],
#                 "m/z 2": mz_irt_df.loc[index2, "MW"],
#                 "iRT 1": mz_irt_df.loc[index1, "iRT"],
#                 "iRT 2": mz_irt_df.loc[index2, "iRT"],
#             }
#         )

#     results_df = pd.DataFrame(results)
#     logger.debug("results_df: %s", results_df)
#     if not results_df.empty:
#         results_df.columns = results_df.columns.str.replace(" ", "")
#     return results_df


def process_spectra_pairs(
    chunk, spectra, mz_irt_df, tolerance=0, ppm=0, m=0, n=0.5, progress_queue=None
):
    results = []

    for index_pair in chunk:
        i, j = index_pair

        # x = spectra[i]
        # y = spectra[j]

        # x_df = pd.DataFrame({"mz": x.peaks.mz, "intensities": x.peaks.intensities})
        # y_df = pd.DataFrame({"mz": y.peaks.mz, "intensities": y.peaks.intensities})
        pep1 = mz_irt_df.loc[i, "peptide_sequences"]
        pep2 = mz_irt_df.loc[j, "peptide_sequences"]
        x_df = (
            spectra.loc[spectra["peptide_sequences"] == pep1, ["mz", "intensities"]]
            .sort_values(by="mz")
            .reset_index(drop=True)
        )
        y_df = (
            spectra.loc[spectra["peptide_sequences"] == pep2, ["mz", "intensities"]]
            .sort_values(by="mz")
            .reset_index(drop=True)
        )
        # print("Processing pair:", pep1, "and", pep2)
        # print("x_df:", x_df)
        # print("y_df:", y_df)
        matcher = joinPeaks(tolerance=tolerance, ppm=ppm)
        x_matched, y_matched = matcher.match(x_df, y_df)

        angle = nspectraangle(x_matched, y_matched, m=m, n=n)

        # Extract the relevant information for the given index pair
        results.append(
            {
                "index1": i,
                "index2": j,
                "peptide 1": mz_irt_df.loc[i, "Name"],
                "peptide 2": mz_irt_df.loc[j, "Name"],
                "m/z  1": mz_irt_df.loc[i, "MW"],
                "m/z 2": mz_irt_df.loc[j, "MW"],
                "iRT 1": mz_irt_df.loc[i, "iRT"],
                "iRT 2": mz_irt_df.loc[j, "iRT"],
                "similarity_score": angle,
            }
        )

        if progress_queue is not None:
            progress_queue.put(1)  # report progress

    return pd.DataFrame(results)


def parallel_process_spectra_pairs(
    spectra_pairs, spectra, mz_irt_df, tolerance=0, ppm=0, m=0, n=0.5, n_chunks=None
):
    """
    spectra_pairs: list of (i,j) index tuples
    spectra: list of spectra objects
    mz_irt_df: DataFrame with peptide info
    """

    if n_chunks is None:
        n_chunks = cpu_count()

    # Split the pairs into roughly equal chunks
    chunk_size = (len(spectra_pairs) + n_chunks - 1) // n_chunks
    chunks = [
        spectra_pairs[i : i + chunk_size]
        for i in range(0, len(spectra_pairs), chunk_size)
    ]

    with Manager() as manager:
        progress_queue = manager.Queue()
        func = partial(
            process_spectra_pairs,
            spectra=spectra,
            mz_irt_df=mz_irt_df,
            tolerance=tolerance,
            ppm=ppm,
            m=m,
            n=n,
            progress_queue=progress_queue,
        )

        with Pool(n_chunks) as pool:
            # Launch jobs
            results_async = pool.map_async(func, chunks)

            # Show progress bar
            with tqdm(
                total=len(spectra_pairs), desc="Processing spectra pairs"
            ) as pbar:
                processed = 0
                while processed < len(spectra_pairs):
                    progress_queue.get()
                    processed += 1
                    pbar.update(1)

            # Collect results
            dfs = results_async.get()

    return pd.concat(dfs, ignore_index=True)
