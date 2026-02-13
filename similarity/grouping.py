from typing import Any, TYPE_CHECKING
from scipy.spatial import cKDTree
import numpy as np
import logging
from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class SpectrumGrouping(Fixture):
    def evaluate(self, experiment: "Experiment") -> list[tuple[int, int]]:
        # very close to legacy implementation, but not the bottleneck for now
        df = experiment.mz_irt_df
        tree = cKDTree(df[["m/z", "irt"]].values)
        logger.debug("Built cKDTree with %d nodes from %d points", tree.size, tree.n)
        radius = np.sqrt(
            experiment.config.mz_tolerance**2 + experiment.config.irt_tolerance**2
        )
        pairs = []
        for i, row in df.iterrows():
            mz, irt = row["m/z"], row["irt"]
            indices = tree.query_ball_point([mz, irt], radius)
            logger.debug(
                "Found %d neighbors for index %d (m/z: %.2f, irt: %.2f)",
                len(indices),
                i,
                mz,
                irt,
            )
            for j in indices:
                if i < j:  # Avoid self-pairing and duplicate pairs
                    logger.debug("Processing pair: index1=%d, index2=%d", i, j)
                    if (
                        abs(mz - df.loc[j, "m/z"]) <= experiment.config.mz_tolerance
                        and abs(irt - df.loc[j, "irt"])
                        <= experiment.config.irt_tolerance
                    ):
                        logger.debug(
                            "Pair (index1=%d, index2=%d) is within tolerance", i, j
                        )
                        pairs.append((i, j))
                    else:
                        logger.debug(
                            "Pair (index1=%d, index2=%d) is NOT within tolerance", i, j
                        )
        logger.debug("Total valid pairs found: %d", len(pairs))
        return pairs
