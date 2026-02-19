from typing import Any, TYPE_CHECKING
from scipy.spatial import cKDTree
import numpy as np
import logging
from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class SpectrumGrouping(Fixture):
    def process_neighbors(
        self, neighbors: list[list[int]], experiment: "Experiment"
    ) -> list[tuple[int, int]]:
        mzrt = experiment.mz_irt_df[["m/z", "irt"]].values
        pairs = []
        inpairs = np.zeros(len(experiment.mz_irt_df), dtype=bool)
        for i, indices in enumerate(neighbors):
            for j in indices:
                if i < j:  # Avoid self-pairing and duplicate pairs
                    if (
                        abs(mzrt[i, 0] - mzrt[j, 0]) <= experiment.config.mz_tolerance
                        and abs(mzrt[i, 1] - mzrt[j, 1])
                        <= experiment.config.irt_tolerance
                    ):
                        pairs.append((i, j))
                        inpairs[i] = True
                        inpairs[j] = True
        experiment.mz_irt_df["in pairs"] = inpairs
        logger.info("Total valid pairs found: %d", len(pairs))
        return pairs

    def evaluate(self, experiment: "Experiment") -> list[tuple[int, int]]:
        df = experiment.mz_irt_df
        df["in pairs"] = False  # Add a column to track if a spectrum is in any pair
        tree = cKDTree(df[["m/z", "irt"]].values)
        logger.info("Built cKDTree with %d nodes from %d points", tree.size, tree.n)
        radius = np.sqrt(
            experiment.config.mz_tolerance**2 + experiment.config.irt_tolerance**2
        )
        neightbors = tree.query_ball_tree(tree, r=radius)
        return self.process_neighbors(neightbors, experiment)
