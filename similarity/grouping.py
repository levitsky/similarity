from typing import Any, TYPE_CHECKING
from scipy.spatial import cKDTree
import numpy as np
import logging
from .utils import Fixture

if TYPE_CHECKING:
    from .experiment import Experiment

logger = logging.getLogger(__name__)


class SpectrumGrouping(Fixture):
    def kdtree(self, experiment: "Experiment") -> cKDTree:
        df = experiment.mz_irt_df
        names = ["m/z", "irt"]
        if experiment.config.model_ccs is not None:
            names.append("ccs")
        return cKDTree(df[names].values)

    def radius(self, experiment: "Experiment") -> float:
        mz_tol = experiment.config.mz_tolerance
        irt_tol = experiment.config.irt_tolerance
        if experiment.config.model_ccs is not None:
            ccs_rtol = experiment.config.ccs_rtolerance
            ccs_tol = ccs_rtol * experiment.mz_irt_df["ccs"].max()
            return np.sqrt(mz_tol**2 + irt_tol**2 + ccs_tol**2)
        return np.sqrt(mz_tol**2 + irt_tol**2)

    def within_tolerance_2d(
        self, i: int, j: int, arr: np.ndarray, experiment: "Experiment"
    ) -> bool:
        mz_tol = experiment.config.mz_tolerance
        irt_tol = experiment.config.irt_tolerance
        return (
            abs(arr[i, 0] - arr[j, 0]) <= mz_tol
            and abs(arr[i, 1] - arr[j, 1]) <= irt_tol
        )

    def within_tolerance_3d(
        self, i: int, j: int, arr: np.ndarray, experiment: "Experiment"
    ) -> bool:
        mz_tol = experiment.config.mz_tolerance
        irt_tol = experiment.config.irt_tolerance
        ccs_rtol = experiment.config.ccs_rtolerance
        return (
            abs(arr[i, 0] - arr[j, 0]) <= mz_tol
            and abs(arr[i, 1] - arr[j, 1]) <= irt_tol
            and abs(arr[i, 2] - arr[j, 2]) <= ccs_rtol * max(arr[i, 2], arr[j, 2])
        )

    def tolerance_check(self, experiment: "Experiment"):
        if experiment.config.model_ccs is not None:
            return self.within_tolerance_3d
        return self.within_tolerance_2d

    def process_neighbors(
        self, neighbors: list[list[int]], experiment: "Experiment"
    ) -> list[tuple[int, int]]:

        if experiment.config.model_ccs is not None:
            mzrt = experiment.mz_irt_df[["m/z", "irt", "ccs"]].values
        else:
            mzrt = experiment.mz_irt_df[["m/z", "irt"]].values
        pairs = []
        inpairs = np.zeros(len(experiment.mz_irt_df), dtype=bool)
        within_tolerance = self.tolerance_check(experiment)
        for i, indices in enumerate(neighbors):
            for j in indices:
                if i < j:  # Avoid self-pairing and duplicate pairs
                    if within_tolerance(i, j, mzrt, experiment):
                        pairs.append((i, j))
                        inpairs[i] = True
                        inpairs[j] = True
        experiment.mz_irt_df["in pairs"] = inpairs
        logger.info("Total valid pairs found: %d", len(pairs))
        return pairs

    def evaluate(self, experiment: "Experiment") -> list[tuple[int, int]]:
        df = experiment.mz_irt_df
        df["in pairs"] = False  # Add a column to track if a spectrum is in any pair
        tree = self.kdtree(experiment)
        logger.info("Built cKDTree with %d nodes from %d points", tree.size, tree.n)
        radius = self.radius(experiment)
        neightbors = tree.query_ball_tree(tree, r=radius)
        return self.process_neighbors(neightbors, experiment)
