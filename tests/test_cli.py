import unittest
import numpy as np
from similarity.experiment import Experiment
from similarity.utils import Config
from pathlib import Path
import logging


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.config = Config(input_file=Path("tests/test_peptides.txt"))
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # overrides any existing logging config
        )
        self.logger = logging.getLogger("TestExperiment")

    def test_run(self):
        """Test that Experiment.run() executes and returns something."""
        exp = Experiment(self.config)
        result = exp.run().sort_values(["index1", "index2"])
        # Check that run() returns the processed_pairs object
        self.logger.debug("Final result:\n%s", result)
        self.assertEqual(result.shape[0], 9)  # Assuming 9 pairs based on the test input
        self.assertTrue(
            np.allclose(
                result["similarity score"],
                [
                    0.847243,
                    0.816326,
                    0.724647,
                    0.912134,
                    0.772697,
                    0.81768,
                    0.974192,
                    0.858346,
                    0.933183,
                ],
                atol=1e-3,
            )
        )


if __name__ == "__main__":
    unittest.main()
