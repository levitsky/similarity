import unittest
import numpy as np
from similarity.experiment import Experiment
from similarity.utils import Config
from pathlib import Path


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.config = Config(input_file=Path("tests/test_peptides.txt"))

    def test_run(self):
        """Test that Experiment.run() executes and returns something."""
        exp = Experiment(self.config)
        result = exp.run()
        # Check that run() returns the processed_pairs object
        print("Final result:")
        print(result)
        print(result.shape)
        self.assertEqual(result.shape[0], 9)  # Assuming 9 pairs based on the test input
        self.assertTrue(
            np.allclose(
                result["similarity_score"],
                [
                    0.81768,
                    0.912134,
                    0.772697,
                    0.816326,
                    0.858346,
                    0.974192,
                    0.724647,
                    0.933183,
                    0.847243,
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
