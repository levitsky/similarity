import unittest

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
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
