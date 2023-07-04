import unittest
from bluepyopt.deapext import DifferentialEvolutionOptimisation

class TestDifferentialEvolutionOptimisation(unittest.TestCase):
    def setUp(self):
        self.deo = DifferentialEvolutionOptimisation()

    def test_init(self):
        self.assertIsInstance(self.deo, DifferentialEvolutionOptimisation)

    def test_run(self):
        # Add necessary setup for running the method
        # Call the run method and check the result
        pass

    # Add more test cases for other methods in the DifferentialEvolutionOptimisation class


