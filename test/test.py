from datetime import datetime
import logging
import time
import unittest
import numpy as np
import pandas as pd
import networkx as nx

import bediscretizer
from bediscretizer.MultivariateDiscretizer import ColumnType

logging.basicConfig(
    filename="{}.log".format(datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

class DiscretizationTest(unittest.TestCase):

    def setUp(self) -> None:
        df = pd.read_csv('test/data_auto_mpg.csv', header=None)
        data = df.to_numpy()
        graph = nx.DiGraph([(0, 1)])
        self.discretizer = bediscretizer.MultivariateDiscretizer(data, 'Test', 'exact', graph)

    def test_loaded_successfully(self):
        self.assertEqual(self.discretizer.data[1, 2], 350, 'Did not load discrete data')
        self.assertAlmostEqual(self.discretizer.data[4, 5], 10.5, 'Did not load continous data')
        self.assertEqual(self.discretizer.graph.degree(0), 1, 'Did not load graph')

    def test_known_structure(self):
        graph = nx.DiGraph([(1,2), (2,4), (4,0), (0,6), (4,6), (2,6), (2,3), (3,5)])
        self.discretizer.graph = graph
        types = [ColumnType.CONTINUOUS, ColumnType.DISCRETE, ColumnType.CONTINUOUS, ColumnType.CONTINUOUS, ColumnType.CONTINUOUS, ColumnType.CONTINUOUS, ColumnType.DISCRETE, ColumnType.DISCRETE]
        self.discretizer.column_types = types
        self.discretizer.number_of_classes = None
        self.discretizer._set_initial_discretizations()
        self.discretizer._discretize_all(8)
        exp_result  = [
            [15.25, 17.65, 20.9, 25.65, 28.9],
            None,
            [70.5, 93.5, 109.0, 159.5, 259.0, 284.5],
            [71.5, 99.0, 127.0],
            [2115.0, 2480.5, 2959.5, 3657.5],
            [12.35, 13.75, 16.05],
            None,
            None]
        self.assertListEqual(self.discretizer.discretization, exp_result, 'Bad discretization')

if __name__ == '__main__':
    unittest.main()