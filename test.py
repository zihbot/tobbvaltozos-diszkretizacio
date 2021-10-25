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
        print(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
        self.discretizer._discretize_all(8)
        print(self.discretizer.discretization)
        print(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
        #self.discretizer.

if __name__ == '__main__':
    unittest.main()