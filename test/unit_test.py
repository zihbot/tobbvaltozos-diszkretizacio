import math
import numpy as np
import bediscretizer
import unittest
import numpy as np
import pandas as pd
import networkx as nx

class UnitTest(unittest.TestCase):
    def test_markov_blanket(self):
        graph = nx.DiGraph([(1,2), (2,4), (4,0), (0,6), (4,6), (2,6), (2,3), (3,5)])
        result = bediscretizer.util.markov_blanket(graph, 4)
        expect = [0, 2, 6]
        self.assertListEqual(result, expect, "Markov blanket not correct")

    def test_discretize(self):
        df = pd.DataFrame([0.1, 0.2, 0.5, 0.9, 1.4])
        policy = [[0.3, 1.3]]
        result = bediscretizer.util.discretize(df, policy).to_numpy().transpose().tolist()[0]
        self.assertListEqual(result, [0, 0, 1, 1, 2], msg="Discretization not correct")

