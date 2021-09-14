# %%
from matplotlib import pyplot as plt
from pandas.core.algorithms import mode
from typing import Tuple
import pandas as pd
import numpy as np
import enum
import pomegranate
import networkx as nx
import logging
import copy
import bediscretizer
import sklearn.datasets
import logging
import os

iris = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(iris['data'], iris['target'])
d = bediscretizer.MultivariateDiscretizer(data, 'Iris')
model = pomegranate.BayesianNetwork.from_samples(d.data, algorithm=d.bn_algorithm)
print(model)
# %%





import bediscretizer
import sklearn.datasets

iris = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(iris['data'], iris['target'])
d = bediscretizer.MultivariateDiscretizer(data, 'Iris')
d.fit()
d.draw_structure_to_file()

















