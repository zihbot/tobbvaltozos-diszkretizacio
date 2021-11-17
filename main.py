# %%
from datetime import date, datetime
from time import time

import pandas as pd
import bediscretizer
import sklearn.datasets
import logging
import os

algos = ['chow-liu', 'greedy', 'exact']
#for algo in algos:
#    try:
algo = 'greedy'
print(algo)
logging.basicConfig(
    filename="logs/{}.log".format(datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

iris = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(iris['data'], iris['target'])
d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo)

d.fit()
print(d.discretization)
print(d.column_types)
d.draw_structure_to_file('out.png')
print(d.evaluate())
# %%
[1,0,1] == 1
# %%
