# %%
from datetime import date, datetime
from time import time
import bediscretizer
import sklearn.datasets
import logging
import os

algos = ['chow-liu', 'greedy', 'exact']
#for algo in algos:
#    try:
algo = 'exact'
print(algo)
if os.path.isfile("{}.log".format(algo)):
    os.remove("{}.log".format(algo))
logging.basicConfig(
    filename="{}.log".format(datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

iris = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(iris['data'], iris['target'])
d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo)

d.fit()
print(d.discretization)
print(d.column_types)
d.draw_structure_to_file()
print(d.evaluate())
# %%
