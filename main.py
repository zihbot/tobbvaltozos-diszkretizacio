# %%
from datetime import date, datetime
from time import time
import numpy as np

import pandas as pd
import bediscretizer
import sklearn.datasets
import logging
import os
from sklearn.model_selection import train_test_split

algos = ['chow-liu', 'greedy', 'exact', 'k2', 'multi_k2', 'best_edge']
#for algo in algos:
#    try:
algo = 'best_edge'
print(algo)
if not os.path.isdir('logs/{}'.format(algo)):
    os.mkdir('logs/{}'.format(algo))
logging.basicConfig(
    filename="logs/{}/{}.log".format(algo, datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

iris = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(iris['data'], iris['target'])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.1, random_state=42)
data = bediscretizer.util.concat_array(X_train, y_train)
d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo)
d.fit(20)


print(d.discretization)
print(d.column_types)
d.draw_structure_to_file('out.png')
#print(d.evaluate())
data_test = bediscretizer.util.concat_array(X_test, np.zeros_like(y_test))
y_pred = d.predict(data_test, 4)
print(y_test, y_pred)
print(d._evaluate(y_test, y_pred))
# %%
[1,0,1] == 1
# %%
