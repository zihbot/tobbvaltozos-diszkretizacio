# %%
from datetime import date, datetime
import random
from time import time
import numpy as np

import pandas as pd
import bediscretizer
import sklearn.datasets
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from bediscretizer.MultivariateDiscretizer import ColumnType

algos = ['chow-liu', 'greedy', 'exact', 'k2', 'multi_k2', 'best_edge']
#for algo in algos:
#    try:
algo = 'k2'
print(algo)
if not os.path.isdir('logs/{}'.format(algo)):
    os.mkdir('logs/{}'.format(algo))
logging.basicConfig(
    filename="logs/{}/{}.log".format(algo, datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


#dataset = sklearn.datasets.load_iris()
dataset = sklearn.datasets.load_diabetes()
data = bediscretizer.util.concat_array(dataset['data'], dataset['target'])
print(dataset)
exit()
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.1, random_state=42)
data = bediscretizer.util.concat_array(X_train, y_train)
#d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo)
coltype = [ColumnType.CONTINUOUS] * data.shape[1]
coltype[1] = ColumnType.DISCRETE
d = bediscretizer.MultivariateDiscretizer(data, 'Diabetes', algo, column_types=coltype)

d.fit(20)

'''

data = pd.read_csv('szivroham.csv')
data.iloc[:,[0,1,2]] = np.nan
data.dropna(axis=1, how='any', inplace=True)
N = data.shape[0]
data = data.drop(random.sample(range(N), k = 49*N//50)).reset_index(drop=True)
data = bediscretizer.util.discretize(data, [[0.9, 1.5],None,None])

min1 = data.quantile(.05)[1]
min2 = data.quantile(.05)[2]
max1 = data.quantile(.95)[1]
max2 = data.quantile(.95)[2]

data[(data.iloc[:,1] < min1) | (data.iloc[:,1] > max1)] = np.nan
data[(data.iloc[:,2] < min2) | (data.iloc[:,2] > max2)] = np.nan
data.dropna(axis=0, how='any', inplace=True)

d = bediscretizer.MultivariateDiscretizer(data.to_numpy(), 'Szivroham', algo)
d.fit(20)

'''
'''
kf = KFold(n_splits=10, shuffle=True, random_state=235)
for train_i, test_i in kf.split(data):
    d = bediscretizer.MultivariateDiscretizer(data[train_i, :], 'Iris', algo)
    d.fit(20)

    print(d.discretization)
    test_col = 4
    y_pred = d.predict(data[test_i, :], test_col)
    print(d._evaluate(data[test_i, test_col], y_pred))
'''

print(d.discretization)
print(d.column_types)
d.draw_structure_to_file('out.png')
#print(d.evaluate())
# %%
[1,0,1] == 1
# %%

# %%
import random
for i in range(10):
    print("{:.2f}".format(random.random()*2))
# %%
