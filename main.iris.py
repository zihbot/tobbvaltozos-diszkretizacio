# %%
from datetime import date, datetime
import random
from time import time
import numpy as np

import pandas as pd
import pomegranate
import bediscretizer
import sklearn.datasets
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from bediscretizer import util
import networkx as nx
from bediscretizer import structure
from bediscretizer.MultivariateDiscretizer import ColumnType

algos = ['greedy', 'chow-liu', 'exact', 'k2', 'multi_k2', 'best_edge']
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

dataset = sklearn.datasets.load_iris()
#dataset = sklearn.datasets.load_diabetes()
data = bediscretizer.util.concat_array(dataset['data'], dataset['target'])

#X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.1, random_state=42)
#data = bediscretizer.util.concat_array(X_train, y_train)
#d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo)
#coltype = [ColumnType.CONTINUOUS] * data.shape[1]

#coltype[1] = ColumnType.DISCRETE
begin_time = time()
d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo, initial_discretizer='equal_sample')
#d.learn_structure(algorithm="chow-liu")

#d._fit_k2([4, 0, 1, 2, 3])
d.learn_structure(order=[4,0,1,2,3])

end_time = time()
print('discretization')
for disc in d.discretization:
    if disc is not None: print(', '.join(map(lambda d: '{:.2f}'.format(d), disc)))
d.draw_structure_to_file('out.png')
print('Time {:.3f}'.format(end_time-begin_time))

# %%
from datetime import date, datetime
import random
from time import time
import numpy as np

import pandas as pd
import pomegranate
import bediscretizer
import sklearn.datasets
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from bediscretizer import util
import networkx as nx
from bediscretizer import structure
from bediscretizer.MultivariateDiscretizer import ColumnType

algo = 'exact'
print(algo)
if not os.path.isdir('logs/{}'.format(algo)):
    os.mkdir('logs/{}'.format(algo))
logging.basicConfig(
    filename="logs/{}/{}.log".format(algo, datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

dataset = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(dataset['data'], dataset['target'])

kf = KFold(n_splits=10, shuffle=True, random_state=235)
evaluation = None
for train_i, test_i in kf.split(data):
    d = bediscretizer.MultivariateDiscretizer(data[train_i, :], 'Iris', algo, initial_discretizer='equal_sample')

    #d._fit_k2([4, 0, 1, 2, 3])
    d.learn_structure(order=[4,0,1,2,3])
    print(list(d.graph.edges))

    test_col = 4
    y_pred = d.predict(data[test_i, :], test_col)
    evaluation = d.evaluate(data[test_i, test_col], y_pred, evaluation)

summary = d.evalutaion_summary(evaluation)
print('summary')
for sumkey in summary.keys():
    print(sumkey)
for sumval in summary.values():
    print(sumval if isinstance(sumval, int) else '{:.2f}'.format(sumval))

# %%
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
# %%
from datetime import date, datetime
import random
from time import time
import numpy as np

import pandas as pd
import pomegranate
import bediscretizer
import sklearn.datasets
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from bediscretizer import util
import networkx as nx
from bediscretizer import structure
from bediscretizer.MultivariateDiscretizer import ColumnType
import matplotlib.pyplot as plt

dataset = sklearn.datasets.load_iris()
disc = [[5.45, 6.15],
[2.95, 3.35],
[2.45, 4.75],
[0.80, 1.75]]
disc2 = [[5.50, 6.70],
[2.90, 3.20],
[1.90, 4.90],
[0.60, 1.60]]
disc3 = [[5.50, 6.70],
[2.80, 3.60],
[2.97, 4.93],
[0.90, 1.70]]


fig, ax = plt.subplots()
plt.scatter(dataset['data'][:, 2], dataset['data'][:, 3], c=dataset['target'])
ax.set_xticks(disc3[2])
ax.set_yticks(disc3[3])
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.set_xlabel('Sziromlevél hossz')
ax.set_ylabel('Sziromlevél szélesség')
plt.savefig('2-3_disc')
# %%
