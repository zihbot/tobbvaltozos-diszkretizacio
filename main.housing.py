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
from bediscretizer import discretization
from bediscretizer.MultivariateDiscretizer import ColumnType

algos = ['greedy', 'chow-liu', 'exact', 'k2', 'multi_k2', 'best_edge']
#for algo in algos:
#    try:
algo = 'chow-liu'
print(algo)
if not os.path.isdir('logs/{}'.format(algo)):
    os.mkdir('logs/{}'.format(algo))
logging.basicConfig(
    filename="logs/{}/{}.log".format(algo, datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

dataset = sklearn.datasets.load_boston()
data = bediscretizer.util.concat_array(dataset['data'], dataset['target'])
column_types = [ColumnType.CONTINUOUS] * 14
column_types[3] = ColumnType.DISCRETE
column_types[8] = ColumnType.DISCRETE

begin_time = time()
d = bediscretizer.MultivariateDiscretizer(
    data,
    'Housing',
    algo,
    column_types=column_types,
    initial_discretizer='equal_sample'
)
print('time')


d.learn_structure()
#order = structure.k2_order_entropies(d.get_discretized_data())
#print(order)
#d.fit_k2(order=[3, 4, 6, 1, 7, 2, 8, 12, 13, 9, 0, 5, 10, 11])
#d.fit(10)

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

algo = 'chow-liu'
print(algo)
if not os.path.isdir('logs/{}'.format(algo)):
    os.mkdir('logs/{}'.format(algo))
logging.basicConfig(
    filename="logs/{}/{}.log".format(algo, datetime.now().strftime("%Y%m%d %H%M%S")),
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

dataset = sklearn.datasets.load_boston()
data = bediscretizer.util.concat_array(dataset['data'], dataset['target'])
column_types = [ColumnType.CONTINUOUS] * 14
column_types[3] = ColumnType.DISCRETE
column_types[8] = ColumnType.DISCRETE

kf = KFold(n_splits=10, shuffle=True, random_state=235)
evaluation = None
for train_i, test_i in kf.split(data):
    print('New epoch')
    d = bediscretizer.MultivariateDiscretizer(
        data,
        'Housing',
        algo,
        column_types=column_types,
        initial_discretizer='equal_sample'
    )

    #d.fit_k2(order=[1,5,6,12,13,9,2,8,10,3,11,0,7,4])
    d.learn_structure()
    #print(list(d.graph.edges))
    #order = structure.k2_order_entropies(d.get_discretized_data())
    #print(order)
    #d.fit_k2(order=order)

    test_col = 8
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

dataset = sklearn.datasets.load_boston()
data = bediscretizer.util.concat_array(dataset['data'], dataset['target'])
column_types = [ColumnType.CONTINUOUS] * 14
column_types[3] = ColumnType.DISCRETE
column_types[8] = ColumnType.DISCRETE

begin_time = time()
d = bediscretizer.MultivariateDiscretizer(
    data,
    'Housing',
    algo,
    column_types=column_types,
    initial_discretizer='equal_width'
)
disc = d.discretization

for di in d.discretization:
    if di is not None: print(', '.join(map(lambda d: '{:.2f}'.format(d), di)))


fig, ax = plt.subplots()
plt.scatter(dataset['data'][:, 0], dataset['data'][:, 1], c=dataset['data'][:, 8])
ax.set_xticks(disc[0])
ax.set_yticks(disc[1])
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.set_xlabel('Bűnözés')
ax.set_ylabel('Zóna')
ax.set_xticklabels(['9.89', '19.78', '29.66', '39.55', '49.43', '59.32','69.21', '79.09'])
ax.set_yticklabels(['11.11', '22.22', '33.33', '44.44', '55.56', '66.67', '77.78', '88.89'])
plt.savefig('housing_0_1_width')
# %%
