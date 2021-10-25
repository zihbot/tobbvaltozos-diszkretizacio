#%%
import sklearn.datasets
import MultivariateDiscretizer
iris = sklearn.datasets.load_iris()
d = MultivariateDiscretizer.MultivariateDiscretizer(MultivariateDiscretizer.concat_array(iris['data'], iris['target']))
d.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique
import pandas as pd
import sklearn.datasets
import util
import discretization
import structure

iris = sklearn.datasets.load_iris()
df = pd.DataFrame(np.hstack([iris['data'], np.expand_dims(iris['target'], axis=1)]))
continous_df = df[[0, 1, 2, 3]].copy()
disc_df = util.discretize(continous_df, discretization.get_initial_disctretization(df, continous_df))
df[disc_df.columns] = disc_df
#discretization.discretize_all(df, structure.get_graph(df), continous_df)
# %%
#labor = pd.read_csv('working_data.csv', index_col=0)
#labor['sex'] = labor['sex'].replace({'f': 0, 'm': 1})
#labor.columns = range(len(labor.columns))

#df = labor.sample(200).reset_index(drop=True)
#continous_df = df[[0, 2, 3, 4]].copy()
# %%
L__X = discretization.get_initial_disctretization(df, continous_df, k=3)
# %%
from importlib import reload
import time
discretization = reload(discretization)
# %%
for i in range(1):
    starttime = time.time()
    L__X_last = L__X.copy()
    disc_df = util.discretize(continous_df, L__X)
    df[disc_df.columns] = disc_df
    L__X, df = discretization.discretize_all(df, structure.get_graph(df), continous_df, L__X=L__X, max_iter=6)
    print(L__X)
    print("Runtime: ", time.time() - starttime)

# %%
import networkx as nx
nx.draw(structure.get_graph(df), with_labels=True)
# %%
print("---------------------------")
#L__X = L__X_last
# %%
import numpy as np
import scipy as sc
