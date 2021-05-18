#%%
import pomegranate
import matplotlib.pyplot as plt
import util
import networkx.algorithms.isomorphism as iso
import pandas as pd
import networkx as nx

def learn_structure(disctretized_df: pd.DataFrame) -> pomegranate.BayesianNetwork:
    return pomegranate.BayesianNetwork.from_samples(disctretized_df, algorithm='chow-liu')

def get_graph(disctretized_df: pd.DataFrame = None) -> nx.DiGraph:
    model = learn_structure(disctretized_df)
    return util.bn_to_graph(model)
# %%
G1 = nx.DiGraph()
G2 = nx.DiGraph()
G3 = nx.DiGraph()
nx.add_path(G1, [1, 2, 3, 4], weight=1)
nx.add_path(G2, [10, 20, 30, 40], weight=1)
nx.add_path(G3, [1, 3, 2, 4], weight=1)
util.show(G1)
plt.plot()
util.show(G2)
plt.plot()
util.show(G3)
plt.plot()

# %%
nm = lambda x, y : print("x: {}, y:{}".format(x,y))
print(nx.is_isomorphic(G1, G2, node_match=nm))
print(nx.is_isomorphic(G1, G3, node_match=nm))
print(nx.is_isomorphic(G1, G1, node_match=nm))
# %%
print(set(G1.edges))
print(set(G2.edges))
print(set(G3.edges))
print(set(G1.edges)==set(G2.edges))
# %%
