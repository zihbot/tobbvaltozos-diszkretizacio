#%%
from os import name
import numpy as np
import pomegranate
import matplotlib.pyplot as plt
import networkx.algorithms.isomorphism as iso
import pandas as pd
import networkx as nx
import random
try:
    from . import util
except:
    import util

def learn_structure(disctretized_df: pd.DataFrame, algorithm: str = 'exact', include_edges: list[tuple[int]] = [], **kwargs) -> pomegranate.BayesianNetwork:
    if algorithm == 'k2':
        g = learn_k2_structure(disctretized_df, **kwargs)
        parents = util.graph_to_bn_structure(g, list(disctretized_df.columns), True)
        return pomegranate.BayesianNetwork.from_structure(disctretized_df, parents)
    else:
        return pomegranate.BayesianNetwork.from_samples(disctretized_df, algorithm=algorithm, include_edges=(include_edges if len(include_edges) != 0 else None))

def get_graph(disctretized_df: pd.DataFrame = None) -> nx.DiGraph:
    model = learn_structure(disctretized_df)
    return util.bn_to_graph(model)

def learn_k2_structure(df: pd.DataFrame, order: list[int] = None, upper_bound: int = 2, p_step: list[list[int]] = None) -> nx.DiGraph:
    if order is None:
        order = list(df.columns)
        random.shuffle(order)
    n = len(order)
    if n != len(list(df.columns)):
        raise ValueError('Number of columns {} and length of order {} not match'.format(len(order), len(list(df.columns))))

    p = [[] for i in range(n)] if p_step is None or len(p_step) == 0 else p_step
    for i in range(n):
        P_old = util.preference_bias(df, order[i], p[i])
        ok_to_proceed = True
        while ok_to_proceed and len(p[i]) < upper_bound+1:
            P_new = None

            # find z that maximizes preference_bias
            z = None
            for zi in range(i):
                if order[zi] in p[i]: continue
                pb = util.preference_bias(df, order[i], [*p[i], order[zi]])
                if P_new is None or pb > P_new:
                    z = order[zi]
                    P_new = pb

            if P_new is not None and P_new > P_old:
                P_old = P_new
                p[i].append(z)
                if p_step is not None:
                    return util.parents_to_graph(p, order[:i+1])
            else:
                ok_to_proceed = False
    return util.parents_to_graph(p, order)

"""
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
"""
if __name__ == '__main__':
    df = pd.DataFrame(np.array([
        [1,0,0],
        [1,1,1],
        [0,0,1],
        [1,1,1],
        [0,0,0],
        [0,1,1],
        [1,1,1],
        [0,0,0],
        [1,1,1],
        [0,0,0],
    ]))
    g = learn_k2_structure(df, [0, 1, 2])
    print(list(nx.topological_sort(g)))
    print(g.edges)
    print(g.nodes)
    print(util.graph_to_bn_structure(g))