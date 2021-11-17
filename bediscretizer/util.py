from typing import Hashable, Iterable, Union
from matplotlib.pyplot import draw
import numpy as np
from numpy.lib import math
import pandas as pd
from pandas.core.algorithms import unique
from pomegranate import BayesianNetwork
import networkx as nx
import scipy as sc
from scipy.special import gammaln
import math

def discretize(data: pd.DataFrame, policy: Iterable[Iterable]) -> pd.DataFrame:
    assert type(data) is pd.DataFrame
    #assert data.shape[1] == policy.shape[0]

    data = data.copy()

    for i in range(data.shape[1]):
        if not policy[i]:
            continue
        data['tmp'] = 0
        for cnt, threshold in enumerate(policy[i]):
            data.loc[data.iloc[:,i] >= threshold, 'tmp'] = cnt + 1
        # If no policy, everything is the same
        if len(policy[i]) == 0:
            data['tmp'] = 0
        data.iloc[:,i] = data['tmp']
        data = data.drop(columns='tmp')

    return data

def bn_to_graph(model: BayesianNetwork) -> nx.DiGraph:
    structure = model.structure
    G = nx.DiGraph()
    G.add_nodes_from(range(len(structure)))
    for node, edges in enumerate(structure):
        for edge in edges:
            G.add_edge(edge, node)
    return G

def graph_to_bn_structure(g: nx.DiGraph, columns: list[int] = None, map_position: bool = False) -> tuple[tuple[int]]:
    if columns is None:
        columns = sorted(g.nodes)
    result = []
    for i in columns:
        if i not in g.nodes:
            result.append(tuple([]))
            continue
        p = g.predecessors(i) if not map_position else [columns.index(x) for x in g.predecessors(i)]
        result.append(tuple(p))
    return tuple(result)

def parents_to_graph(p: list[list[int]], order: list[int]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(order)
    for i, p_list in enumerate(p):
        if i >= len(order): break
        for pi in p_list:
            g.add_edge(pi, order[i])
    return g

def show(G: BayesianNetwork) -> None:
    if type(G) is BayesianNetwork:
        G = bn_to_graph(G)
    nx.draw(G, with_labels=True)

def concat_array(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(data.shape) > len(target.shape):
        target = np.expand_dims(target, 0)
    if data.shape[0] != target.shape[0]:
        target = target.transpose()
    return np.hstack((data, target))

# The largest cardinality over all discrete variables in the Markov blanket
def largest_markov_cardinality(D: pd.DataFrame, G: nx.DiGraph, x: Hashable) -> int:
    return max([len(unique(D[col])) for col in markov_blanket(G, x)])

def markov_blanket(G: nx.DiGraph, x: Hashable) -> list[Hashable]:
    parents = set(G.predecessors(x))
    children = set(G.successors(x))
    spouses = set()
    for c in children:
        spouses = spouses.union(G.predecessors(c))
    return sorted(parents.union(children).union(spouses).difference([x]))

def preference_bias(df: pd.DataFrame, i: int, p: list) -> float:
    D = df[[i, *p]].copy()
    if len(p) == 0:
        p = [i+1]
        D[p] = 0
    q = len(p)
    r = len(D[i].unique())

    # count of instantiation in j. parent and k. i
    Da = D.copy()
    Da['x'] = 1
    Da = Da.groupby(list(D.columns)).count().reset_index()
    Da = Da[['x', *p]].groupby(p)['x'].apply(list)
    La = Da.values.tolist()
    a = np.zeros([len(La),len(max(La, key=len))])
    for i,j in enumerate(La):
        a[i][0:len(j)] = j
    #a = np.array(Da.values.tolist(), dtype=object)
    N = np.sum(a, -1)

    result = q * math.log(math.factorial(r - 1))
    result -= np.sum(gammaln(N + r))
    result += np.sum(gammaln(a + 1))
    return math.exp(result)

if __name__ == "__main__":
    data = np.array([[1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1]])
    policy = np.array([[1.05], [1.9, 3.0], [3.01, 3.07], [4.02]])
    data = pd.DataFrame(data)
    print(data)
    data_disc = discretize(data, policy)
    print(data_disc)
    print('-------')
    G = nx.DiGraph([(0, 1), (1, 3), (2, 3)])
    print(markov_blanket(G, 0))
    print(markov_blanket(G, 1))
    print(markov_blanket(G, 2))
    print(markov_blanket(G, 3))
    print(largest_markov_cardinality(data_disc, G, 2))
