from typing import Iterable, Union
import numpy as np
import pandas as pd
import pomegranate
import networkx as nx

def discretize(data: pd.DataFrame, policy: Iterable[Iterable]) -> pd.DataFrame:
    assert type(data) is pd.DataFrame
    #assert data.shape[1] == policy.shape[0]

    data = data.copy()

    for i in range(data.shape[1]):
        data['tmp'] = 0
        for cnt, threshold in enumerate(policy[i]):
            data.loc[data.iloc[:,i] >= threshold, 'tmp'] = cnt + 1
        data.iloc[:,i] = data['tmp']
    data = data.drop(columns='tmp')

    return data

def bn_to_graph(model: pomegranate.BayesianNetwork) -> nx.DiGraph:
    structure = model.structure
    G = nx.DiGraph()
    G.add_nodes_from(range(len(structure)))
    for node, edges in enumerate(structure):
        for edge in edges:
            G.add_edge(node, edge)
    return G

def show(G: Union[pomegranate.BayesianNetwork, nx.DiGraph]) -> None:
    if type(G) is pomegranate.BayesianNetwork:
        G = bn_to_graph(G)
    nx.draw(G, with_labels=True)

if __name__ == "__main__":
    data = np.array([[1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1]])
    policy = np.array([[1.05], [1.9, 3.0], [3.01, 3.07], [4.02]])
    data = pd.DataFrame(data)
    print(data)
    data = discretize(data, policy)
    print(data)