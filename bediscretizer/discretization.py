# %%
from typing import Hashable, Tuple
import numpy as np
import pandas as pd
from pandas.core.algorithms import unique
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pomegranate
import matplotlib.pyplot as plt
from . import util
import networkx as nx
import scipy as sc
import math
import logging

logger = logging.getLogger("Discretization")

class DiscretizationError(Exception):
    pass

def precalculate_probability_table_as_definition(D: pd.DataFrame, G: nx.DiGraph, ci: Hashable) -> np.ndarray:
    n = D.shape[0]
    H = np.zeros((n, n))
    P = [p for p in G.predecessors(ci)]

    J_P = 1
    for p in P:
        J_P *= len(np.unique(D[p]))

    C = [c for c in G.successors(ci)]
    S = [None] * len(C)
    S_c = [None] * len(C)
    J_C = [1] * len(C)
    J_S = [1] * len(C)
    for i, c in enumerate(C):
        S[i] = [s for s in G.predecessors(c)]
        S[i].remove(ci)
        S_c[i] = [s for s in S[i]]
        S_c[i].append(c)
        J_C[i] = len(np.unique(D[c]))
        for spouse in S[i]:
            J_S[i] *= len(np.unique(D[spouse]))

    for v in range(n):
        for u in range(v + 1):
            g_ = v + 1 - u
            h = math.log(sc.special.comb(g_ + J_P - 1, J_P - 1))
            if len(P) != 0:
                h += math.log(math.factorial(g_))
                n_P_q = D.iloc[u:v+1,P].groupby(P).size().reset_index(name='cnt')['cnt']
                h -= sum(np.log(sc.special.factorial(n_P_q)))

            for i, c in enumerate(C):
                n_j_i_m_l = D.iloc[u:v+1,S_c[i]].groupby(S_c[i]).size().reset_index(name='cnt')['cnt']
                if len(S[i]) == 0:
                    n_j_i_l = [sum(n_j_i_m_l)]
                else:
                    n_j_i_l = D.iloc[u:v+1,S[i]].groupby(S[i]).size().reset_index(name='cnt')['cnt']
                h += sum([math.log(sc.special.comb(n + J_C[i] - 1, J_C[i] - 1)) for n in n_j_i_l])
                h += sum(np.log(sc.special.factorial(n_j_i_l)))
                h -= sum(np.log(sc.special.factorial(n_j_i_m_l)))

            H[u, v] = h

    return H

def precalculate_probability_table_dynamic_programming(D: pd.DataFrame, G: nx.DiGraph, ci: Hashable) -> np.ndarray:
    n = D.shape[0]
    H = np.zeros((n, n))
    P = [p for p in G.predecessors(ci)]
    C = [c for c in G.successors(ci)]
    S = [None] * len(C)

    J_P = 1
    S_c = [None] * len(C)
    J_C = [1] * len(C)
    J_S = [1] * len(C)
    for p in P:
        J_P *= len(np.unique(D[p]))
    for i, c in enumerate(C):
        S[i] = [s for s in G.predecessors(c)]
        S[i].remove(ci)
        S_c[i] = [s for s in S[i]]
        S_c[i].append(c)
        J_C[i] = len(np.unique(D[c]))
        for spouse in S[i]:
            J_S[i] *= len(np.unique(D[spouse]))

    for v in range(n):
        for u in range(v + 1):
            H[u, v] = math.log(sc.special.comb(v - u + J_P, J_P - 1))

    # Parent table
    for p in P:
        p_dist = pd.get_dummies(D[p])
        dist_table = np.zeros((n, n, len(p_dist.iloc[0,:])), dtype=int)
        for v in range(n):
            for u in range(v + 1):
                # fill dist_table
                if v == u:
                    dist_table[u, v] = p_dist.iloc[v,:]
                else:
                    dist_table[u, v] = dist_table[u, v-1] + p_dist.iloc[v,:]

                # calculate probability
                h = math.log(math.factorial(v+1-u))
                h -= sum(np.log(sc.special.factorial(dist_table[u, v])))
                H[u, v] += h

    # Child-Spouse table
    for i, c in enumerate(C):
        c_dist = pd.get_dummies(D[c])
        n_c = len(c_dist.iloc[0,:])

        s_class: pd.Series
        if len(S[i]) > 0:
            s_class = D[S[i]].groupby(S[i]).ngroup()
        else:
            s_class = pd.Series(np.zeros(n))
        n_s_class = len(unique(s_class))

        dist_table = np.zeros((n, n, n_s_class, n_c), dtype=int)
        for v in range(n):
            for u in range(v + 1):
                h = 0

                for i_s_class in range(n_s_class):
                    z = np.zeros(n_c)
                    if i_s_class == s_class[v]:
                        z = c_dist.iloc[v,:]

                    # fill dist_table
                    if v == u:
                        dist_table[u, v, i_s_class] = z
                    else:
                        dist_table[u, v, i_s_class] = dist_table[u, v-1, i_s_class] + z

                    # calculate probability
                    c_over_s_dist = dist_table[u, v, i_s_class]
                    n_c_over_s = sum(c_over_s_dist)
                    h += math.log(sc.special.comb(n_c_over_s + J_C[i] - 1, J_C[i] - 1))
                    h += math.log(math.factorial(n_c_over_s))
                    h -= sum(np.log(sc.special.factorial(c_over_s_dist)))
                H[u, v] += h

    print('TAB', H[:10,:10])
    exit()

    return H

def discretize_one(D: pd.DataFrame, G: nx.DiGraph, X: pd.Series, L: int) -> list:
    #structure = pomegranate.BayesianNetwork.from_samples(disc_df, algorithm='chow-liu').structure
    ci = X.name # continous variable iterator

    D = D.copy()
    D[ci] = X
    D.sort_values(ci, inplace=True)
    #n = D.shape[0]
    #D = D.iloc[n//20:n*19//20,:].copy() # Drop lower and upper 5 percentile
    D.reset_index(drop=True, inplace=True)

    H = precalculate_probability_table_dynamic_programming(D, G, ci)

    n = D.shape[0]
    uX, s = np.unique(D[ci], return_index=True)
    m = len(uX)
    s[0] = n
    s = np.roll(s - 1, -1)

    S = np.zeros(m)
    L_ = [set()] * m
    W = np.append(-np.log([1 - math.exp(-L * (uX[i+1] - uX[i]) / (uX[m-1] - uX[0])) for i in range(m-1)]), 0)
    for v in range(m):
        if v == 0:
            S[v] = H[0, s[v]] + W[v]
            L_[v] = {(uX[v] + uX[v+1]) / 2}
        else:
            _S, _u, DiscEdge = None, 0, None
            for u in range(v + 1):
                if u == v:
                    __S = W[v] + H[0, s[v]] + L * (uX[v] - uX[0]) / (uX[m-1] - uX[0])
                else:
                    __S = W[v] + H[s[u]+1, s[v]] + L * (uX[v] - uX[u+1]) / (uX[m-1] - uX[0]) + S[u]
                if _S is None or _S > __S:
                    if u == len(uX)-1:
                        logger.warning("No discretization edge found" + str(L_))
                        _S, _u, DiscEdge = __S, u, uX[u]
                        raise DiscretizationError("No discretization edge found " + str(L_))
                    else:
                        _S, _u, DiscEdge = __S, u, (uX[u] + uX[u+1]) / 2
            S[v] = _S
            L_[v] = L_[_u].union({DiscEdge})
    return sorted(L_[m-1])

#%%
def discretize_all(D: pd.DataFrame, G: nx.DiGraph, X: pd.DataFrame, L__X: list = None, max_iter: int = 3) -> Tuple[list, pd.DataFrame]:
    D = D.copy()
    continous_classes = X.columns
    if L__X is None:
        L__X = get_initial_disctretization(D, X)
    _D_X = util.discretize(X, np.array(L__X))
    for c in continous_classes:
        D[c] = _D_X[c]

    for iter in range(max_iter):
        print("Iteration: {}".format(iter))
        for i, c in enumerate(continous_classes):
            print("\tDiscretize: {}".format(c))
            L__X[i] = discretize_one(D, G, X[c])
            _D_X = util.discretize(X, np.array(L__X))
            D[c] = _D_X[c]
    return L__X, D
#%%
def get_initial_disctretization(D: pd.DataFrame, X: pd.DataFrame, k: int = None) -> list:
    continous_classes = X.columns
    D_d = D.drop(continous_classes, axis=1)
    n = X.shape[0]
    L__X = [[]] * X.shape[1]
    if k is None:
        k = max(D_d.nunique())
    cutpoints = [x for x in range(0, n-k, n//k)][1:]
    for i, c in enumerate(continous_classes):
        values = sorted(X[c])
        L__X[i] = [values[it] for it in cutpoints]
    return L__X
#%%
import sklearn.datasets
if __name__ == "__main__":
    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(np.hstack([iris['data'], np.expand_dims(iris['target'], axis=1)]))
    disc_df = util.discretize(df, np.array([[5,6],[3],[2.5,3.5],[1,2],[0.5,1.5]]))
    model = pomegranate.BayesianNetwork.from_samples(disc_df, algorithm='chow-liu')
    #L_ = discretize_one(disc_df, util.bn_to_graph(model), df[0])
# %%
