#%%
import pomegranate
import matplotlib.pyplot as plt
import util
import pandas as pd
import networkx as nx

def learn_structure(disctretized_df: pd.DataFrame) -> pomegranate.BayesianNetwork:
    return pomegranate.BayesianNetwork.from_samples(disctretized_df, algorithm='chow-liu')

def get_graph(disctretized_df: pd.DataFrame = None) -> nx.DiGraph:
    model = learn_structure(disctretized_df)
    return util.bn_to_graph(model)