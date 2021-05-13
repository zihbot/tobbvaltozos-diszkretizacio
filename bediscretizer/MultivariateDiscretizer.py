from .discretization import discretize_one
from typing import Tuple
import pandas as pd
import numpy as np
import enum
import pomegranate
import networkx as nx

from . import util

class ColumnType(enum.Enum):
    DISCRETE = 0
    CONTINUOUS = 1

def get_column_type(array: np.ndarray) -> ColumnType:
    return ColumnType.DISCRETE if np.equal(np.mod(array, 1), 0).all() else ColumnType.CONTINUOUS

class MultivariateDiscretizer:
    data: np.ndarray = None
    columns: list[int] = None
    column_labels: list[str] = None
    column_types: list[ColumnType] = None
    column_unique_values: dict[int, np.ndarray] = None
    discretization: list[list[float]] = None
    graph: nx.digraph = None

    def __init__(self, data: np.ndarray) -> None:
        assert len(data.shape) == 2, 'Only supports 2 dimensional matricies!'
        self.data, self.column_unique_values = self._string_array_to_int(data)
        self.columns = self.column_labels = range(data.shape[1])
        self._set_column_types()
        self._set_initial_discretizations()
        self.learn_structure()

    #region Preprocessing

    def _set_column_types(self) -> None:
        self.column_types = [ColumnType.CONTINUOUS] * len(self.columns)
        for i in self.columns:
            self.column_types[i] = get_column_type(self.data[:, i])
    
    def _string_array_to_int(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        decoders = {}
        if np.issubdtype(data.dtype, str):
            for col_id in range(data.shape[1]):
                try:
                    data[:, col_id].astype(float)
                except ValueError:
                    unique: np.ndarray = np.unique(data[:, col_id])
                    decoders[col_id] = unique
                    encoder = dict((j,i) for i,j in enumerate(unique))
                    data[:, col_id] = np.array([encoder[i] for i in data[:, col_id]])
        return data.astype(float), decoders
    
    #endregion
    
    #region Discretization

    def _set_initial_discretizations(self, number_of_classes: int = None) -> None:
        self.discretization = [[]] * len(self.columns)
        if number_of_classes is None:
            number_of_classes = 1
            for i, t in enumerate(self.column_types):
                if t == ColumnType.DISCRETE and number_of_classes < len(np.unique(self.data[:, i])):
                    number_of_classes = len(np.unique(self.data[:, i]))
        for i, t in enumerate(self.column_types):
            if t == ColumnType.CONTINUOUS:
                d = self.data[:, i]                
                cutpoints = [x for x in range(0, len(d)-number_of_classes, len(d)//number_of_classes)][1:]
                values = sorted(d)
                self.discretization[i] = [values[it] for it in cutpoints]

    def _discretize_one(self, i: int) -> None:
        df = self.as_dataframe()
        discretize_one(self.as_dataframe(self.get_discretized_data()), self.graph, df[df.columns[i]])

    #endregion

    #region Graph

    def learn_structure(self):
        model = pomegranate.BayesianNetwork.from_samples(self.data, algorithm='exact')
        self.graph = util.bn_to_graph(model)

    def show(self):
        util.show(self.graph)

    #endregion

    #region Format

    def as_dataframe(self, data: np.ndarray = None):
        if data is None:
            data = self.data
        return pd.DataFrame(data)

    def get_discretized_data(self):
        return util.discretize(self.as_dataframe, self.discretization)

    #endregion

def concat_array(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(data.shape) > len(target.shape):
        target = np.expand_dims(target, 0)
    if data.shape[0] != target.shape[0]:
        target = target.transpose()
    return np.hstack((data, target))


if __name__ == '__main__':
    if False:
        data = np.array([['a', 1, 1.1], ['b', 1, 2.1], ['b', 2, 1.3], ['a', 2, 2.4]])
        d = MultivariateDiscretizer(data)
        d.show()

    if True:
        import sklearn.datasets
        iris = sklearn.datasets.load_iris()
        d = MultivariateDiscretizer(concat_array(iris['data'], iris['target']))
        print(d.discretization)
        print(d.column_types)
