from matplotlib import pyplot as plt
from pandas.core.algorithms import mode
from .discretization import DiscretizationError, discretize_one
from typing import Tuple
import pandas as pd
import numpy as np
import enum
import pomegranate
import networkx as nx
import logging
import copy
from sklearn import metrics

from . import util

logger = logging.getLogger("MultivariateDiscretizer")

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
    graph: nx.digraph.DiGraph = None
    name: str = None
    bn_algorithm: str = None
    number_of_classes = None

    def __init__(self, data: np.ndarray, name: str = "Unknown",
            bn_algorithm = 'chow-liu', graph: nx.digraph.DiGraph = None) -> None:
        assert len(data.shape) == 2, 'Only supports 2 dimensional matricies!'
        self.name = name
        self.bn_algorithm = bn_algorithm
        self.data, self.column_unique_values = self._string_array_to_int(data)
        self.columns = self.column_labels = range(data.shape[1])
        self._set_column_types()
        self._set_initial_discretizations()
        if graph is None:
            self.learn_structure()
        else:
            self.graph = graph

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

    def _set_initial_discretizations(self) -> None:
        self.discretization = [[]] * len(self.columns)
        if self.number_of_classes is None:
            self.number_of_classes = 1
            for i, t in enumerate(self.column_types):
                if t == ColumnType.DISCRETE and self.number_of_classes < len(np.unique(self.data[:, i])):
                    self.number_of_classes = len(np.unique(self.data[:, i]))
        for i, t in enumerate(self.column_types):
            if t == ColumnType.CONTINUOUS:
                d = self.data[:, i]
                cutpoints = [x for x in range(0, len(d)-self.number_of_classes, len(d)//self.number_of_classes)][1:]
                values = sorted(d)
                self.discretization[i] = [values[it] for it in cutpoints]
        logger.debug("_set_initial_discretizations() Initial discretization: {}".format(self.discretization))

    def _discretize_one(self, i: int) -> None:
        df = self.as_dataframe()
        D = self.as_dataframe(self.get_discretized_data())
        try:
            L = util.largest_markov_cardinality(D, self.graph, df.columns[i])
        except:
            L = self.number_of_classes
        logger.debug("_discretize_one() {}. column, L={}, discretization before: {}".format(i, L, self.discretization[i]))
        try:
            disc = discretize_one(D, self.graph, df[df.columns[i]], L)
            logger.debug("_discretize_one() discretization after: {}".format(disc))
            self.discretization[i] = disc
        except DiscretizationError as e:
            logger.debug("_discretize_one() discretization after error: {}".format(self.discretization[i]))

    def _discretize_all(self, max_cycles: int = 10) -> None:
        for i in range(max_cycles):
            logger.debug("_discretize_all() {}. cycle, discretization before: {}".format(i, self.discretization))
            before_disc = copy.deepcopy(self.discretization)
            for c in self._node_fit_order():
                if self.column_types[c] == ColumnType.CONTINUOUS:
                    self._discretize_one(c)
            logger.debug("_discretize_all() {}. cycle, discretization after: {}".format(i, self.discretization))
            if self.discretization == before_disc:
                break

    def fit(self, max_epochs: int = 10) -> None:
        logger.info("fit() started, max epochs: {}".format(max_epochs))
        for i in range(max_epochs):
            before_disc = copy.deepcopy(self.discretization)
            logger.info("fit() {}. epoch".format(i))
            self._discretize_all()
            self.learn_structure()
            if self.discretization == before_disc:
                break
        logger.info("fit() ended")

    #endregion

    #region Evaluate
    def evaluate(self) -> list[float]:
        model = self.model()
        result =  {}
        for c in self.columns:
            if self.column_types[c] == ColumnType.DISCRETE:
                logger.info("evaluate() {}. column".format(c))
                df = self.get_discretized_data()
                df.loc[:, c] = None
                y_true = self.data[:, c].astype(int)
                y_pred = np.array(model.predict(df.to_numpy()))[:, c].astype(int)
                precision = metrics.precision_score(y_true, y_pred, average='weighted')
                recall = metrics.recall_score(y_true, y_pred, average='weighted')
                result[str(c)] = {'precision': precision, 'recall': recall}
        return result


    #endregion

    #region Graph

    def model(self) -> pomegranate.BayesianNetwork:
        return pomegranate.BayesianNetwork.from_samples(self.get_discretized_data(), algorithm=self.bn_algorithm)

    def learn_structure(self):
        model = self.model()
        self.graph = util.bn_to_graph(model)

    def show(self):
        util.show(self.graph)

    def draw_structure_to_file(self, filename=None) -> None:
        if filename is None:
            filename = "{}.{}.structure.png".format(self.name, self.bn_algorithm)
        util.show(self.graph)
        plt.savefig(filename)
        plt.close()

    def _node_fit_order(self) -> list[int]:
        return list(reversed(list(nx.topological_sort(self.graph))))

    #endregion

    #region Format

    def as_dataframe(self, data: np.ndarray = None) -> pd.DataFrame:
        if data is None:
            data = self.data
        return pd.DataFrame(data)

    def get_discretized_data(self):
        return util.discretize(self.as_dataframe(), self.discretization)

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
        from . import MultivariateDiscretizer
        iris = sklearn.datasets.load_iris()
        d = MultivariateDiscretizer(concat_array(iris['data'], iris['target']))
        print(d.discretization)
        print(d.column_types)
