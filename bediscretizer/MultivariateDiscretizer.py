import itertools
import random
from matplotlib import pyplot as plt
from pandas.core.algorithms import mode

from bediscretizer import structure
from .discretization import DiscretizationError, discretize_one
from typing import Tuple, Union
import pandas as pd
import numpy as np
import enum
import pomegranate
import networkx as nx
import logging
import copy
import math
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
            bn_algorithm = 'multi_k2', graph: nx.digraph.DiGraph = None) -> None:
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

    def _reset(self):
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
                #cutpoints = [x for x in range(0, len(d)-self.number_of_classes, len(d)//self.number_of_classes)][1:]
                #values = sorted(d)
                #self.discretization[i] = [values[it] for it in cutpoints]

                # Equal width
                min_val, max_val = min(d), max(d)
                span = (max_val - min_val) / self.number_of_classes
                self.discretization[i] = [min_val + span * i for i in range(1, self.number_of_classes)]
        logger.debug("_set_initial_discretizations() Initial discretization: {}".format(self.discretization))

    def _discretize_one(self, i: int) -> None:
        if len(list(self.graph.predecessors(i))) + len(list(self.graph.successors(i))) == 0: return
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
            self.discretization[i] = []
            logger.debug("_discretize_one() discretization after error: {}".format(self.discretization[i]))

    def _discretize_all(self, max_cycles: int = 10) -> None:
        self._set_initial_discretizations()
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
        if self.bn_algorithm == 'k2':
            self._fit_k2()

        if self.bn_algorithm == 'multi_k2':
            self._fit_multi_k2(max_epoch=max_epochs)

        if self.bn_algorithm in ['chow-liu', 'greedy', 'exact']:
            self._fit_basic_structure_after_node_added(max_epoch=max_epochs)

        logger.info("fit() ended")

    def _fit_basic_structure_learner(self, max_epoch: int = 10):
        for i in range(max_epoch):
            before_disc = copy.deepcopy(self.discretization)
            logger.info("fit() {}. epoch".format(i))
            for cvar in self._node_fit_order():
                logger.debug('Structure around {}: parents={}, children={}'.format(cvar, list(self.graph.predecessors(cvar)), list(self.graph.successors(cvar))))
            self._discretize_all()
            self.learn_structure()
            if self.discretization == before_disc:
                break

    def _fit_basic_structure_after_node_added(self, max_epoch: int = 100):
        i = 0

        full_graph = nx.DiGraph()
        full_graph.add_nodes_from(self.columns)
        self._reset()
        while i < max_epoch:
            before_disc = copy.deepcopy(full_graph.edges)
            logger.info("fit() {}. edge".format(i))

            self._reset()
            self.graph = full_graph

            self._discretize_all()

            model = self.model(include_edges=list(self.graph.edges))
            edges = list(util.bn_to_graph(model).edges)
            random.shuffle(edges)
            for e in edges:
                if not full_graph.has_edge(*e):
                    full_graph.add_edge(*e)
                    break
            print(full_graph.edges)

            if full_graph.edges == before_disc:
                break
            i += 1

    def _fit_multi_k2(self, max_epoch: int = 10):
        n = len(self.columns)
        orders = []
        if math.factorial(n) > max_epoch:
            orders = [None] * max_epoch
        else:
            orders = itertools.permutations(self.columns)
        best_order = None
        best_value = None
        for i, order in enumerate(orders):
            logger.info("_fit_multi_k2() {}.".format(i))
            order_back = self._fit_k2(order)
            value = 0
            df = self.get_discretized_data()
            for i in self.columns:
                value += util.preference_bias(df, i, list(self.graph.predecessors(i)))
            if best_value is None or value > best_value:
                best_value = value
                best_order = order_back
            self._reset()
        self._fit_k2(best_order)

    def _fit_k2(self, order: list[int] = None):
        if order is None:
            '''
            # first are discrete, then conti values
            discrete = [i for i in range(len(self.column_types)) if self.column_types[i] == ColumnType.DISCRETE]
            conti = [i for i in range(len(self.column_types)) if self.column_types[i] == ColumnType.CONTINUOUS]
            random.shuffle(discrete)
            random.shuffle(conti)
            order = [*discrete, *conti]
            '''
            order = list(range(len(self.columns)))
            random.shuffle(order)
        logger.info("_fit_k2() with order: {}".format(order))
        p_step = []
        p_prev = None
        while p_step != p_prev:
            logger.info("fit()")
            for cvar in self._node_fit_order():
                logger.debug('Structure around {}: parents={}, children={}'.format(cvar, list(self.graph.predecessors(cvar)), list(self.graph.successors(cvar))))
            p_prev = copy.deepcopy(p_step)
            self.learn_structure(order=order, algorithm='k2', p_step=p_step)
            self._discretize_all()

            p_step_by_df_order = [list(self.graph.predecessors(i)) for i in sorted(self.graph.nodes)]
            p_step = [p_step_by_df_order[x] for x in order]
        self.learn_structure(order=order, algorithm='k2')
        return order

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

    def model(self, include_columns: list[int] = None, **kwargs) -> pomegranate.BayesianNetwork:
        #return pomegranate.BayesianNetwork.from_samples(self.get_discretized_data(), algorithm=self.bn_algorithm, penalty=5)
        disc_data = self.get_discretized_data()
        if include_columns is not None:
            disc_data = disc_data[include_columns].copy()
        return structure.learn_structure(disc_data, **kwargs)

    def learn_structure(self, **kwargs):
        model = self.model(**kwargs)
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
