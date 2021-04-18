from typing import Tuple
import pandas as pd
import numpy as np
import enum

class ColumnType(enum.Enum):
    DISCRETE = 0
    CONTINUOUS = 1

def get_column_type(array: np.ndarray) -> ColumnType:
    return ColumnType.DISCRETE if np.equal(np.mod(array, 1), 0).all() else ColumnType.CONTINUOUS

class MultivariateDiscretizer:
    data = None
    columns = None
    column_labels = None
    column_types = None
    column_unique_values = None

    def __init__(self, data: np.ndarray) -> None:
        assert len(data.shape) == 2, 'Only supports 2 dimensional matricies!'
        self.data, self.column_unique_values = self._string_array_to_int(data)
        self.columns = self.column_labels = range(data.shape[1])
        self._set_column_types()

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
    
def concat_array(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(data.shape) > len(target.shape):
        target = np.expand_dims(target, 0)
    if data.shape[0] != target.shape[0]:
        target = target.transpose()
    return np.hstack((data, target))


if __name__ == '__main__':
    if True:
        data = np.array([['a', 1, 1.1], ['b', 1, 2.1], ['b', 2, 1.3], ['a', 2, 2.4]])
        d = MultivariateDiscretizer(data)
        print(d.data)
        print(d.column_unique_values)
        print(d.column_types)

    if False:
        import sklearn.datasets
        iris = sklearn.datasets.load_iris()
        d = MultivariateDiscretizer(concat_array(iris['data'], iris['target']))
        print(d.column_types)
