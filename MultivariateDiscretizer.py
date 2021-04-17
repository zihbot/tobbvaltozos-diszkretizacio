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

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.columns = self.column_labels = range(data.shape[1])
        self._set_column_types()

    def _set_column_types(self) -> None:
        self.column_types = [ColumnType.CONTINUOUS] * len(self.columns)
        for i in self.columns:
            self.column_types[i] = get_column_type(self.data[:, i])
    
def concat_array(data: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(data.shape) > len(target.shape):
        target = np.expand_dims(target, 0)
    if data.shape[0] != target.shape[0]:
        target = target.transpose()
    return np.hstack((data, target))

if __name__ == '__main__':
    import sklearn.datasets
    iris = sklearn.datasets.load_iris()
    d = MultivariateDiscretizer(concat_array(iris['data'], iris['target']))
    print(d.column_types)