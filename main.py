import bediscretizer
import sklearn.datasets
import time

iris = sklearn.datasets.load_iris()
d = bediscretizer.MultivariateDiscretizer(bediscretizer.util.concat_array(iris['data'], iris['target']), 'Iris')
print(d.discretization)
print(d.column_types)
d.draw_to_file()