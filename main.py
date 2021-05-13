import bediscretizer
import sklearn.datasets

iris = sklearn.datasets.load_iris()
d = bediscretizer.MultivariateDiscretizer(bediscretizer.util.concat_array(iris['data'], iris['target']))
print(d.discretization)
print(d.column_types)