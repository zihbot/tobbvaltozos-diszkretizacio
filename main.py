# %%
import bediscretizer
import sklearn.datasets
import logging
import os

os.remove("run.log")
logging.basicConfig(filename="run.log", level=logging.DEBUG)

iris = sklearn.datasets.load_iris()
data = bediscretizer.util.concat_array(iris['data'], iris['target'])
d = bediscretizer.MultivariateDiscretizer(data, 'Iris')
# %%
d.fit()
print(d.discretization)
print(d.column_types)
d.draw_structure_to_file()