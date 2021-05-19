# %%
import bediscretizer
import sklearn.datasets
import logging
import os

algos = ['chow-liu', 'greedy', 'exact']
for algo in algos:
    try:
        print(algo)
        if os.path.isfile("{}.log".format(algo)):
            os.remove("{}.log".format(algo))
        logging.basicConfig(filename="{}.log".format(algo), level=logging.DEBUG)

        iris = sklearn.datasets.load_iris()
        data = bediscretizer.util.concat_array(iris['data'], iris['target'])
        d = bediscretizer.MultivariateDiscretizer(data, 'Iris', algo)

        d.fit()
        #d.discretization = [[5.45, 5.85], [2.95, 3.3499999999999996], [2.45, 4.25, 4.75], [0.8, 1.35, 1.75], []]
        print(d.discretization)
        print(d.column_types)
        d.draw_structure_to_file()
        print(d.evaluate())
    except:
        print('Exception')
        pass