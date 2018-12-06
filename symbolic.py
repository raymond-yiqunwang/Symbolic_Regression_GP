from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('135C_curve30.csv', sep=',', header=None).values

#X_train = np.array([2.935, 3.942, 5.474, 6.902, 9.901, 13.693, 19.657, 29.323]).reshape(-1, 1)
#y_train = np.array([3.89,  8.05,  11.52, 22.87, 62.42, 92.1,   97.34,  99.99 ]) / 100.
X_train = np.float32(data[1:, 0].reshape(-1, 1))
X_train = X_train * 10 / max(X_train)
y_train = np.float32(data[1:, 1]) / 100.
y_train = 1 - y_train

plt.plot(X_train, y_train, 'r-')
plt.show()

est_gp = SymbolicRegressor(population_size=5000, 
                           generations=20, tournament_size=50,
                           stopping_criteria=0.01, const_range=(0., 0.1),
                           init_depth=(2, 6), init_method='half and half',
                           function_set=('add', 'sub', 'mul', 'neg', 'log'), 
                           metric='rmse', parsimony_coefficient=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                           p_point_mutation=0.1, p_point_replace=0.05,
                           max_samples=1.0, warm_start=False,
                           n_jobs=1, verbose=1, random_state=5)
                           
est_gp.fit(X_train, y_train)
print(est_gp._program)
