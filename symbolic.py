from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('tmp.csv', sep=',', header=None)
#X_train = np.array([2.935, 3.942, 5.474, 6.902, 9.901, 13.693, 19.657, 29.323]).reshape(-1, 1)
#y_train = np.array([3.89,  8.05,  11.52, 22.87, 62.42, 92.1,   97.34,  99.99 ]) / 100.
X_train = data.drop(labels=1, axis=1).values.reshape(-1, 1)
y_train = data.drop(labels=0, axis=1).values.reshape(-1)

#rng = check_random_state(0)

# training samples
#X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
#y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

# testing samples
#X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
#y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

est_gp = SymbolicRegressor(population_size=10000, 
                           generations=20, tournament_size=50,
                           stopping_criteria=0.01, const_range=(-1.0, 1.0),
                           init_depth=(2, 6), init_method='half and half',
                           function_set=('add', 'sub', 'mul', 'neg', 'log', 'pow2'), 
                           metric='rmse', parsimony_coefficient=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                           p_point_mutation=0.1, p_point_replace=0.05,
                           max_samples=0.9, warm_start=False,
                           n_jobs=1, verbose=1, random_state=0)
                           
"""
Available individual functions are:
     - 'add' : addition, arity=2.
     - 'sub' : subtraction, arity=2.
     - 'mul' : multiplication, arity=2.
     - 'div' : protected division where a denominator near-zero returns 1.,
       arity=2.
     - 'sqrt' : protected square root where the absolute value of the
       argument is used, arity=1.
     - 'log' : protected log where the absolute value of the argument is
       used and a near-zero argument returns 0., arity=1.
     - 'abs' : absolute value, arity=1.
     - 'neg' : negative, arity=1.
     - 'inv' : protected inverse where a near-zero argument returns 0.,
       arity=1.
     - 'max' : maximum, arity=2.
     - 'min' : minimum, arity=2.
     - 'sin' : sine (radians), arity=1.
     - 'cos' : cosine (radians), arity=1.
     - 'tan' : tangent (radians), arity=1.
"""

est_gp.fit(X_train, y_train)
print(est_gp._program)
