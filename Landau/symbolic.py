import sys
from gplearn.genetic import SymbolicRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from IPython.display import Image
import pydotplus


def landau(T, theta):
    return (1.696 * (T - 2.057) * theta**2 + 1.71 * 0.01 * theta**4)
 

hyperparams = [
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 20), 500, 5, 25, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 5000, 15, 25, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 5000, 15, 25, 0.05],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 5000, 15, 25, 0.1],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 5000, 15, 10, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 5000, 15, 10, 0.1],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 10000, 15, 50, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 10000, 15, 50, 0.05],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 10000, 15, 50, 0.1],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 10000, 15, 20, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 10000, 15, 20, 0.05],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 10000, 15, 20, 0.1],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 20000, 15, 100, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 20000, 15, 100, 0.05],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 20000, 15, 100, 0.1],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 20000, 15, 40, 0.02],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 20000, 15, 40, 0.05],
    [np.linspace(0, 10, 11), np.linspace(-10, 10, 200), 20000, 15, 40, 0.1],
    [np.linspace(0, 1, 20), np.linspace(-15, 15, 200), 10000, 15, 50, 0.05],
    [np.linspace(0, 1, 20), np.linspace(-15, 15, 200), 10000, 15, 20, 0.02],
    [np.linspace(0, 1, 20), np.linspace(-15, 15, 200), 10000, 15, 20, 0.05],
    [np.linspace(0, 1, 20), np.linspace(-15, 15, 200), 10000, 15, 20, 0.1]
]

forest = []

for params in hyperparams:
    T_range = params[0]
    theta_range = params[1]
    pop_size = params[2]
    gen_size = params[3]
    tour_size = params[4]
    pars_coeff = params[5]
    
    data = []
    for T in T_range:
        for theta in theta_range:
            E = landau(T, theta)
            data.append([T, theta, E])
    data = pd.DataFrame(data, index=None, columns=['T', 'theta', 'E'])
    X_data = data.drop('E', axis=1).values.astype(float)
    y_data = data['E'].values.astype(float)

        
                    
    print('')
    print('')
    line_format = '{:>16}{:>16} {:>16} {:>16} {:>16} {:>16}'
    print(line_format.format('T_range','theta_range','pop_size', 'gen_size', 'tour_size', 'pars_coeff'))
    print(line_format.format(str(min(T_range))+"~"+str(max(T_range)), str(min(theta_range))+"~"+str(max(theta_range)), 
                             str(pop_size), str(gen_size), str(tour_size), str(pars_coeff)))
    print('')
    print('')

    est_gp = SymbolicRegressor(population_size=pop_size,
                   generations=gen_size, tournament_size=tour_size,
                   stopping_criteria=0.0, const_range=(-2, 2),
                   init_depth=(4, 8), init_method='half and half',
                   function_set=('add', 'sub', 'mul'),
                   metric='rmse', parsimony_coefficient=pars_coeff,
                   p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                   p_point_mutation=0.1, p_point_replace=0.05,
                   max_samples=0.9, warm_start=False, low_memory=True,
                   n_jobs=8, verbose=1, random_state=0)

    est_gp.fit(X_data, y_data.ravel())
#    print('printing the best individuals for the last 10 generations:')
    tree = [str(min(T_range))+"~"+str(max(T_range)), str(min(theta_range))+"~"+str(max(theta_range)), pop_size, gen_size, tour_size, pars_coeff]
    for p in est_gp._final_programs[-5:]:
        tree.append(p.export_graphviz())
    forest.append(tree)
    print('Age of extinction, kill all.. Start new generation..')
    print('')
    sys.stdout.flush()
    break


cols = ['T_range','theta_range','pop_size', 'gen_size', 'tour_size', 'pars_coeff', 'tree0', 'tree1', 'tree2', 'tree3', 'tree4']
forest = pd.DataFrame(forest, index=None, columns=None)
forest.to_csv('forest.csv', index=None, header=cols, sep=';')
