import sys
from gplearn.genetic import SymbolicRegressor
#from sklearn.utils.random import check_random_state
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#data = pd.read_csv("./landau_data.csv", sep=';', header=0)
#X_data = data.drop('Energy', axis=1).values.astype(float)
#y_data = data['Energy'].values.astype(float)

# hyperparameters
Lpop_size = [5000, 10000, 20000]
Ltour_factor = [200, 500]                 # tour_size = pop_size / tour_factor
Lpars_coeff = [0.02, 0.05, 0.1, 'auto']

def landau(T, theta):
    return (1.696 * (T - 2.057) * theta**2 + 1.71 * 0.01 * theta**4)
 
T_range = [np.linspace(0, 10, 11), np.linspace(0, 1, 20), np.linspace(0, 2, 40)]
Theta_range = [np.linspace(-15, 15, 200), np.linspace(-20, 20, 200), np.linspace(-10, 10, 200)]
for t_range in  T_range:
    for theta_range in Theta_range:
        data = []
        for T in t_range:
            for theta in theta_range:
                E = landau(T, theta)
                data.append([theta, T, E])
        data = pd.DataFrame(data, index=None, columns=['theta', 'T', 'E'])
        X_data = data.drop('E', axis=1).values.astype(float)
        y_data = data['E'].values.astype(float)

        for pop_size in Lpop_size:
            for tour_factor in Ltour_factor:
                for pars_coeff in Lpars_coeff:
        
                    gen_size = 15
                    tour_size = int(pop_size / tour_factor)
                    
                    print('')
                    print('')
                    line_format = '{:>16}{:>16}{:>16} {:>16} {:>16} {:>16} {:>16}'
                    print(line_format.format('T_range','theta_range','pop_size', 'gen_size', 'tour_size', 'tour_factor', 'pars_coeff'))
                    print(line_format.format(str(min(t_range))+"~"+str(max(t_range)), str(min(theta_range))+"~"+str(max(theta_range)), 
                                             str(pop_size), str(gen_size), str(tour_size), str(tour_factor), str(pars_coeff)))
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
                    print('printing the best individuals for the last 10 generations:')
                    for p in est_gp._final_programs[-5:]:
                        print(p)
                        print('='*70)
                    print('Age of extinction, kill all.. Start new generation..')
                    print('')
                    sys.stdout.flush()
