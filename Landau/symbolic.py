import sys
from gplearn.genetic import SymbolicRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from IPython.display import Image
import pydotplus

def landau(T, theta):
    return (1.696 * (T - 2.057) * theta**2 + 1.71 * 0.01 * theta**4)

T_range = np.linspace(0, 1, 11)
theta_range = np.linspace(-20, 20, 100)
pop_size = 20000
gen_size = 15
tour_size = 25
pars_coeff = 0.02

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
               n_jobs=4, verbose=1, random_state=0)

est_gp.fit(X_data, y_data.ravel())
print('printing the best individuals for the last 10 generations:')
forest = []
for p in est_gp._final_programs[-5:]:
    print(p)
    print('='*70)
print('Age of extinction, kill all.. Start new generation..')
print('')
sys.stdout.flush()


