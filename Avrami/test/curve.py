import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def avrami(t, k, n):
    return (1 - np.exp(-k * np.power(t, n)))

#files = ['135C_curve30.csv', '119C_curve30.csv', '112C_curve30.csv', '102C_curve30.csv', '88C_curve30.csv', '43C_curve30.csv',
#         '135C_origin.csv',  '119C_origin.csv',  '112C_origin.csv',  '102C_origin.csv',  '88C_origin.csv',  '43C_origin.csv']
files = ['119C_mod.csv']

f = open("fit_results.txt", 'w')
f.close()

for filename in files:
    data = pd.read_csv(filename)
    
    x_data = np.array(data.loc[:, 'X'])
    x_data = x_data * 10 / max(x_data)
    y_data = np.array(data.loc[:, 'Y']) / 100.
    
    plt.plot(x_data, y_data, 'b-', label='data')
    
    popt, pcov = curve_fit(avrami, x_data, y_data)
    y_pred = avrami(x_data, *popt)

    plt.plot(x_data, y_pred, 'r-', label='fit')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
#    plt.savefig(filename+'.png')
    plt.clf()

    f = open("fit_results.txt", "a")
    f.write("%s : " %filename)
    for var in popt:
        f.write(str(var) + '; ')
    f.write('\n')
    f.close()
