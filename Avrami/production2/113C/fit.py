import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def avrami(t, k, n):
    return (1 - np.exp(-k * np.power(t, n)))


filename = '113C_origin.csv'
data = pd.read_csv(filename)

scale_X = 1
x_data = np.array(data.loc[:, 'X'])
if scale_X is not None:
    x_data = x_data * scale_X / max(x_data)
y_data = np.array(data.loc[:, 'Y']) / 100.

plt.plot(x_data, y_data, 'b*', label='data')

popt, pcov = curve_fit(avrami, x_data, y_data)
y_pred = avrami(x_data, *popt)

plt.plot(x_data, y_pred, 'r-', label='fit')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
#plt.show()

for var in popt:
    print(str(var) + '; ')
