import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = (1 - np.exp(-0.6 * pow(x, 2))) * 100.

plt.plot(x, y, 'r-')
plt.show()

f = open("perfect.dat", 'w')
f.write('X,Y\n')
for xi, yi in zip(x, y):
    f.write(str(xi)+',' + str(yi) + '\n')
