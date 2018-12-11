import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 1000)
y = 1 - np.exp(-0.6 * pow(x, 2))

plt.plot(x, y, 'r-')
plt.show()

f = open("k0d6n2_5.dat", 'w')
f.write('X,Y\n')
for xi, yi in zip(x, y):
    f.write(str(xi)+',' + str(yi) + '\n')
