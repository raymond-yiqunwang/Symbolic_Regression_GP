import numpy as np

def landau(T, theta):
    return (1.696 * (T - 2.057) * theta**2 + 1.71 * 0.01 * theta**4)


print("Theta;T;Energy")
for T in np.linspace(0, 10, 11):
    for theta in np.linspace(-15, 15, 300):
        E = landau(T, theta)
        print("{:.2f};{:.2f};{:.6f}".format(theta, T, E))
