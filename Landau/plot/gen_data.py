import numpy as np

def landau(theta, T):
    return (1.696*(T - 2.057)*theta**2 + 0.0171*theta**4)

def gpsr(theta, T):
    A = 1.983 * T * theta**2 -(3.756 + 0.0165 * T**2) * theta**2 + 0.0165 * theta**4
    B = (1.214 * T - 1.72 * T**2 - 0.24) * theta - 1.334
    return (A + B)

T_range = [0., 0.5, 1., 2., 3., 4.]
for T in T_range:
    for theta in np.linspace(-20, 20, 200):
        E = landau(theta, T)
        print("{:2f}  {:4f}".format(theta, E))
    print('')
    print('')
    for theta in np.linspace(-20, 20, 20):
        E = gpsr(theta, T)
        print("{:2f}  {:4f}".format(theta, E))
    print('')
    print('')
