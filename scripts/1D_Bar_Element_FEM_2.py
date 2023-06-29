# Imports
import numpy as np
import matplotlib.pyplot as plt

'''
1D Finite Element Analysis of a bar with N-1 elements and N nodes.

This program performs a finite element analysis of a 1D bar subjected to tensile or compressive forces. 
The bar has variable cross-section defined by the function A(x) = A0 * exp(-x), where A0 is the initial cross-section
area.
The analysis calculates the displacements of the nodes and plots the results.
'''


def main():
    def stiffness(e_, i_, E_, A_, dL_):
        K_ = np.zeros((NUMBER_NODES, NUMBER_NODES))
        selection = np.ix_(e_, e_)
        K_[selection] += E_ * A_[i_] / dL_ * np.array([[1, -1], [-1, 1]])
        return K_

    NUMBER_ELEMENTS = 10
    NUMBER_NODES = NUMBER_ELEMENTS + 1
    NODE_CONNECTIONS = [[i, i + 1] for i in range(NUMBER_ELEMENTS)]

    X = np.linspace(0, 1, NUMBER_NODES)
    dL = np.abs(X[1] - X[0])

    E = 2 * 10 ** 9
    A0 = 1
    A = A0 * np.exp(-X)

    K = np.zeros((NUMBER_NODES, NUMBER_NODES))
    U = np.zeros((NUMBER_NODES, 1))
    F = np.zeros((NUMBER_NODES, 1))

    F[-1] = 1000.0

    # Calculate the global stiffness matrix
    for i, e in enumerate(NODE_CONNECTIONS):
        K += stiffness(e, i, E, A, dL)

    PRESCRIBED_NODES = [0]
    ACTIVE_NODES = np.setdiff1d(np.arange(0, NUMBER_NODES), PRESCRIBED_NODES)

    ACTIVE_FORCE = F[ACTIVE_NODES]
    ACTIVE_STIFFNESS = K[np.ix_(ACTIVE_NODES, ACTIVE_NODES)]

    # Solve for nodal displacements
    U_SOLUTION = np.linalg.solve(ACTIVE_STIFFNESS, ACTIVE_FORCE)

    U[ACTIVE_NODES] = U_SOLUTION

    REACTIONS = K.dot(U)

    print("Forces / Reactions: ", REACTIONS)

    # Plotting the results
    plt.plot(X, U)
    plt.title("1D Bar Element - Tensile/Compressive Forces")
    plt.xlabel("X [m]")
    plt.ylabel("U [m]")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
