# Imports
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Function to calculate stiffness matrix
    def stiffness_calculation(nodes_, l_, E_, A_):
        selection = np.ix_(nodes_, nodes_)
        K[selection] += E_ * A_ / l_ * np.array([[1, -1], [-1, 1]])

    # Constants and parameters
    NUMBER_ELEMENTS = 10
    NUMBER_NODES = NUMBER_ELEMENTS + 1
    X_START = 0.0
    X_END = 1.0

    # Arrays for elements and nodes
    ELEMENTS = np.arange(NUMBER_ELEMENTS)
    NODES = np.arange(NUMBER_NODES)

    # Nodal coordinates and connectivity
    NODAL_COORDINATES = np.linspace(X_START, X_END, NUMBER_NODES)
    NODAL_CONNECTIVITIES = [[i, i + 1] for i in ELEMENTS]

    # Creation of Stiffness, Displacement and Force Matrix
    U = np.zeros((NUMBER_NODES, 1))
    F = np.zeros((NUMBER_NODES, 1))
    K = np.zeros((NUMBER_NODES, NUMBER_NODES))

    # Parameters and Characteristics of the Material
    E = 2 * 10 ** 9  # Young Modulus [Pa]
    A = 1  # Cross Section Area [m^2]
    l = np.abs(X_END - X_START) / NUMBER_ELEMENTS

    # Loop over elements to calculate stiffness matrix
    for i, element in enumerate(NODAL_CONNECTIVITIES):
        stiffness_calculation(element, l, E, A)

    # Force
    F[-1] = 10000.0

    # Prescribed nodes
    PRESCRIBED_NODES = [0]
    ACTIVE_NODES = np.setdiff1d(NODES, PRESCRIBED_NODES)

    # Solve for prescribed displacements
    U_PRESCRIBED = np.linalg.solve(K[np.ix_(ACTIVE_NODES, ACTIVE_NODES)], F[ACTIVE_NODES])

    # Assign prescribed displacements to the global displacement vector
    U[ACTIVE_NODES] = U_PRESCRIBED

    FORCES_REACTIONS = K.dot(U)

    print('Forces: ', FORCES_REACTIONS)

    # Plotting the results
    plt.plot(NODAL_COORDINATES, U)
    plt.xlabel('X [m]')
    plt.ylabel('U [m]')
    plt.title('1D Bar Element - Tensile Force')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
