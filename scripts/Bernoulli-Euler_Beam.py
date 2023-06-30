# Imports
import numpy as np
import plotly.graph_objects as go

'''

FEM Beam Analysis - Euler-Bernoulli Theory (1D)

This script performs a finite element analysis for a 1D beam using the Euler-Bernoulli theory. It calculates the 
deflection of the beam by indiscretion it into smaller elements and solving the governing equations.

The straight beam element has a constant cross-sectional area. By applying principles from simple beam theory, we'll 
derive the stiffness matrix. K beam problem solution reveals node-associated degrees of freedom, including transverse 
displacement and rotation.

------------------------------------------------------------------------------------------------------------------------
Problem Statement:
-----------------
        y, v(x) 
         ˄
         |
         |   
         ┌─────────────────────────────────────────────────┐          ┌──┐
       ──│• ── • ── • ── • ── • ── • ── •── • ── • ──  • ──│• ── • ── │• │──> x
         └─────────────────────────────────────────────────┘          └──┘
         O                                                 O            
         ↑                                                 ↑
Apply Forces and Beams to the beam, in order to calculate the Deflection of the beam.

References:
----------
https://www.12000.org/my_notes/stiffness_matrix/stiffness_matrix_report.htm

'''


def main():
    def local_stiffness_matrix(p1_, p2_, E_, I_):
        L_ = np.linalg.norm(p1_ - p2_)
        ke_ = E_ * I_ / L_ ** 3 * np.array([
            [12, 6 * L_, -12, 6 * L_],
            [6 * L_, 4 * L_ ** 2, - 6 * L_, 2 * L_ ** 2],
            [- 12, - 6 * L_, 12, - 6 * L_],
            [6 * L_, 2 * L_ ** 2, - 6 * L_, 4 * L_ ** 2]
        ])

        return ke_

    X_START = 0  # Starting x-coordinate of the beam [m]
    X_END = 5  # Ending x-coordinate of the beam [m]

    NUMBER_OF_ELEMENTS = 21  # Number of Elements
    NUMBER_OF_NODES = NUMBER_OF_ELEMENTS + 1  # Number of Nodes
    DOF = 2  # Degrees of Freedom
    POINTS = np.linspace(X_START, X_END, NUMBER_OF_NODES)  # Points Coordinates
    ELEMENTS = np.arange(NUMBER_OF_ELEMENTS)  # Elements' Indices
    CONNECTIONS = np.array([[i, i + 1] for i in ELEMENTS])  # Node Connectivities
    NODES_DOF = np.arange(NUMBER_OF_NODES * DOF)  # Nodes' Degrees of Freedom

    CONNECTIONS_DOF = np.array([
        [i[0] * DOF, i[0] * DOF + 1,  # (v1, phi1)
         i[1] * DOF, i[1] * DOF + 1]  # (v2, phi2)
        for i in CONNECTIONS])

    E = 200e9  # Young's Modulus for Steel
    WIDTH = 0.1  # Width of the Cross-Section of the Beam [m]
    HEIGHT = 0.1  # Height of the Cross-Section of the Beam [m]
    I = WIDTH * HEIGHT ** 3 / 12  # Sectional Moment of Inertia [m^4]

    K = np.zeros((NUMBER_OF_NODES * DOF, NUMBER_OF_NODES * DOF))  # Construction of Stiffness Matrix K
    U = np.zeros(NUMBER_OF_NODES * DOF)  # Construction of Displacement Matrix U
    F = np.zeros(NUMBER_OF_NODES * DOF)  # Construction of Force Matrix F

    F[1] = 1000  # Apply the Force / Moment at a Degree of Freedom of our choice
    F[-1] = -1000
    # Run through all connections of nodes to create Stiffness Matrix
    for c, connection in enumerate(CONNECTIONS):
        n1, n2 = connection  # Nodes 1 & 2

        selection = np.ix_(CONNECTIONS_DOF[c], CONNECTIONS_DOF[c])  # Find the places of the connection nodes
        ke = local_stiffness_matrix(POINTS[n1], POINTS[n2], E, I)  # Create local Stiffness Matrix from the function

        K[selection] += ke  # Apply it to global Stiffness Matrix K

    PRESCRIBED_DOF = np.array([0, NODES_DOF[-2]])  # Prescribed DOFs
    ACTIVE_DOF = np.setdiff1d(NODES_DOF, PRESCRIBED_DOF)  # Active DOFs
    K_ACTIVE = K[np.ix_(ACTIVE_DOF, ACTIVE_DOF)]  # Active Stiffness Matrix

    U_PRESCRIBED = np.linalg.solve(K_ACTIVE, F[ACTIVE_DOF])  # U solved without prescribed DOFs

    U[ACTIVE_DOF] = U_PRESCRIBED  # Apply the solution to U matrix

    V = U[0::2]  # v displacement of each node
    PHI = U[1::2]  # phi rotation of each node

    FORCES = K.dot(U)  # All Forces-Moments & Reactions in each node

    # Plot the results
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=POINTS,  # x-axis is the Beam axis
        y=V,  # y-axis is the displacement v of the beam from the initial position

        mode='lines'
    ))

    fig.update_layout(
        title='Euler-Bernoulli Beam Theory',
        xaxis_title='X [m]',
        yaxis_title='Deflection [m]'
    )

    fig.show()


if __name__ == "__main__":
    main()
