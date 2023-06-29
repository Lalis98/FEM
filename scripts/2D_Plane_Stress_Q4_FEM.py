# Imports 
import numpy as np
import matplotlib.pyplot as plt

'''
The rectangular stress plane represents a two-dimensional (2D) mesh composed of NxN elements. This mesh is used in 
Finite Element Method (FEM) to analyze and calculate stress distribution in a structural component or material.

To perform the analysis on the stress plane mesh, the governing equations, such as the equilibrium equations or the 
elasticity equations, are discrete using the Finite Element Method.
         
    The Shape is a rectangular beam
    The Geometry is created by determining the following:
    
            +-----------------------------------------(X_END,Y_END)
            |                                               |
            |                                               |
            |                                               |
            |                                               |
            |                                               |
            |                                               |
    (X_START,Y_START)---------------------------------------+
    

Example of discretize the domain, with Elements and nodes. 

   31----32----33----34----35----36----37----38----39----40-----41
    |     |     |     |     |     |     |     |     |     |     |
    |     |     |     |     |     |     |     |     |     |     |
   21----22----23----24----25----26----27----28----29----30-----31
    |     |     |     |     |     |     |     |     |     |     |
    |     |     |     |     |     |     |     |     |     |     |
   11----12----13----14----15----16----17----18----19----20-----21
    |     |     |     |     |     |     |     |     |     |     |
    |     |     |     |     |     |     |     |     |     |     |
    0-----1-----2-----3-----4-----5-----6-----7-----8-----9-----10

By inputting local coordinates of the quadrilateral 4-node shape, this function returns shape function and natural
derivative of shape function as:

• Shape Function:

              [(1 - ξ)(1 - η)] 
    Ν = 1/4 * [(1 + ξ)(1 - η)]
              [(1 + ξ)(1 - η)]
              [(1 - ξ)(1 + η)]
              
• Natural Derivative of Shape Function

    dN = [dN1/dξ,...,dN4/dξ]
         [dN1/dη,...,dN4/dη]

• Jacobian Matrix:

                           [x1 y1] 
    J = [dN1/dξ,...,dN1/dξ][...  ]
        [dN1/dη,...,dN1/dη][...  ]
                           [x4 y4]
                           
• Matrix B:

        [dN1/dx   0    | ... | dN4/dx   0   ]
    B = [  0    dN1/dy | ... |    0   dN4/dy]    
        [dN1/dy dN1/dx | ... | dN4/dy dN4/dx]
'''


def main():
    # -----------------------------------------------FUNCTIONS-----------------------------------------------------------
    def shapeFunctionQ4(xi_, eta_):
        N1 = 1 / 4 * (1 - xi_) * (1 - eta_)
        N2 = 1 / 4 * (1 + xi_) * (1 - eta_)
        N3 = 1 / 4 * (1 + xi_) * (1 + eta_)
        N4 = 1 / 4 * (1 - xi_) * (1 + eta_)

        dN1 = 1 / 4 * np.array([-(1 - eta_), -(1 - xi_)])
        dN2 = 1 / 4 * np.array([1 - eta_, -(1 + xi_)])
        dN3 = 1 / 4 * np.array([1 + eta_, 1 + xi_])
        dN4 = 1 / 4 * np.array([-(1 + eta_), (1 - xi_)])

        N_ = np.array([
            [N1],
            [N2],
            [N3],
            [N4]
        ])

        dN_ = np.array([
            dN1,
            dN2,
            dN3,
            dN4
        ])

        return N_, dN_.T

    def Jacobian(natural_derivative, xy_coordinates):
        J_ = natural_derivative.dot(xy_coordinates)  # Multiply Natural derivative with global x & y coordinates matrix

        # Elements of Jacobian matrix
        J11 = J_[0, 0]
        J12 = J_[0, 1]
        J21 = J_[1, 0]
        J22 = J_[1, 1]

        # Calculate Determinant of Jacobian Matrix
        detJ = 1 / (J11 * J22 - J12 * J21)

        # Finding Inverse Jacobian Matrix
        invJ = detJ * np.array([
            [J22, J11],
            [-J21, -J12]
        ])

        # Return from function Jacobian Matrix, Inverse Jacobian Matrix and Determinant of Jacobian Matrix
        return J_, invJ, detJ

    # In this function we take the global coordinates of each element

    def get_xy_coords_from_element(element):
        e1, e2, e3, e4 = element  # e1, e2, e3, e4 are the nodes of the quadrilateral element. The counting starts
        # anticlockwise from the south-west node

        xy = np.array([
            [COORDINATES[e1][0], COORDINATES[e1][1]],  # Coordinates (x, y) of Node 1
            [COORDINATES[e2][0], COORDINATES[e2][1]],  # Coordinates (x, y) of Node 2
            [COORDINATES[e3][0], COORDINATES[e3][1]],  # Coordinates (x, y) of Node 3
            [COORDINATES[e4][0], COORDINATES[e4][1]]  # Coordinates (x, y) of Node 4
        ])

        # Return (x,y) coordinates in the form of matrix 4x2
        return xy

    def gaussQuadrature(option):
        # 2nd order degree 4-node Gauss Quadrature
        if option == 0:

            # Gauss-Points Coordinates of (ξ,η)
            xi_eta = np.array([
                [-np.sqrt(1 / 3), -np.sqrt(1 / 3)],
                [np.sqrt(1 / 3), -np.sqrt(1 / 3)],
                [-np.sqrt(1 / 3), np.sqrt(1 / 3)],
                [np.sqrt(1 / 3), np.sqrt(1 / 3)]])

            w = np.array([1, 1, 1, 1])  # Weight of Gauss-Points

        # 2nd order degree 4-node Gauss Quadrature
        else:
            xi_eta = np.array([[0, 0]])
            w = np.array([4])

        # Return (ξ,η) coordinates and their weights
        return xi_eta, w

    def createMatrixB(dN_):
        B_ = np.zeros((3, DEGREE_OF_FREEDOM * NODES_IN_ELEMENT))  # Creating Matrix B 3x8

        B_[0, 0::2] = dN_[0, :]  # row 1 of Matrix B
        B_[1, 1::2] = dN_[1, :]  # row 2 of Matrix B
        B_[2, 0::2] = dN_[1, :]  # row 3, odd columns of Matrix B
        B_[2, 1::2] = dN_[0, :]  # row 3, even columns of Matrix B

        # Return Matrix B
        return B_

    # ------------------------------------------------MATERIAL CHARACTERISTICS------------------------------------------

    E = 200 * 10 ** 9  # Young's Modulus [Pa]
    POISSON = 0.3  # Poisson's ratio
    THICKNESS = 0.5  # Thickness of the elements by z-axis

    # -------------------------------------------PROBLEM DEFINITION-----------------------------------------------------

    # C Matrix of Plane Stress Problem
    C = E / (1 - POISSON ** 2) * np.array([
        [1, POISSON, 0],
        [POISSON, 1, 0],
        [0, 0, (1 - POISSON) / 2]
    ])

    # C Matrix of Plane Strain Problem
    # C = E / (1.0 + POISSON) / (1.0 - 2.0 * POISSON) * np.array([[1.0 - POISSON, POISSON, 0.0],
    #                                                 [POISSON, 1.0 - POISSON, 0.0],
    #                                                 [0.0, 0.0, 0.5 - POISSON]])

    # NOTE: You can uncomment the desired C matrix configuration by removing the '#' symbol at the beginning of the
    # corresponding lines. This allows you to choose the specific behavior or condition you want for your problem.

    NODES_IN_ELEMENT = 4  # Number of nodes in each element
    NUMBER_NODES_X = 6  # Number of Nodes across x-axis
    NUMBER_NODES_Y = 6  # Number of Nodes across y-axis
    NUMBER_ELEMENTS_X = NUMBER_NODES_X - 1  # Number of Elements across x-axis
    NUMBER_ELEMENTS_Y = NUMBER_NODES_Y - 1  # Number of Elements across y-axis
    NUMBER_NODES = NUMBER_NODES_X * NUMBER_NODES_Y  # Total Number of Nodes
    DEGREE_OF_FREEDOM = 2  # Degree of Freedoms of each element

    X_START = 0  # x-axis starting x-coordinate of the rectangular element
    Y_START = 0  # y-axis starting y-coordinate of the rectangular element
    X_END = 5  # x-axis ending x-coordinate of the rectangular element
    Y_END = 1  # y-axis ending y-coordinate of the rectangular element

    NODAL_COORDINATES_X = np.linspace(X_START, X_END, NUMBER_NODES_X)  # Coordinates Range for x
    NODAL_COORDINATES_Y = np.linspace(Y_START, Y_END, NUMBER_NODES_Y)  # Coordinates Range for y

    COORDINATES = np.array(
        [[i, j] for j in NODAL_COORDINATES_Y for i in NODAL_COORDINATES_X])  # Coordinates of all nodes

    CONNECTIVITIES = []  # Create a Matrix CONNECTIONS were there are connections of all nodes of each element
    for j in range(NUMBER_ELEMENTS_Y):
        for i in range(NUMBER_ELEMENTS_X):
            n1 = i + NUMBER_NODES_Y * j  # Node 1
            n2 = i + 1 + NUMBER_NODES_Y * j  # Node 2
            n3 = i + 1 + NUMBER_NODES_Y * (j + 1)  # Node 3
            n4 = i + NUMBER_NODES_Y * (j + 1)  # Node 4

            CONNECTIVITIES.append([n1, n2, n3, n4])

    CONNECTIVITIES = np.array(CONNECTIVITIES)

    # -----------------------------------------------MATRICES CREATION--------------------------------------------------

    U = np.zeros((DEGREE_OF_FREEDOM * NUMBER_NODES, 1))  # Displacement Matrix
    F = np.zeros((DEGREE_OF_FREEDOM * NUMBER_NODES, 1))  # Force Matrix
    K = np.zeros((DEGREE_OF_FREEDOM * NUMBER_NODES, DEGREE_OF_FREEDOM * NUMBER_NODES))  # Stiffness Matrix

    # -------------------------------------------STIFFNESS MATRIX CREATION----------------------------------------------

    for c, connection in enumerate(CONNECTIVITIES):  # Run through connectivities of each element

        xy_coord = COORDINATES[connection]  # xy_coord gives the 4-nodes (x,y) coordinates of each element
        ke = np.zeros((DEGREE_OF_FREEDOM * NODES_IN_ELEMENT, DEGREE_OF_FREEDOM * NODES_IN_ELEMENT))

        gauss_locations = gaussQuadrature(0)[0]  # Call from function gauss_quadrature, the Gauss-Points
        gauss_weights = gaussQuadrature(0)[1]  # Gauss-Point Weights

        for g, gauss_locs in enumerate(gauss_locations):  # Run through local (ξ,η) Gauss-Points coordinates
            xi, eta = gauss_locs  # Assign (ξ,η) of each node

            N, dN_xieta = shapeFunctionQ4(xi, eta)  # Create Shape function and Natural Derivative Shape Function

            J, J_inv, det_J = Jacobian(dN_xieta, xy_coord)  # By multiplying Natural Derivative Shape Function and (x,y)
            # Matrix we can find Jacobian Matrix, Inverse Jacobian Matrix and Determinant of Jacobian Matrix

            dN_xy = J_inv.dot(dN_xieta)  # Calculation of Shape Function derivatives (to physical  x,y not natural η,ξ)
            B = createMatrixB(dN_xy)  # Creation of B Matrix

            ke += np.dot(np.dot(B.T, C), B) * det_J * gauss_weights[g] * THICKNESS  # Element Stiffness Matrix, due to
            # specif Gauss-Point is given by ke = B.T * C * B * |J| * Wi * THICKNESS

        # Having element Stiffness Matrix ke, we have to construct Global Stiffness Matrix by adding ke where it must
        for i, ii in enumerate(connection):  # Runs through all elements nodes of the connectivity
            for j, jj in enumerate(connection):  # Runs through all element nodes, again, of the connectivity
                # ke is 8x8 (8 = 2Dof * 4-node)

                # i: index of first for-loop (Local ke index)
                # j: index of second for-loop (Local ke index)
                # ii: node of first for-loop (Global K index)
                # jj: node of second for-loop (Global K index)

                K[2 * ii, 2 * jj] += ke[2 * i, 2 * j]  # Ke11
                K[2 * ii + 1, 2 * jj] += ke[2 * i + 1, 2 * j]  # Ke21
                K[2 * ii, 2 * jj + 1] += ke[2 * i, 2 * j + 1]  # Ke12
                K[2 * ii + 1, 2 * jj + 1] += ke[2 * i + 1, 2 * j + 1]  # Ke22

    # ---------------------------------------------LOADS & BOUNDARY CONDITIONS------------------------------------------

    # Loading
    P = 1000.0  # Force [N]
    F[1] = -P  # Location of the Force (minus symbol means it shows downwards-negative)

    # Prescribed Nodes are essential for solving K * x = b. R.H.S. nodes are prescribed us the boundary condition, were
    # the (u,v) are equal to (0,0) for all the nodes. PRESCRIBED_NODES gives us the boundary nodes.
    PRESCRIBED_NODES = np.array([
        [2 * i, 2 * i + 1] for i in range((NUMBER_NODES_X - 1), NUMBER_NODES, NUMBER_NODES_X)
    ]).flatten()

    # ACTIVE_NODES are the nodes that are NOT PRESCRIBED_NODES
    ACTIVE_NODES = np.setdiff1d(np.arange(0, DEGREE_OF_FREEDOM * NUMBER_NODES), PRESCRIBED_NODES)

    # Solving system K * U = F, of only active nodes
    U_PRESCRIBED = np.linalg.solve(K[np.ix_(ACTIVE_NODES, ACTIVE_NODES)], F[ACTIVE_NODES])

    # Assign Active nodes to the U Matrix, so the prescribed nodes remain (u,v)=(0,0)
    U[ACTIVE_NODES] = U_PRESCRIBED

    # Find (u,v) coordinates of each node
    # the U Matrix goes like: U = [u1, v1, u2, v2,..., uN, vN].T
    UU = U[0::2]  # u displacement of each node
    VV = U[1::2]  # v displacement of each node

    # x,y and resultant displacement
    XVEC = []
    YVEC = []
    RES = []

    UU = np.reshape(UU, (NUMBER_NODES_X, NUMBER_NODES_Y))  # Reshape Uu by Nx * Ny
    VV = np.reshape(VV, (NUMBER_NODES_X, NUMBER_NODES_Y))  # Reshape Uv by Nx * Ny

    # Find final (x,y) coordinates, after cumulating the displacements
    for j, J in enumerate(NODAL_COORDINATES_Y):
        for i, I in enumerate(NODAL_COORDINATES_X):
            XVEC.append(I + UU[i, j])
            YVEC.append(J + VV[i, j])
            RES.append(VV[i, j])

    XVEC = np.array(XVEC)
    YVEC = np.array(YVEC)
    RES = np.array(RES)

    # Reshape them in order to plot them
    XVEC = np.reshape(XVEC, (NUMBER_NODES_X, NUMBER_NODES_Y))
    YVEC = np.reshape(YVEC, (NUMBER_NODES_X, NUMBER_NODES_Y))

    # Create again X, Y coordinates in order to plot them
    X, Y = np.meshgrid(NODAL_COORDINATES_X, NODAL_COORDINATES_Y)

    # Reshape them in order to plot them
    UU = np.reshape(UU, (NUMBER_NODES_X, NUMBER_NODES_Y))
    VV = np.reshape(VV, (NUMBER_NODES_X, NUMBER_NODES_Y))

    plt.imshow(VV, cmap='coolwarm', extent=[XVEC.min(), XVEC.max(), YVEC.min(), YVEC.max()])
    plt.xlim(X_START - 1, X_END + 1)  # Set the x-axis limits
    plt.ylim(Y_START + 1, Y_END + 1)  # Set the y-axis limits
    plt.grid()
    plt.colorbar(label='Displacement [m]')  # Add colorbar legend with a label
    plt.title('2D Plane Stress Analysis')  # Set the title of the plot
    plt.xlabel('X [m]')  # Set the x-label
    plt.ylabel('Y [m]')  # Set the y-label
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
