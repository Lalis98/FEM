# Imports
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ==================================================================================================================
    # ------------------------------------------------SHAPE FUNCTIONS---------------------------------------------------
    # ==================================================================================================================

    def elementXY_Coord(element_, nodal_coords):
        xy_coord_ = nodal_coords[element_]

        return xy_coord_

    def nodeXY_Coord(node_, nodal_coords):
        xy_coord_ = nodal_coords[node_]

        return xy_coord_

    def distanceBetween_AB(xA_, yA_, xB_, yB_):
        return np.sqrt((xB_ - xA_) ** 2 + (yB_ - yA_) ** 2)

    def forceVector(nodes_, nodal_traction_force_, traction_force_, nodal_coords):

        for i_, I_ in enumerate(nodal_traction_force_):

            xA_, yA_ = nodeXY_Coord(nodes_[i_][0], nodal_coords)
            xB_, yB_ = nodeXY_Coord(nodes_[i_][1], nodal_coords)

            li = distanceBetween_AB(xA_, yA_, xB_, yB_)

            F[I_] += li * traction_force_ / 2

        return F

    # Shape Function, of (ξ,η), for triangular Shape with 3 nodes
    def shapeFunctionT3(xi_, eta_):
        shape_function_ = np.array([
            [1 - xi_ - eta_],  # N1
            [xi_],  # N2
            [eta_]  # N3
        ])

        derivative_shape_function_ = np.array([
            [-1, 1, 0],  # [dN1/dξ, dN2/dξ, dN3/dξ]
            [-1, 0, 1]  # [dN1/dη, dN2/dη, dN3/dη]
        ])

        return shape_function_, derivative_shape_function_

    def gaussQuadratureT3(option_):
        if option_ == 1:  # 3 Gauss Points = option 1

            locations_ = np.array([[0.5, 0.5],
                                   [0, 0.5],
                                   [0.5, 0]
                                   ])
            weights_ = 1 / 6 * np.ones((3, 1))

        elif option_ == 2:  # 3 Gauss Point = option 2

            locations_ = np.array([[1 / 6, 1 / 6],
                                   [2 / 3, 1 / 6],
                                   [1 / 6, 2 / 3]
                                   ])
            weights_ = 1 / 6 * np.ones((3, 1))

        elif option_ == 3:  # 4 Gauss Point = option 3

            locations_ = np.array([[1 / 3, 1 / 3],
                                   [1 / 5, 1 / 5],
                                   [3 / 5, 1 / 5],
                                   [1 / 5, 3 / 5]
                                   ])
            weights_ = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96])

        else:  # 1 Gauss Point

            locations_ = np.array([[1 / 3, 1 / 3]])
            weights_ = 1 / 2 * np.ones((1, 1))

        return locations_, weights_

    def Jacobian(derivative_shape_function_, xy_coordinates_):
        J_ = derivative_shape_function_.dot(xy_coordinates_)

        invJ_ = np.linalg.inv(J_)
        detJ_ = np.linalg.det(J_)

        return J_, invJ_, detJ_

    def createMatrixB(DerivativeShapeFunction_xy_, dof, number_nodes_in_element):
        B_ = np.zeros((3, dof * number_nodes_in_element))  # Creating Matrix B 3x8

        B_[0, 0::2] = DerivativeShapeFunction_xy_[0, :]  # row 1 of Matrix B
        B_[1, 1::2] = DerivativeShapeFunction_xy_[1, :]  # row 2 of Matrix B
        B_[2, 0::2] = DerivativeShapeFunction_xy_[1, :]  # row 3, odd columns of Matrix B
        B_[2, 1::2] = DerivativeShapeFunction_xy_[0, :]  # row 3, even columns of Matrix B

        # Return Matrix B
        return B_

    # ==================================================================================================================
    # -----------------------------------------NODES AND DEGREES OF FREEDOM---------------------------------------------
    # ==================================================================================================================

    DEGREE_OF_FREEDOM = 2  # 2 degrees of freedom, u and v displacements
    NUMBER_NODES_IN_ELEMENT = 3  # Triangular element, so 3 nodes in each element

    # ==================================================================================================================
    # --------------------------------------------GEOMETRY OF THE PROBLEM-----------------------------------------------
    # ==================================================================================================================

    X_START = 0  # x-axis starting x-coordinate of the rectangular element
    Y_START = 0  # y-axis starting y-coordinate of the rectangular element
    X_END = 3  # x-axis ending x-coordinate of the rectangular element
    Y_END = 3  # y-axis ending y-coordinate of the rectangular element
    H = 0.5  # width of the plate, in [m]

    NUMBER_NODES_X = 51  # Number of Nodes at x-axis
    NUMBER_NODES_Y = 51  # Number of Nodes at y-axis

    NUMBER_NODES = NUMBER_NODES_X * NUMBER_NODES_Y  # Total Number of Nodes
    NODES = np.arange(NUMBER_NODES)  # Indices of all Nodes

    NODES_RESHAPED = np.reshape(NODES, (NUMBER_NODES_Y, NUMBER_NODES_X))  # Reshaped Nodes into Ny x Nx

    # ======================================================================================================================
    # -----------------------------------------------CONNECTIONS---------------------------------------------------------
    # ======================================================================================================================

    CONNECTIVITIES = []  # Create Matrix of Connectivities

    # In the next for-loop, Connectivities of triangle shapes will be created
    for i, ii in enumerate(NODES_RESHAPED):  # runs the rows of the rectangular beam
        for n, node in enumerate(ii):  # Runs the nodes of each row

            if i == (NUMBER_NODES_Y - 1):  # If it is the last row, break
                break
            elif n == (NUMBER_NODES_X - 1):  # If node is the last element of the row, pass
                pass
            else:
                CONNECTIVITIES.append([node, node + 1, NODES_RESHAPED[i + 1][n]])  # Add the first triangle shape
                CONNECTIVITIES.append([node + 1, NODES_RESHAPED[i + 1][n + 1], NODES_RESHAPED[i + 1][n]])  # Add the
                # second triangle shape

    CONNECTIVITIES = np.array(CONNECTIVITIES)  # Convert CONNECTIONS into array

    # ==================================================================================================================
    # ----------------------------------------------------MESH----------------------------------------------------------
    # ==================================================================================================================

    NODAL_COORDINATES_X = np.linspace(X_START, X_END, NUMBER_NODES_X)  # Create Nodal Coordinates of a x-axis row
    NODAL_COORDINATES_Y = np.linspace(Y_START, Y_END, NUMBER_NODES_Y)  # Create Nodal Coordinates of a y-axis column

    NODAL_COORDINATES = np.array([[i, j] for j in NODAL_COORDINATES_Y for i in NODAL_COORDINATES_X])  # Create the (x,y)
    # coordinates of each node, in an array

    # ==================================================================================================================
    # --------------------------------------------INITIALIZING MATRICES-------------------------------------------------
    # ==================================================================================================================

    K = np.zeros((NUMBER_NODES * DEGREE_OF_FREEDOM, NUMBER_NODES * DEGREE_OF_FREEDOM))  # Stiffness Matrix
    U = np.zeros((NUMBER_NODES * DEGREE_OF_FREEDOM, 1))  # Displacement Matrix
    F = np.zeros((NUMBER_NODES * DEGREE_OF_FREEDOM, 1))  # Force Matrix

    # ==================================================================================================================
    # -------------------------------------------MATERIAL CHARACTERISTICS-----------------------------------------------
    # ==================================================================================================================

    E = 200 * 10 ** 9  # Young's Modulus of Steel, in [Pa]
    POISSON = 0.33  # Poison's ratio for Steel

    # C Matrix of Plane Stress Problem
    C = E / (1 - POISSON ** 2) * np.array(
        [
            [1, POISSON, 0],
            [POISSON, 1, 0],
            [0, 0, (1 - POISSON) / 2]
        ])

    # Plane Strain Problem
    # C = E / (1.0 + POISSON) / (1.0 - 2.0 * POISSON) * np.array([[1.0 - POISSON, POISSON, 0.0],
    #                                                 [POISSON, 1.0 - POISSON, 0.0],
    #                                                 [0.0, 0.0, 0.5 - POISSON]])

    # NOTE: You can uncomment the desired C matrix configuration by removing the '#' symbol at the beginning of the
    # corresponding lines. This allows you to choose the specific behavior or condition you want for your problem.

    # ==================================================================================================================
    # ------------------------------------------STIFFNESS MATRIX CREATION-----------------------------------------------
    # ==================================================================================================================

    # Run all connectivities of the rectangular beam
    for c, connection in enumerate(CONNECTIVITIES):
        xy_coord = elementXY_Coord(connection, NODAL_COORDINATES)  # (x,y) coordinates of the element (anticlockwise)

        # Creation of element's Stiffness Matrix
        Ke = np.zeros((DEGREE_OF_FREEDOM * NUMBER_NODES_IN_ELEMENT, DEGREE_OF_FREEDOM * NUMBER_NODES_IN_ELEMENT))

        GaussLocations, Weights = gaussQuadratureT3(1)  # Gauss Quadrature Locations & Weights

        # Run through each Gauss Location/Weight
        for gl, gauss_location in enumerate(GaussLocations):

            xi, eta = gauss_location  # Assign ξ and η to the specific Gauss Point Coordinates

            ShapeFunction, DerivativeShapeFunction = shapeFunctionT3(xi, eta)

            J, invJ, detJ = Jacobian(DerivativeShapeFunction, xy_coord)

            DerivativeShapeFunction_xy = invJ.dot(DerivativeShapeFunction)

            B = createMatrixB(DerivativeShapeFunction_xy, DEGREE_OF_FREEDOM, NUMBER_NODES_IN_ELEMENT)

            # Element Stiffness Matrix, due to specific Gauss-Point is given by Ke = B.T * C * B * |J| * Wi * H
            Ke += np.dot(np.dot(B.T, C), B) * detJ * Weights[gl] * H

            for i, ii in enumerate(connection):  # Runs through all elements nodes of the connectivity
                for j, jj in enumerate(connection):  # Runs through all element nodes, again, of the connectivity
                    # Ke is 6x6 (8 = 2Dof * 3-node)

                    # i_: index of first for-loop (Local Ke index)
                    # j: index of second for-loop (Local Ke index)
                    # ii: node of first for-loop (Global K index)
                    # jj: node of second for-loop (Global K index)

                    K[2 * ii, 2 * jj] += Ke[2 * i, 2 * j]  # Ke11
                    K[2 * ii + 1, 2 * jj] += Ke[2 * i + 1, 2 * j]  # Ke21
                    K[2 * ii, 2 * jj + 1] += Ke[2 * i, 2 * j + 1]  # Ke12
                    K[2 * ii + 1, 2 * jj + 1] += Ke[2 * i + 1, 2 * j + 1]  # Ke22

    # ==================================================================================================================
    # ----------------------------------------LOADS & BOUNDARY CONDITIONS-----------------------------------------------
    # ==================================================================================================================

    Px = 1000.0  # Force [N/m]

    # Left Hand Side (L.H.S.) index of Nodes
    LHS_NODES = np.array(
        [[i, i + NUMBER_NODES_X]
         for i in np.arange(start=0,
                            stop=NUMBER_NODES - NUMBER_NODES_X,
                            step=NUMBER_NODES_X)]
    )

    # Left Hand Side (L.H.S.) index of Nodes, with Degrees of Freedom
    LHS_NODES_DOF = np.array(
        [[i, i + 1, i + NUMBER_NODES_X * DEGREE_OF_FREEDOM, i + NUMBER_NODES_X * DEGREE_OF_FREEDOM + 1]
         for i in np.arange(start=0,
                            stop=(NUMBER_NODES - NUMBER_NODES_X) * DEGREE_OF_FREEDOM,
                            step=NUMBER_NODES_X * DEGREE_OF_FREEDOM)]
    )

    F = forceVector(LHS_NODES, LHS_NODES_DOF, Px, NODAL_COORDINATES)  # Create

    # Prescribed Nodes are essential for solving K * x = b. R.H.S. nodes are prescribed us the boundary condition,
    # were the (u,v) are equal to (0,0) for all the nodes. PRESCRIBED_NODES gives us the boundary nodes.
    PRESCRIBED_NODES = np.array([
        [2 * i, 2 * i + 1] for i in range((NUMBER_NODES_X - 1), NUMBER_NODES, NUMBER_NODES_X)
    ]).flatten()

    # ACTIVE_NODES are the nodes that are NOT PRESCRIBED_NODES
    ACTIVE_NODES = np.setdiff1d(np.arange(0, DEGREE_OF_FREEDOM * NUMBER_NODES), PRESCRIBED_NODES)
    # PRESCRIBED_NODES[-NUMBER_NODES_Y:-1], PRESCRIBED_NODES[-1] = np.array([0, 0])

    # Solving system K * x = b, of only active nodes
    U_PRESCRIBED = np.linalg.solve(K[np.ix_(ACTIVE_NODES, ACTIVE_NODES)], F[ACTIVE_NODES])
    # U_PRESCRIBED = np.linalg.lstsq(K[np.ix_(ACTIVE_NODES, ACTIVE_NODES)], F[ACTIVE_NODES])

    # Assign Active nodes to the U Matrix, so the prescribed nodes remain (u,v)=(0,0)
    U[ACTIVE_NODES] = U_PRESCRIBED

    # Find (u,v) coordinates of each node
    # the U Matrix goes like: U = [u1, v1, u2, v2,..., uN, vN].T
    UU = U[0::2]  # u displacement of each node
    VV = U[1::2]  # v displacement of each node

    UU = np.reshape(UU, (NUMBER_NODES_X, NUMBER_NODES_Y))  # Reshape Uu by Nx * Ny
    VV = np.reshape(VV, (NUMBER_NODES_X, NUMBER_NODES_Y))  # Reshape Uv by Nx * Ny

    # Create again X, Y coordinates in order to plot them
    X, Y = np.meshgrid(NODAL_COORDINATES_X, NODAL_COORDINATES_Y)

    # Plot the beam mesh with color VV=(X,Y)
    plt.imshow(VV, cmap='coolwarm', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)])

    plt.xlim(np.min(X) - 1, np.max(X) + 1)  # Set the x-axis limits
    plt.ylim(np.min(Y) - 1, np.max(Y) + 1)  # Set the y-axis limits
    plt.grid()
    plt.colorbar(label='U [m]')  # Add colorbar label
    plt.xlabel('X [m]')  # Add x-axis label
    plt.ylabel('Y [m]')  # Add y-axis label
    plt.title('2D Plane Stress Analysis with 3-node Triangles')  # Add plot title
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()
