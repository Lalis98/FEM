# Imports
import numpy as np
import plotly.graph_objects as go

'''
This Python script implements the finite element analysis of a rectangular plate
under uniform transverse pressure using the Mindlin-Reissner plate theory. The 
plate has dimensions a and b, and is discretized into quadrilateral elements 
with linear shape functions. Each node has three degrees of freedom: the 
out-of-plane displacement (w) and rotations about the x and y axes (θx and θy).

The material properties of the plate are defined by the Young's modulus (E = 200 GPa),
Poisson's ratio (ν = 0.3), and thickness. The shear modulus (G) is calculated from E 
and ν using the formula G = 1 / 2 * E / (1 + ν).

The load on the plate is a uniform transverse pressure.
The response of the plate to this load is calculated by assembling the global stiffness
matrix from the bending and shear stiffness matrices of each element, and then solving
the resulting system of equations.

The bending stiffness matrix (Kb) is calculated using the bending strain-displacement
matrix and the bending material property matrix, which is derived from E, ν, and the 
plate thickness. The shear stiffness matrix (Ks) is calculated using the shear
strain-displacement matrix and the shear material property matrix, which is derived 
from G and the plate thickness.

The boundary conditions can be either clamped or simply supported. For clamped boundary
conditions, all degrees of freedom at the boundary nodes are constrained. For simply
supported boundary conditions, only the out-of-plane displacement at the boundary nodes
is constrained.

The solution of the problem gives the displacements and rotations at each node of the plate,
which can be used to analyze the deformation and stresses in the plate under the applied load.
    
                                  North Side

                               +-----------+
                              /↓↓↓↓↓↓↓↓↓↓↓/
                             /↓↓↓↓↓↓↓↓↓↓↓/
                West Side   /↓↓↓↓↓↓↓↓↓↓↓/  East Side
                           /↓↓↓↓↓↓↓↓↓↓↓/
                          /↓↓↓↓↓↓↓↓↓↓↓/
                         +-----------+

                          South Side
'''


# ======================================================================================================================
# --------------------------------------------------FUNCTIONS-----------------------------------------------------------
# ======================================================================================================================
def main():
    def ShapeFunctionQ4(xi_, eta_):
        """
        This function computes the shape functions and their derivatives
        for a 4-node quadrilateral element (Q4) in a finite element analysis.

        Parameters:
            xi_, eta_: Natural (or iso-parametric) coordinates. These vary from -1 to 1.

        Returns:
            N_: Shape function matrix. Each row represents a node of the element.
            dN_: Derivative of shape function matrix with respect to natural coordinates xi and eta.
        """

        # Define shape functions for each node in terms of xi and eta
        N1 = 1 / 4 * (1 - xi_) * (1 - eta_)
        N2 = 1 / 4 * (1 + xi_) * (1 - eta_)
        N3 = 1 / 4 * (1 + xi_) * (1 + eta_)
        N4 = 1 / 4 * (1 - xi_) * (1 + eta_)

        # Construct shape function matrix
        N_ = np.array([
            [N1],
            [N2],
            [N3],
            [N4]
        ])

        # Compute derivatives of shape functions with respect to xi
        dN1dxi = -1 / 4 * (1 - eta_)
        dN2dxi = 1 / 4 * (1 - eta_)
        dN3dxi = 1 / 4 * (1 + eta_)
        dN4dxi = -1 / 4 * (1 + eta_)

        # Compute derivatives of shape functions with respect to eta
        dN1deta = -1 / 4 * (1 - xi_)
        dN2deta = -1 / 4 * (1 + xi_)
        dN3deta = 1 / 4 * (1 + xi_)
        dN4deta = 1 / 4 * (1 - xi_)

        # Construct matrix of shape function derivatives
        dN_ = np.array([
            [dN1dxi, dN2dxi, dN3dxi, dN4dxi],
            [dN1deta, dN2deta, dN3deta, dN4deta]
        ])

        return N_, dN_

    def GaussQuadrature(order):
        """
        This function provides Gauss quadrature points and weights for
        numerical integration over a quadrilateral element in a finite element analysis.

        Parameters:
            order: Order of Gauss quadrature, can be 'second' or otherwise.

        Returns:
            xi_eta: An array of Gauss quadrature points in the (xi, eta) natural coordinates.
            w: Weights of the Gauss points.
        """
        # 2nd order degree 4-node Gauss Quadrature
        if order == 'second':
            # Gauss-Points Coordinates of (ξ,η)
            xi_eta = np.array([
                [-np.sqrt(1 / 3), -np.sqrt(1 / 3)],
                [np.sqrt(1 / 3), -np.sqrt(1 / 3)],
                [-np.sqrt(1 / 3), np.sqrt(1 / 3)],
                [np.sqrt(1 / 3), np.sqrt(1 / 3)]])

            # Weight of Gauss-Points
            w = np.array([1, 1, 1, 1])

        # 1st order degree 1-node Gauss Quadrature
        else:
            # Single Gauss-Point Coordinate at the center of the element
            xi_eta = np.array([[0, 0]])

            # Weight of the single Gauss-Point
            w = np.array([4])

        # Return (ξ,η) coordinates and their weights
        return xi_eta, w

    def Jacobian(natural_derivative_shape_function, xy_coordinates_):
        """
        This function computes the Jacobian matrix, its inverse, and determinant
        for transformation between natural and global coordinates in a finite element analysis.

        Parameters:
            natural_derivative_shape_function: The derivative of the shape function
                                               with respect to natural coordinates (xi, eta).
            xy_coordinates_: The xy coordinates of the nodes of the element.

        Returns:
            J: The Jacobian matrix.
            np.linalg.inv(J): The inverse of the Jacobian matrix.
            np.linalg.det(J): The determinant of the Jacobian matrix.
        """

        # Compute the Jacobian matrix
        J_ = natural_derivative_shape_function.dot(xy_coordinates_)

        # Return the Jacobian matrix, its inverse, and determinant
        return J_, np.linalg.inv(J_), np.linalg.det(J_)

    def Matrix_B_Bending(dN_dxy_):
        """
        This function computes the bending strain-displacement matrix (B matrix)
        for a plate or shell element in a finite element analysis.

        Parameters:
            dN_dxy_: The derivative of the shape function
                     with respect to global coordinates (x, y).

        Returns:
            B_: The bending strain-displacement matrix.

        Note:
            This function assumes a constant DEGREE_OF_FREEDOM and NODES_IN_ELEMENT across the script.
        """

        # DEGREE_OF_FREEDOM and NODES_IN_ELEMENT should be defined elsewhere in your code

        B_ = np.zeros((3, DEGREE_OF_FREEDOM * NODES_IN_ELEMENT))  # Creating Matrix B_ 3x8

        B_[1, 1::3] = dN_dxy_[1, :]  # row 1 of Matrix B_
        B_[2, 0::3] = -dN_dxy_[0, :]  # row 2 of Matrix B_
        B_[2, 1::3] = dN_dxy_[0, :]  # row 3, odd columns of Matrix B_
        B_[2, 2::3] = -dN_dxy_[1, :]  # row 3, even columns of Matrix B_

        # Return Matrix B_
        return B_

    def Matrix_B_Shear(dN_dxy_, shape_):
        """
        This function computes the shear strain-displacement matrix (B matrix)
        for a plate or shell element in a finite element analysis.

        Parameters:
            dN_dxy_: The derivative of the shape function
                     with respect to global coordinates (x, y).
            shape_: The shape function evaluated at a specific point in the
                    natural coordinate system.

        Returns:
            B_: The shear strain-displacement matrix.

        Note:
            This function assumes a constant DEGREE_OF_FREEDOM and NODES_IN_ELEMENT across the script.
        """

        # DEGREE_OF_FREEDOM and NODES_IN_ELEMENT should be defined elsewhere in your code

        B_ = np.zeros((2, DEGREE_OF_FREEDOM * NODES_IN_ELEMENT))  # Creating Matrix B_ 2x8

        B_[0, 0::3] = dN_dxy_[0, :]  # row 1 of Matrix B_
        B_[1, 0::3] = dN_dxy_[1, :]  # row 2 of Matrix B_
        B_[1, 1::3] = -shape_  # row 3, odd columns of Matrix B_
        B_[0, 2::3] = shape_  # row 3, even columns of Matrix B_

        # Return Matrix B_
        return B_

    def Force(shape_, pressure_):
        """
        This function computes the element force vector for a plate or shell element
        in a finite element analysis.

        Parameters:
            shape_: The shape function evaluated at a specific point in the
                    natural coordinate system.
            pressure_: The pressure applied on the element.

        Returns:
            f_: The force vector.

        Note:
            This function assumes a constant DEGREE_OF_FREEDOM and NODES_IN_ELEMENT across the script.
        """

        # DEGREE_OF_FREEDOM and NODES_IN_ELEMENT should be defined elsewhere in your code

        f_ = np.zeros((DEGREE_OF_FREEDOM * NODES_IN_ELEMENT, 1))  # Creating force vector f_
        fef_ = shape_ * pressure_  # Calculating element force

        f_[0::3, :] = fef_[0]  # Assigning values to the force vector

        # Return force vector f_
        return f_

    def BoundaryConditions(type_, xy_coordinates_):
        """
        This function determines the boundary conditions of a plate or shell element in a finite element analysis.

        Parameters:
            type_: String which gives the type of boundary condition. This can be either 'Clamped Supported'
                   or 'Simply Supported'.
            xy_coordinates_: The xy-coordinates of nodes.

        Returns:
            boundary_dof: The boundary condition degrees of freedom (dof).

        Note:
            This function assumes a constant DEGREE_OF_FREEDOM across the script.
        """

        # DEGREE_OF_FREEDOM should be defined elsewhere in your code

        # Find the nodes on the boundaries
        lower_boundary_y = np.where(xy_coordinates_[:, 1] == min(xy_coordinates_[:, 1]))[0]
        upper_boundary_y = np.where(xy_coordinates_[:, 1] == max(xy_coordinates_[:, 1]))[0]
        lower_boundary_x = np.where(xy_coordinates_[:, 0] == min(xy_coordinates_[:, 0]))[0]
        upper_boundary_x = np.where(xy_coordinates_[:, 0] == max(xy_coordinates_[:, 0]))[0]

        if type_ == 'Clamped Supported':
            # For clamped boundary condition, all dofs (displacement and rotation) are constrained
            boundary = np.hstack((lower_boundary_y, upper_boundary_y, lower_boundary_x, upper_boundary_x))
            boundary = list(set(boundary))

            boundary_dof = np.array([
                # (w1, θx1, θy1)
                [i * DEGREE_OF_FREEDOM, i * DEGREE_OF_FREEDOM + 1, i * DEGREE_OF_FREEDOM + 2]
                for i in boundary])

            return boundary_dof.flatten()

        else:
            # For other boundary conditions, only certain dofs are constrained
            boundary = np.hstack((lower_boundary_y, upper_boundary_y, lower_boundary_x, upper_boundary_x))
            boundary = list(set(boundary))

            boundary_dof = np.array([
                # (w1)
                [i * DEGREE_OF_FREEDOM]
                for i in boundary])

            return boundary_dof.flatten()

    # ==================================================================================================================
    # ---------------------------------------------INPUT DATA FOR NODAL CONNECTIVITY------------------------------------
    # ==================================================================================================================
    print('The Program has started!')
    print('Calculating...')

    NODES_IN_ELEMENT = 4  # Number of nodes in each element

    NUMBER_NODES_X = 21  # Number of Nodes across x-axis
    NUMBER_NODES_Y = 21  # Number of Nodes across y-axis

    NUMBER_ELEMENTS_X = NUMBER_NODES_X - 1  # Number of Elements across x-axis
    NUMBER_ELEMENTS_Y = NUMBER_NODES_Y - 1  # Number of Elements across y-axis

    NUMBER_NODES = NUMBER_NODES_X * NUMBER_NODES_Y  # Total Number of Nodes

    DEGREE_OF_FREEDOM = 3  # Degrees of Freedoms of each element
    TOTAL_NUMBER_DOF = NUMBER_NODES * DEGREE_OF_FREEDOM  # Total Number of Degrees of Freedom
    NODES_DOF = np.arange(NUMBER_NODES * DEGREE_OF_FREEDOM)  # Total Indices of Degrees of Freedom

    # ==================================================================================================================
    # --------------------------------------GEOMETRICAL & MATERIAL PROPERTIES OF THE PLATE------------------------------
    # ==================================================================================================================

    X_START = 0  # x-axis starting x-coordinate of the rectangular element
    Y_START = 0  # y-axis starting y-coordinate of the rectangular element
    X_END = 2  # x-axis ending x-coordinate of the rectangular element
    Y_END = 2  # y-axis ending y-coordinate of the rectangular element

    NODAL_COORDINATES_X = np.linspace(X_START, X_END, NUMBER_NODES_X)  # Coordinates Range for x
    NODAL_COORDINATES_Y = np.linspace(Y_START, Y_END, NUMBER_NODES_Y)  # Coordinates Range for y

    COORDINATES = np.array(
        [[i, j] for j in NODAL_COORDINATES_Y for i in NODAL_COORDINATES_X])  # Coordinates of all nodes

    CONNECTIONS = []  # Create a Matrix CONNECTIONS were there are connections of all nodes of each element
    for j in range(NUMBER_ELEMENTS_Y):
        for i in range(NUMBER_ELEMENTS_X):
            n1 = i + NUMBER_NODES_Y * j  # Node 1
            n2 = i + 1 + NUMBER_NODES_Y * j  # Node 2
            n3 = i + 1 + NUMBER_NODES_Y * (j + 1)  # Node 3
            n4 = i + NUMBER_NODES_Y * (j + 1)  # Node 4

            CONNECTIONS.append([n1, n2, n3, n4])

    CONNECTIONS = np.array(CONNECTIONS)

    CONNECTIONS_DOF = np.array([
        # (w1, θx1, θy1)
        [i[0] * DEGREE_OF_FREEDOM, i[0] * DEGREE_OF_FREEDOM + 1, i[0] * DEGREE_OF_FREEDOM + 2,
         # (w2, θx2, θy2)
         i[1] * DEGREE_OF_FREEDOM, i[1] * DEGREE_OF_FREEDOM + 1, i[1] * DEGREE_OF_FREEDOM + 2,
         # (w3, θx3, θy3)
         i[2] * DEGREE_OF_FREEDOM, i[2] * DEGREE_OF_FREEDOM + 1, i[2] * DEGREE_OF_FREEDOM + 2,
         # (w4, θx4, θy4)
         i[3] * DEGREE_OF_FREEDOM, i[3] * DEGREE_OF_FREEDOM + 1, i[3] * DEGREE_OF_FREEDOM + 2]
        for i in CONNECTIONS])

    E = 200 * 10 ** 9  # Young's Modulus [Pa]
    POISSON = 0.3  # Poisson Ratio [-]
    G = 1 / 2 * E / (1 + POISSON)  # Young's Modulus [Pa]
    THICKNESS = 0.01  # Plate Thickness [m]
    SHEAR_CORRECTION_FACTOR = 5 / 6  # Shear Correction Factor

    # ==================================================================================================================
    # ----------------------------------------INITIALIZATION OF MATRICES AND VECTORS------------------------------------
    # ==================================================================================================================

    K = np.zeros((TOTAL_NUMBER_DOF, TOTAL_NUMBER_DOF))  # Stiffness Matrix
    U = np.zeros((TOTAL_NUMBER_DOF, 1))  # Displacement Vector
    F = np.zeros((TOTAL_NUMBER_DOF, 1))  # Force Vector

    # ==================================================================================================================
    # ------------------------------------------TRANSVERSE UNIFORM PRESSURE ON PLATE------------------------------------
    # ==================================================================================================================

    PRESSURE = -1e7  # Transverse Pressure on plate [in Pa]

    # ==================================================================================================================
    # ----------------------------------------COMPUTATION OF ELEMENT MATRICES AND VECTORS-------------------------------
    # ==================================================================================================================

    D = E * THICKNESS ** 3 / 12 / (1 - POISSON ** 2) * np.array([  # Bending Material Property
        [1, POISSON, 0],
        [POISSON, 1, 0],
        [0, 0, (1 - POISSON) / 2]
    ])

    Ds = G * SHEAR_CORRECTION_FACTOR * THICKNESS * np.array([  # Shear Material Property
        [1, 0],
        [0, 1]
    ])

    # Iterate over all CONNECTIONS, which should represent the list of elements
    for c, connection in enumerate(CONNECTIONS):

        # For each element, select the corresponding degrees of freedom (DOF)
        selection = np.ix_(CONNECTIONS_DOF[c], CONNECTIONS_DOF[c])

        # Initialize local stiffness and force matrices
        kb = np.zeros(
            (DEGREE_OF_FREEDOM * NODES_IN_ELEMENT, DEGREE_OF_FREEDOM * NODES_IN_ELEMENT))  # Bending stiffness matrix
        ks = np.zeros(
            (DEGREE_OF_FREEDOM * NODES_IN_ELEMENT, DEGREE_OF_FREEDOM * NODES_IN_ELEMENT))  # Shear stiffness matrix

        f = np.zeros((DEGREE_OF_FREEDOM * NODES_IN_ELEMENT, 1))  # Force matrix

        # Get Gauss points and weights for 2nd order Gauss quadrature
        GaussPoint, GaussWeight = GaussQuadrature(order='second')

        # Get the coordinates of the current element's nodes
        xy_coord = COORDINATES[connection]

        # Perform Gaussian integration for each Gauss point
        for gp, GP in enumerate(GaussPoint):
            xi, eta = GP  # Gauss Point coordinates (xi, eta)
            WP = GaussWeight[gp]  # Corresponding Gauss weight

            # Calculate Shape Functions and their derivatives
            N, dN_dxieta = ShapeFunctionQ4(xi, eta)

            # Compute Jacobian and its inverse and determinant
            J, invJ, detJ = Jacobian(natural_derivative_shape_function=dN_dxieta, xy_coordinates_=xy_coord)

            # Transform the derivatives to the physical coordinates
            dN_dxy = invJ.dot(dN_dxieta)

            # Create the bending B matrix
            B = Matrix_B_Bending(dN_dxy)

            # Accumulate the contributions to the local bending stiffness matrix
            kb += B.T.dot(D.dot(B)) * WP * detJ

        # Get Gauss points and weights for 1st order Gauss quadrature (for shear)
        GaussPoint, GaussWeight = GaussQuadrature(order='first')

        # Perform Gaussian integration for each Gauss point (for shear)
        for gp, GP in enumerate(GaussPoint):
            xi, eta = GP  # Gauss Point coordinates (xi, eta)
            WP = GaussWeight[0]  # Corresponding Gauss weight

            # Calculate Shape Functions and their derivatives
            N, dN_dxieta = ShapeFunctionQ4(xi, eta)

            # Compute Jacobian and its inverse and determinant
            J, invJ, detJ = Jacobian(natural_derivative_shape_function=dN_dxieta, xy_coordinates_=xy_coord)

            # Transform the derivatives to the physical coordinates
            dN_dxy = invJ.dot(dN_dxieta)

            # Create the shear B matrix
            Bs = Matrix_B_Shear(dN_dxy, N.flatten())

            # Accumulate the contributions to the local shear stiffness matrix
            ks += Bs.T.dot(Ds.dot(Bs)) * WP * detJ

            # Calculate force for each element
            fe = Force(N, PRESSURE)
            # Accumulate the contributions to the local force vector
            f += fe * WP * detJ

        # Combine bending and shear stiffness matrices
        ke = kb + ks

        # Accumulate the contributions to the global stiffness matrix and force vector
        K[selection] += ke
        F[CONNECTIONS_DOF[c]] = f

    # Determine the nodes at which boundary conditions are prescribed. The boundary condition is 'Clamped Supported'.
    PRESCRIBED_DOF = BoundaryConditions('Clamped Supported', COORDINATES)

    # Get the indices of active DOFs, which are the DOFs that are not prescribed.
    ACTIVE_DOF = np.setdiff1d(NODES_DOF, PRESCRIBED_DOF)

    # Solve the system of equations, KU=F, to get the values of displacements and rotations at the active DOFs.
    U_PRESCRIBED = np.linalg.solve(K[np.ix_(ACTIVE_DOF, ACTIVE_DOF)], F[ACTIVE_DOF])

    # Store the solved displacements and rotations back to their corresponding positions in the global displacement
    # vector.
    U[ACTIVE_DOF] = U_PRESCRIBED

    # Extract the displacement ('UU') and rotations ('THITA_X' and 'THITA_Y') from the global displacement vector.
    UU = U[0::3]
    THITA_X = U[1::3]
    THITA_Y = U[2::3]

    # Reshape 'UU' to match the grid dimensions for visualization.
    UU = UU.reshape((NUMBER_NODES_X, NUMBER_NODES_Y))

    # Create a meshgrid with the x and y coordinates of the nodes.
    X, Y = np.meshgrid(NODAL_COORDINATES_X, NODAL_COORDINATES_Y)

    # Create a 3D plot of the displacement field.
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=UU)])

    # Customize the 3D plot to clearly present the data.
    fig.update_layout(scene=dict(
        xaxis_title='X [m]',  # Set the title for the x-axis
        yaxis_title='Y [m]',  # Set the title for the y-axis
        zaxis_title='Z [m]',  # Set the title for the z-axis
        aspectmode='manual',  # Set the aspect mode of the 3D plot manually
        aspectratio=dict(x=1, y=1, z=0.5)),  # Set the aspect ratio of the x, y, and z axes
        margin=dict(l=65, r=50, b=65, t=90),
        title="Mindlin-Reissner Bending Plate "
    )  # Set the margin of the plot

    # Display the 3D plot.
    fig.show()


if __name__ == "__main__":
    main()
    print('\n Finished!')
