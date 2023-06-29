# Imports
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import time

'''
The mesh and indexing of the 8-node quadrilateral shapes are the following:

                8─────────9─────────10────────11──────────12
                │                    │                    │
                │                    │                    │ 
                5                    6                    7
                │                    │                    │
                │                    │                    │
                0──────────1─────────2─────────3──────────4

---------
Notations and Symbols:

πp: Potential Energy [J]
F, f: Forces [N]
Xb: Body Forces [N]
Tx: Surface Forces [N]
ε: Strain [-]
u: Nodal Displacement [m]
σ: Stress [Pa]
Ε: Young's Modulus, E = 200 * 10 ** 9 [Pa]
ν: Poisson's Ratio, ν = 0.3 [-]
h: Thickness of the element at z direction [m]
[ ]: Matrix
{ }: Vector
[ ].T: Transpose Matrix
[ ]^-1: Inverse Matrix

Governing Equations for Finite Element Method:

                            Tx
                       ─> ─> ─> ─> ─>
                ┌──────────────────────────┐
        f1x ───>│            o───> Xb      │─────> f2x ---------------------> x,u
              1 └──────────────────────────┘ 2

1. Potential Energy:

   πp = A / 2 * ∫ (σ * ε) dx - f1x * u1 - f2x * u2 - ∫∫ (us * Tx)dS - ∫∫∫ (u * Xb)dV

2. Minimization of Potential Energy:

   ∂πp/∂u = 0

3. Stiffness Matrix

    [K] = ∫∫([B].T * [D] * [B] * h [J]) dξ dη

4. Governing Equation

   [K] * {U} = {F}

-------------

Solution strategy: 

1. Create mesh Matrices K, U and F (Initialization)

2. Create Element Stiffness Matrix for each element

3. Create Shape Function of natural coordinates Ν(ξ,η) for each Gaussian point:

     [N1]
     [N2]
     [N3]   
N  = [N4]
     [N5]
     [N6]
     [N7]
     [N8]

4. Create the Natural Derivative of Shape Function for each Gaussian point:

[dN] = [∂N1/∂ξ ∂N2/∂ξ ∂N3/∂ξ ∂N4/∂ξ ∂N5/∂ξ ∂N6/∂ξ ∂N7/∂ξ ∂N8/∂ξ]
       [∂N1/∂η ∂N2/∂η ∂N3/∂η ∂N4/∂η ∂N5/∂η ∂N6/∂η ∂N7/∂η ∂N8/∂η]

4. Calculate the inverse Jacobian Matrix [J] and the determinant |J|

Create the Matrix B

      [∂N1/∂x    0  | ... |∂N8/∂x    0  ]
[B] = [  0    ∂N1/∂y| ... |  0    ∂N8/∂y]
      [∂N1/∂y ∂N1/∂x| ... |∂N8/∂y ∂N8/∂x]

5. Solve the double integral numerically with Gauss Quadrature

6. Apply Boundary Conditions and then solve {U} = [K]^-1 * {F}

7. Sigma (σ) is calculated as:

{ε} = [B] * {U}
{σ} = [D] * {ε}     ==>      {σ} = [D] * [B] * {U}  

'''


def main():
    print('The Program has been executed!')
    print('Calculating...')

    # ------------------------------------------------------------------------------------------------------------------
    # =============================================== FUNCTIONS ========================================================
    # ------------------------------------------------------------------------------------------------------------------

    def createQuadrilateralMesh(nx_, ny_, length_, height_):

        dx_ = length_ / (2 * (nx_ - 1))
        dy_ = height_ / (2 * (ny_ - 1))

        connections_ = []
        for j in range(ny_ - 1):

            for i in range((nx_ - 1)):
                # nodes 0,1,2,..,8
                n0 = 2 * i + j * (3 * nx_ - 1)
                n1 = 1 + 2 * i + j * (3 * nx_ - 1)
                n2 = 2 + 2 * i + j * (3 * nx_ - 1)
                n3 = i + 1 + (j + 1) * (nx_ - 1) * 2 + (nx_ + 1) * j
                n4 = i + 2 + (j + 1) * (nx_ - 1) * 2 + (nx_ + 1) * j
                n5 = 1 + 2 * i + j + (j + 1) * 2 * (nx_ - 1) + (j + 1) * nx_
                n6 = 2 + 2 * i + j + (j + 1) * 2 * (nx_ - 1) + (j + 1) * nx_
                n7 = 3 + 2 * i + j + (j + 1) * 2 * (nx_ - 1) + (j + 1) * nx_

                connections_.append([n0, n1, n2, n3, n4, n5, n6, n7])

        coordinates_ = []
        center_coordinates_ = []
        for j in range(ny_ - 1):

            for i in range(nx_ - 1):
                # coordinates 0,1,2,..,8
                x0, y0 = (2 * i) * dx_, dy_ * j * 2
                x1, y1 = (1 + 2 * i) * dx_, dy_ * j * 2
                x2, y2 = (2 + 2 * i) * dx_, dy_ * j * 2
                x3, y3 = (2 * i) * dx_, dy_ * (2 * j + 1)
                x4, y4 = (2 + 2 * i) * dx_, dy_ * (2 * j + 1)
                x5, y5 = (2 * i) * dx_, dy_ * (j + 1) * 2
                x6, y6 = (1 + 2 * i) * dx_, dy_ * (j + 1) * 2
                x7, y7 = (2 + 2 * i) * dx_, dy_ * (j + 1) * 2

                coordinates_.append([
                    [x0, y0],
                    [x1, y1],
                    [x2, y2],
                    [x3, y3],
                    [x4, y4],
                    [x5, y5],
                    [x6, y6],
                    [x7, y7]
                ])

                center_coordinates_.append(
                    [(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) / 8, (y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7) / 8]
                )

        return np.array(connections_), np.array(coordinates_), np.array(center_coordinates_)

    def shapeFunctions(xi_, eta_):

        N1_ = -1 / 4 * (1 - xi_) * (1 - eta_) * (1 + xi_ + eta_)
        N2_ = 1 / 2 * (1 - xi_ ** 2) * (1 - eta_)
        N3_ = -1 / 4 * (1 + xi_) * (1 - eta_) * (1 - xi_ + eta_)
        N4_ = 1 / 2 * (1 - xi_) * (1 - eta_ ** 2)
        N5_ = 1 / 2 * (1 + xi_) * (1 - eta_ ** 2)
        N6_ = -1 / 4 * (1 - xi_) * (1 + eta_) * (1 + xi_ - eta_)
        N7_ = 1 / 2 * (1 - xi_ ** 2) * (1 + eta_)
        N8_ = -1 / 4 * (1 + xi_) * (1 + eta_) * (1 - xi_ - eta_)

        dN1_dxi_ = -1 / 4 * (-1 + eta_) * (2 * xi_ + eta_)
        dN1_deta_ = -1 / 4 * (-1 + xi_) * (xi_ + 2 * eta_)

        dN2_dxi_ = xi_ * (-1 + eta_)
        dN2_deta_ = 1 / 2 * (1 + xi_) * (-1 + xi_)

        dN3_dxi_ = 1 / 4 * (-1 + eta_) * (eta_ - 2 * xi_)
        dN3_deta_ = 1 / 4 * (1 + xi_) * (2 * eta_ - xi_)

        dN4_dxi_ = 1 / 2 * (1 + eta_) * (-1 + eta_)
        dN4_deta_ = eta_ * (-1 + xi_)

        dN5_dxi_ = -1 / 2 * (1 + eta_) * (-1 + eta_)
        dN5_deta_ = -eta_ * (1 + xi_)

        dN6_dxi_ = -1 / 4 * (1 + eta_) * (eta_ - 2 * xi_)
        dN6_deta_ = -1 / 4 * (-1 + xi_) * (2 * eta_ - xi_)

        dN7_dxi_ = -xi_ * (1 + eta_)
        dN7_deta_ = -1 / 2 * (1 + xi_) * (-1 + xi_)

        dN8_dxi_ = 1 / 4 * (1 + eta_) * (eta_ + 2 * xi_)
        dN8_deta_ = 1 / 4 * (1 + xi_) * (xi_ + 2 * eta_)

        N_ = np.array([
            [N1_],
            [N2_],
            [N3_],
            [N4_],
            [N5_],
            [N6_],
            [N7_],
            [N8_]
        ])

        dN_ = np.array([
            [dN1_dxi_, dN2_dxi_, dN3_dxi_, dN4_dxi_, dN5_dxi_, dN6_dxi_, dN7_dxi_, dN8_dxi_],
            [dN1_deta_, dN2_deta_, dN3_deta_, dN4_deta_, dN5_deta_, dN6_deta_, dN7_deta_, dN8_deta_]
        ])

        return N_, dN_

    def gaussQuadrature(order):
        if order == '3rd':
            gauss_point_ = np.array([
                [-np.sqrt(0.6), -np.sqrt(0.6)],
                [0, -np.sqrt(0.6)],
                [np.sqrt(0.6), -np.sqrt(0.6)],
                [-np.sqrt(0.6), 0],
                [0, 0],
                [np.sqrt(0.6), 0],
                [-np.sqrt(0.6), np.sqrt(0.6)],
                [0, np.sqrt(0.6)],
                [np.sqrt(0.6), np.sqrt(0.6)]
            ])
            gauss_weight_ = np.array([
                [25 / 81],
                [40 / 81],
                [25 / 81],
                [40 / 81],
                [64 / 81],
                [40 / 81],
                [25 / 81],
                [40 / 81],
                [25 / 81]
            ])

        elif order == '2nd':
            gauss_point_ = np.array([
                [-1 / np.sqrt(3), -1 / np.sqrt(3)],
                [1 / np.sqrt(3), -1 / np.sqrt(3)],
                [-1 / np.sqrt(3), 1 / np.sqrt(3)],
                [1 / np.sqrt(3), 1 / np.sqrt(3)]
            ])
            gauss_weight_ = np.array([
                [1],
                [1],
                [1],
                [1]
            ])

        else:
            gauss_point_ = np.array([
                [0, 0],
            ])
            gauss_weight_ = np.array([
                [4]
            ])

        return gauss_point_, gauss_weight_

    def jacobian(natural_derivative_shape_function, xy_coordinates):

        J_ = natural_derivative_shape_function.dot(xy_coordinates)  # Compute the Jacobian matrix

        return J_, np.linalg.inv(J_), np.linalg.det(J_)  # Return the Jacobian matrix, its inverse, and determinant

    def matrixB(derivative_shape_function, dof, nodes_in_element):

        B_ = np.zeros((3, dof * nodes_in_element))  # Creating Matrix B_ 2x8

        B_[0, 0::2] = derivative_shape_function[0, :]  # row 1 of Matrix B_
        B_[1, 1::2] = derivative_shape_function[1, :]  # row 2 of Matrix B_
        B_[2, 0::2] = derivative_shape_function[1, :]  # row 3, odd columns of Matrix B_
        B_[2, 1::2] = derivative_shape_function[0, :]  # row 3, even columns of Matrix B_

        # Return Matrix B_
        return B_

    def boundary(boundary_side, nx_, ny_):
        connections_ = []

        if boundary_side == 'left':
            for j in range(ny_):
                i0u = (j * (3 * nx_ - 1)) * 2
                i0v = (j * (3 * nx_ - 1)) * 2 + 1
                connections_.append([i0u, i0v])

            for j in range(ny_ - 1):
                i1u = (1 + (j + 1) * (nx_ - 1) * 2 + (nx_ + 1) * j) * 2
                i1v = (1 + (j + 1) * (nx_ - 1) * 2 + (nx_ + 1) * j) * 2 + 1
                connections_.append([i1u, i1v])

        elif boundary_side == 'right':
            for j in range(ny_):
                i0u = (j * (3 * nx_ - 1) + ny_ + nx_ - 1) * 2 - 2
                i0v = (j * (3 * nx_ - 1) + ny_ + nx_ - 1) * 2 - 1

                connections_.append([i0u, i0v])

            for j in range(ny_ - 1):
                i1u = (j * (3 * nx_ - 1) + ny_ + nx_ - 1 + (nx_ - 1)) * 2
                i1v = (j * (3 * nx_ - 1) + ny_ + nx_ - 1 + (nx_ - 1)) * 2 + 1
                connections_.append([i1u, i1v])

        elif boundary_side == 'upper':
            for i in range(nx_):
                i0u = 2 * (nx_ * (ny_ - 1) + (nx_ - 1) * (ny_ - 1) + (nx_ - 1) * ny_) + 4 * i
                i0v = 2 * (nx_ * (ny_ - 1) + (nx_ - 1) * (ny_ - 1) + (nx_ - 1) * ny_) + 4 * i + 1
                connections_.append([i0u, i0v])

            for i in range(nx_ - 1):
                i1u = 2 * (nx_ * (ny_ - 1) + (nx_ - 1) * (ny_ - 1) + (nx_ - 1) * ny_) + 4 * i + 2
                i1v = 2 * (nx_ * (ny_ - 1) + (nx_ - 1) * (ny_ - 1) + (nx_ - 1) * ny_) + 4 * i + 3
                connections_.append([i1u, i1v])

        elif boundary_side == 'lower':
            for i in range(nx_):
                i0u = 4 * i
                i0v = 4 * i + 1

                connections_.append([i0u, i0v])

            for i in range(nx_ - 1):
                i1u = 4 * i + 2
                i1v = 4 * i + 3

                connections_.append([i1u, i1v])

        return np.array(connections_).flatten()

    def displacedNodes(coordinates_, displacement_u_, displacement_v_, scale_factor_):

        if coordinates_.ndim > 2:
            coordinates_ = coordinates_.reshape((-1, 2))

        coordinates_ = np.unique(coordinates_, axis=0)

        coordinates_new_ = np.empty((0, 2))
        for i, c in enumerate(coordinates_):
            x_, y_ = c
            x_ += scale_factor_ * displacement_u_[i][0]
            y_ += scale_factor_ * displacement_v_[i][0]

            coordinates_new_ = np.append(coordinates_new_, [[x_, y_]], axis=0)

        return coordinates_new_[:, 0], coordinates_new_[:, 1]

    # ------------------------------------------------------------------------------------------------------------------
    # ========================================== INPUT PARAMETERS ======================================================
    # ------------------------------------------------------------------------------------------------------------------

    LENGTH = 10  # Length in meters
    HEIGHT = 1  # Height in meters
    NX = 31
    NY = 31
    NODES_IN_ELEMENT = 8
    CONNECTIONS, COORDINATES, CENTER_COORDINATES = createQuadrilateralMesh(nx_=NX, ny_=NY, length_=LENGTH,
                                                                           height_=HEIGHT)
    NUMBER_OF_NODES = NX * NY + (NX - 1) * NY + (NY - 1) * NX
    DOF = 2
    NODES_DOF = np.arange(start=0, stop=DOF * NUMBER_OF_NODES)

    CONNECTIONS_DOF = np.array([
        [
            DOF * c[0], DOF * c[0] + 1,
            DOF * c[1], DOF * c[1] + 1,
            DOF * c[2], DOF * c[2] + 1,
            DOF * c[3], DOF * c[3] + 1,
            DOF * c[4], DOF * c[4] + 1,
            DOF * c[5], DOF * c[5] + 1,
            DOF * c[6], DOF * c[6] + 1,
            DOF * c[7], DOF * c[7] + 1,
        ] for i, c in enumerate(CONNECTIONS)
    ])

    # ------------------------------------------------------------------------------------------------------------------
    # ======================================== MATERIAL CHARACTERISTICS ================================================
    # ------------------------------------------------------------------------------------------------------------------

    E = 2e9  # Young's Modulus in MPa
    POISSON = 0.3  # Poisson Ratio
    THICKNESS = 0.05  # Thickness in meters

    D = E / (1 - POISSON ** 2) * np.array([  # material stiffness matrix
        [1, POISSON, 0],
        [POISSON, 1, 0],
        [0, 0, (1 - POISSON) / 2]
    ])

    # ------------------------------------------------------------------------------------------------------------------
    # ========================================= CONSTRUCTION OF MATRICES ===============================================
    # ------------------------------------------------------------------------------------------------------------------

    K = np.zeros((len(NODES_DOF), len(NODES_DOF)))
    U = np.zeros((len(NODES_DOF), 1))
    F = np.zeros((len(NODES_DOF), 1))

    # ------------------------------------------------------------------------------------------------------------------
    # ==================================== CONSTRUCTION OF GLOBAL STIFFNESS MATRIX ==========+==========================
    # ------------------------------------------------------------------------------------------------------------------

    for c, connection in enumerate(CONNECTIONS):

        selection = np.ix_(CONNECTIONS_DOF[c], CONNECTIONS_DOF[c])

        ke = np.zeros((NODES_IN_ELEMENT * DOF, NODES_IN_ELEMENT * DOF))

        coordinates = COORDINATES[c]
        gauss_point, gauss_weights = gaussQuadrature(order='3rd')

        for i, gp in enumerate(gauss_point):
            xi, eta = gp
            gw = gauss_weights[i][0]

            N, dN_dxi_eta = shapeFunctions(xi, eta)
            J, invJ, detJ = jacobian(natural_derivative_shape_function=dN_dxi_eta, xy_coordinates=coordinates)

            dN_dx_y = invJ.dot(dN_dxi_eta)

            B = matrixB(derivative_shape_function=dN_dx_y, dof=DOF, nodes_in_element=NODES_IN_ELEMENT)

            ke += B.T.dot(D.dot(B)) * gw * detJ * THICKNESS

        K[selection] += ke

    # ------------------------------------------------------------------------------------------------------------------
    # ====================================== BOUNDARY CONDITIONS AND LOADS =============================================
    # ------------------------------------------------------------------------------------------------------------------

    PRESCRIBED_DOF = boundary('right', NX, NY)
    large_value = 1e20  # Choose an appropriate large value
    for i in PRESCRIBED_DOF:
        K[i, :] = 0
        K[:, i] = 0
        K[i, i] = large_value
        F[i] = 0

    FORCE_X, FORCE_Y = boundary('upper', NX, NY)[0::2], boundary('upper', NX, NY)[1::2]

    F[FORCE_Y] = -1e3

    # ------------------------------------------------------------------------------------------------------------------
    # ============================================ SOLVING K * U = F ===================================================
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()  # Start the timer

    U = np.linalg.solve(K, F)  # Solve the equation K * U = F

    elapsed_time = time.time() - start_time  # Calculate the elapsed time

    print("Time elapsed for solving K * U = F: {:.3f} seconds, for {:.0f} nodes".format(elapsed_time, NX * NY * 2))

    # ------------------------------------------------------------------------------------------------------------------
    # ========================================= CALCULATE EPSILON & SIGMA ==============================================
    # ------------------------------------------------------------------------------------------------------------------
    EPSILON = np.empty((0, 3))
    SIGMA = np.empty((0, 3))
    for c, connection in enumerate(CONNECTIONS):
        coordinates = COORDINATES[c]

        xi = eta = 0

        N, dN_dxi_eta = shapeFunctions(xi, eta)

        J, invJ, detJ = jacobian(natural_derivative_shape_function=dN_dxi_eta, xy_coordinates=coordinates)

        dN_dx_y = invJ.dot(dN_dxi_eta)

        B = matrixB(derivative_shape_function=dN_dx_y, dof=DOF, nodes_in_element=NODES_IN_ELEMENT)

        epsilon = B.dot(U[CONNECTIONS_DOF[c]]).flatten()
        sigma = D.dot(epsilon).flatten()

        EPSILON = np.append(EPSILON, [epsilon], axis=0)
        SIGMA = np.append(SIGMA, [sigma], axis=0)

    SIGMA_X = [SIGMA[:, 0], "σx"]
    SIGMA_Y = [SIGMA[:, 1], "σy"]
    TAPH_XY = [SIGMA[:, 2], "τxy"]

    # Extract x and y coordinates
    X_COORDINATES = COORDINATES[:, :, 0].flatten()
    Y_COORDINATES = COORDINATES[:, :, 1].flatten()
    DISPLACEMENT_U = U[0::2]
    DISPLACEMENT_V = U[1::2]

    X_NEW, Y_NEW = displacedNodes(COORDINATES, DISPLACEMENT_U, DISPLACEMENT_V, 1)

    # ------------------------------------------------------------------------------------------------------------------
    # ============================================ PLOT RESULTS ========================================================
    # ------------------------------------------------------------------------------------------------------------------

    # PLACE IN z_var THE VALUE YOU WANT TO PLOT
    # Create a contour plot for V
    # Create the main heatmap trace
    sigma_x = go.Heatmap(
        x=CENTER_COORDINATES[:, 0],
        y=CENTER_COORDINATES[:, 1],
        z=SIGMA_X[0] * 10 ** (-6),
        colorscale=px.colors.sequential.RdBu,
    )

    # Create additional heatmap traces
    sigma_y = go.Heatmap(
        x=CENTER_COORDINATES[:, 0],
        y=CENTER_COORDINATES[:, 1],
        z=SIGMA_Y[0] * 10 ** (-6),
        colorscale=px.colors.sequential.RdBu,
        visible=False  # Set the trace to initially not be shown
    )

    taph_xy = go.Heatmap(
        x=CENTER_COORDINATES[:, 0],
        y=CENTER_COORDINATES[:, 1],
        z=TAPH_XY[0] * 10 ** (-6),
        colorscale=px.colors.sequential.RdBu,
        visible=False  # Set the trace to initially not be shown
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[sigma_x, sigma_y, taph_xy])

    # Update the layout
    fig.update_layout(
        title='Stress Contour',
        xaxis_title='x [m]',
        yaxis_title='y [m]',
        annotations=[
            dict(
                x=1.05,
                y=1.05,
                xref='paper',
                yref='paper',
                text=' [MPa]',
                showarrow=False,
                font=dict(size=14)
            )
        ],
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'visible': [True, False, False]}],
                        label=SIGMA_X[1],
                        method='update'
                    ),
                    dict(
                        args=[{'visible': [False, True, False]}],
                        label=SIGMA_Y[1],
                        method='update'
                    ),
                    dict(
                        args=[{'visible': [False, False, True]}],
                        label=TAPH_XY[1],
                        method='update'
                    )
                ]),
                direction='down',
                showactive=True,
                x=0.90,
                xanchor='center',
                y=1.08,
                yanchor='top'
            )
        ]
    )

    fig.show()


if __name__ == "__main__":
    main()
    print('\nFinished!')
