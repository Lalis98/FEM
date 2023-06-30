# Imports 
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def Ke(p1, p2, Ee, Ae):
    """
    Calculates the local stiffness matrix (ke) for an individual element.

    Arguments:
    p1, p2: coordinates of the two nodes (numpy arrays) connected by the element
    E: Young's modulus of the element
    K: Cross-sectional area of the element

    Returns:
    ke: 6x6 element stiffness matrix
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    L_ = np.linalg.norm(p1 - p2)
    cx = (x2 - x1) / L_
    cy = (y2 - y1) / L_
    cz = (z2 - z1) / L_

    ke = Ee * Ae / L_ * np.array([
        [cx ** 2, cx * cy, cx * cz, - cx ** 2, - cx * cy, - cx * cz],
        [cx * cy, cy ** 2, cy * cz, - cx * cy, - cy ** 2, - cy * cz],
        [cx * cz, cy * cz, cz ** 2, - cx * cz, - cy * cz, - cz ** 2],
        [- cx ** 2, - cx * cy, - cx * cz, cx ** 2, cx * cy, cx * cz],
        [- cx * cy, - cy ** 2, - cy * cz, cx * cy, cy ** 2, cy * cz],
        [- cx * cz, - cy * cz, - cz ** 2, cx * cz, cy * cz, cz ** 2]
    ])

    return ke


def Te(p1, p2):
    """
    Calculates the transformation matrix (Te) for an individual element.

    Arguments:
    p1, p2: coordinates of the two nodes (numpy arrays) connected by the element

    Returns:
    Te: 2x6 element transformation matrix
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    L_ = np.linalg.norm(p1 - p2)
    cx = (x2 - x1) / L_
    cy = (y2 - y1) / L_
    cz = (z2 - z1) / L_

    te = np.array([
        [cx, cy, cz, 0, 0, 0],
        [0, 0, 0, cx, cy, cz]
    ])

    return te


class truss_3D:
    def __init__(self, points, connections, E, A, prescribed_dof, force, force_dof_location):
        """Initializes the truss_3D class with given parameters."""
        self.points = points  # Points is an array containing the coordinates of the nodes in the truss system
        self.connections = connections  # Connections is an array defining how the nodes are connected to form elements
        self.E = E  # E is an array containing the Young's modulus for each element
        self.A = A  # K is an array containing the cross-sectional area for each element
        self.dof = 3  # Degrees of freedom per node (in this case, for a 3D truss, there are 3 degrees of freedom per
        # node)
        self.prescribed_dof = prescribed_dof  # Prescribed_dof is an array containing the indices of the prescribed
        # degrees of freedom
        self.force = force  # Force is an array containing the applied forces at specific degrees of freedom
        self.force_dof_location = force_dof_location  # Force_dof_location is an array containing the indices of the
        # degrees of freedom where the forces are applied

        self.nodes = np.arange(len(points))  # Nodes' Indices
        self.elements = np.arange(len(connections))  # Elements' Indices

        self.nodes_dof = np.arange(len(points) * self.dof)  # Indices Degrees of Freedom of Nodes

        # Indices of Degrees of Freedom of nodes' connections
        self.connections_dof = np.array([
            [i[0] * self.dof, i[0] * self.dof + 1, i[0] * self.dof + 2,  # (u1, v1, w1)
             i[1] * self.dof, i[1] * self.dof + 1, i[1] * self.dof + 2]  # (u2, v2, w2)
            for i in connections])

        # Initialize Stiffness, Displacement and Force Matrices
        self.K = np.zeros((len(self.nodes) * self.dof, len(self.nodes) * self.dof))  # Stiffness Matrix
        self.F = np.zeros((len(self.nodes) * self.dof, 1))  # Force Matrix
        self.U = np.zeros((len(self.nodes) * self.dof, 1))  # Displacement Matrix

        # Elongation, Strain and Sigma Matrices
        self.dL = np.empty((0, 1))
        self.EPSILON = np.empty((0, 1))
        self.SIGMA = np.empty((0, 1))

        # Mapping Variables
        self.X = []
        self.Y = []
        self.Z = []

    def stiffness_matrix(self):
        """
        Calculates and returns the global stiffness matrix by assembling
        individual element stiffness matrices.
        """
        # Run through all connectivities
        for c, connection in enumerate(self.connections):
            n1, n2 = connection  # node 1 & node 2

            selection = np.ix_(self.connections_dof[c], self.connections_dof[c])  # Selection of the Connectivities

            ke = Ke(self.points[n1], self.points[n2], self.E[c], self.A[c])  # Stiffness Matrix of the element

            self.K[selection] += ke  # Adding element's Stiffness Matrix to global Stiffness Matrix

        return self.K

    def force_vector(self):
        """
        Updates the force vector based on the input forces and their
        respective degree of freedom locations.
        Returns the updated force vector.
        """
        for i, I in enumerate(self.force):
            self.F[self.force_dof_location[i]] = I

        return self.F

    def sigma(self):
        """
        Computes elongation, strain, and stress for each element in the truss.
        Returns the stress (SIGMA) and normalized stress (NORMALIZED_SIGMA) arrays.
        """
        U = self.analysis()  # Call Displacements from the analysis

        for c, connection in enumerate(self.connections):
            n1, n2 = connection  # node 1 & node 2

            l = np.linalg.norm(self.points[n2] - self.points[n1])  # Calculate distance between two points

            T = Te(self.points[n1], self.points[n2])  # Calculates the transformation matrix (Te) for an individual
            # element.

            # An array of the Displacements of each element
            Ue = np.array(
                [n1 * self.dof, n1 * self.dof + 1, n1 * self.dof + 2,  # (u1, v1, w1)
                 n2 * self.dof, n2 * self.dof + 1, n2 * self.dof + 2]  # (u2, v2, w2)
            )

            u1, u2 = T.dot(U[Ue])  # Resultant vectors u1(u1, v1, w1) and u2(u2, v2, w2)

            self.dL = np.append(self.dL, [u2 - u1], axis=0)  # Elongation of each element

            self.EPSILON = np.append(self.EPSILON, [(u2 - u1) / l], axis=0)  # Strain of each element

            self.SIGMA = np.append(self.SIGMA, [(u2 - u1) / l * self.E[c]], axis=0)  # σ of each element

        # Normalized sigma (from 0 to 1)
        NORMALIZED_SIGMA = (self.SIGMA - np.min(self.SIGMA)) / (np.max(self.SIGMA) - np.min(self.SIGMA))

        return self.SIGMA, NORMALIZED_SIGMA

    def analysis(self):
        """
        Performs a static analysis of the truss structure by solving
        the system of equations (KU = F) and obtaining displacements and forces.
        Returns the displacement vector (U).
        """
        # Calculate the active degrees of freedom by removing the prescribed degrees of freedom
        active_dof = np.setdiff1d(self.nodes_dof, self.prescribed_dof)

        # Extract the active submatrix of the global stiffness matrix
        K_active = self.stiffness_matrix()[np.ix_(active_dof, active_dof)]

        # Extract the active force vector corresponding to the active degrees of freedom
        F_active = self.force_vector()[active_dof]

        # Solve the system of equations (K_active * U_prescribed = F_active) to obtain the displacements
        U_prescribed = np.linalg.solve(K_active, F_active)

        # Assign the solved displacements to their corresponding positions in the global displacement vector
        self.U[active_dof] = U_prescribed

        # forces = np.round(self.stiffness_matrix().dot(self.U))  # Find all reaction forces

        # Return the displacement vector
        return self.U

    def mapping(self, colormap):
        """
        Creates a 3D plot of the truss structure, color-coded based on
        the normalized stress in each element, using the provided colormap.
        """
        # Call σ and σ_normalized from the
        sigma, normalized_sigma = self.sigma()

        # Generate a color array based on the colormap and normalized stress values
        color_array = [colormap[int(i * (len(colormap) - 1))] for i in normalized_sigma]

        # Create an empty Plotly figure
        fig = go.Figure()

        # Loop through all connections, appending their coordinates to X, Y, and Z lists
        for i, I in enumerate(self.connections):
            self.X.append([self.points[I[0]][0], self.points[I[1]][0]])

            self.Y.append([self.points[I[0]][1], self.points[I[1]][1]])

            self.Z.append([self.points[I[0]][2], self.points[I[1]][2]])

        # Convert X, Y, Z lists to NumPy arrays
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Z = np.array(self.Z)

        # Add each connection as a trace (line) in the 3D plot with corresponding colors
        for i in range(len(self.X)):
            fig.add_trace(go.Scatter3d(
                x=self.X[i],
                y=self.Y[i],
                z=self.Z[i],
                mode='lines',
                line=dict(
                    width=3,
                    color=color_array[i]
                )
            ))

        # Add a color-bar to the plot with min and max stress values
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=0,
                color=np.linspace(min(sigma), max(sigma), len(colormap)),
                colorscale=colormap,
                colorbar=dict(title='max(σ)=%.1f MPa, min(σ)=%.1f MPa' % (np.max(sigma), np.min(sigma)))
            ),
            showlegend=False,
        ))

        # Update the layout of the plot with axis labels
        fig.update_layout(
            scene=dict(
                xaxis_title="x [m]",
                yaxis_title="y [m]",
                zaxis_title="z [m]"
            ),
            title='3D Truss Finite Element Method'
        )

        # Hide the legend and show the plot
        fig.update_layout(showlegend=False)
        fig.show()


def main():
    L = 4.2  # Parametrized length for the truss grid structure

    # Points Coordinates i.e. [ x, y, z ]
    POINTS = np.array([
        [-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0],  # BASE 0

        [-L / 2, -L / 2, L], [L / 2, -L / 2, L], [L / 2, L / 2, L], [-L / 2, L / 2, L],  # BASE 1

        [-L / 2, -L / 2, 2 * L], [L / 2, -L / 2, 2 * L], [L / 2, L / 2, 2 * L], [-L / 2, L / 2, 2 * L],  # BASE 2

        [-L / 2, -L / 2, 3 * L], [L / 2, -L / 2, 3 * L], [L / 2, L / 2, 3 * L], [-L / 2, L / 2, 3 * L],  # BASE 3

        [-L / 2, -L / 2, 3.75 * L], [L / 2, -L / 2, 3.75 * L], [L / 2, L / 2, 3.75 * L], [-L / 2, L / 2, 3.75 * L],
        # BASE 4

        [0, - 3 * L / 2, 3 * L], [3 * L / 2, 0, 3 * L], [0, 3 * L / 2, 3 * L], [- 3 * L / 2, 0, 3 * L]  # WINGS

    ])
    # Connectivities array i.e. [ Node 1, Node 2 ]
    CONNECTIVITIES = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # BASE 0

        [0, 4], [1, 5], [2, 6], [3, 7],  # Perpendicular elements connecting BASE 0 with BASE 1

        [0, 5], [1, 4],  # Diagonal elements of 4-sides between BASE 0 and BASE 1
        [1, 6], [2, 5],
        [2, 7], [3, 6],
        [3, 4], [0, 7],

        [4, 5], [5, 6], [6, 7], [7, 4],  # BASE 1

        [4, 6], [5, 7],  # BASE 1 Center Node Connections

        [4, 8], [5, 9], [6, 10], [7, 11],  # Perpendicular elements connecting BASE 1 with BASE 2

        [4, 9], [5, 8],  # Diagonal elements of 4-sides between BASE 0 and BASE 1
        [5, 10], [6, 9],
        [6, 11], [7, 10],
        [7, 8], [4, 11],

        [8, 9], [9, 10], [10, 11], [11, 8],  # BASE 2

        [8, 12], [9, 13], [10, 14], [11, 15],  # Perpendicular elements connecting BASE 2 with BASE 3

        [12, 13], [13, 14], [14, 15], [15, 12],  # BASE 3

        [8, 13], [9, 12],  # Diagonal elements of 4-sides between BASE 0 and BASE 1
        [9, 14], [10, 13],
        [10, 15], [11, 14],
        [11, 12], [8, 15],

        [12, 14], [13, 15],  # BASE 3 Center Node Connections

        [12, 16], [13, 17], [14, 18], [15, 19],  # Perpendicular elements connecting BASE 3 with BASE 4

        [16, 17], [17, 18], [18, 19], [19, 16],  # BASE 4

        [13, 16], [12, 17],  # Diagonal elements of 4-sides between BASE 0 and BASE 1
        [13, 18], [14, 17],
        [14, 19], [15, 18],
        [15, 16], [12, 19],

        [17, 19], [16, 18],  # BASE 3 Center Node Connections

        [12, 20], [13, 20], [17, 20], [16, 20],  # WING 1

        [13, 21], [14, 21], [18, 21], [17, 21],  # WING 2

        [14, 22], [15, 22], [19, 22], [18, 22],  # WING 3

        [12, 23], [15, 23], [16, 23], [19, 23],  # WING 4

    ])

    E = 200e9 * np.ones(len(CONNECTIVITIES))
    A = 1.0 * np.ones(len(CONNECTIVITIES))
    PRESCRIBED_NODES = np.array([0, 1, 2, 3])  # Prescribed Nodes of the Structure
    DOF = 3
    PRESCRIBED_NODES_DOF = np.array([[i * DOF, i * DOF + 1, i * DOF + 2] for i in PRESCRIBED_NODES]).flatten()
    FORCES = np.array([-1000, -1000, -1000, -1000])
    FORCES_DOF_LOCATION = np.array([50, 53, 56, 59])
    COLORMAP = px.colors.sequential.RdBu

    st1 = truss_3D(points=POINTS,
                   connections=CONNECTIVITIES,
                   E=E,
                   A=A,
                   force=FORCES,
                   force_dof_location=FORCES_DOF_LOCATION,
                   prescribed_dof=PRESCRIBED_NODES_DOF
                   )
    st1.mapping(colormap=COLORMAP)


if __name__ == "__main__":
    main()
