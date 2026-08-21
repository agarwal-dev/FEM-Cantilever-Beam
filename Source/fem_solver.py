import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class Material:
    def __init__(self, E=210e9, nu=0.3, rho=7850):
        self.E = E
        self.nu = nu
        self.rho = rho
        # Plane stress constitutive matrix
        self.D = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

class Mesh:
    def __init__(self, L, H, nx, ny):
        self.L = L
        self.H = H
        self.nx = nx
        self.ny = ny
        self.nnodes = (nx + 1) * (ny + 1)
        self.nelems = nx * ny
        self.nodes = self._generate_nodes()
        self.elements = self._generate_elements()
        
    def _generate_nodes(self):
        x = np.linspace(0, self.L, self.nx + 1)
        y = np.linspace(0, self.H, self.ny + 1)
        xx, yy = np.meshgrid(x, y)
        return np.vstack([xx.ravel(), yy.ravel()]).T

    def _generate_elements(self):
        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = j * (self.nx + 1) + i
                n2 = j * (self.nx + 1) + (i + 1)
                n3 = (j + 1) * (self.nx + 1) + (i + 1)
                n4 = (j + 1) * (self.nx + 1) + i
                elements.append([n1, n2, n3, n4])
        return np.array(elements)

class FEMSolver:
    def __init__(self, mesh, material, thickness=0.1):
        self.mesh = mesh
        self.mat = material
        self.t = thickness
        self.ndof = 2 * mesh.nnodes
        self.K = None
        self.M = None
        # 2x2 Gaussian Quadrature
        self.gauss_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
        self.gauss_wts = [1.0, 1.0]

    def shape_functions(self, xi, eta):
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])
        dN_dxi = 0.25 * np.array([
            -(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)
        ])
        dN_deta = 0.25 * np.array([
            -(1 - xi), -(1 + xi),  (1 + xi),  (1 - xi)
        ])
        return N, np.vstack([dN_dxi, dN_deta])

    def assemble(self):
        I = []
        J = []
        V_K = []
        V_M = []
        
        for e, elem in enumerate(self.mesh.elements):
            coords = self.mesh.nodes[elem]
            ke = np.zeros((8, 8))
            me = np.zeros((8, 8))
            
            for xi, w_xi in zip(self.gauss_pts, self.gauss_wts):
                for eta, w_eta in zip(self.gauss_pts, self.gauss_wts):
                    N, dN_dxieta = self.shape_functions(xi, eta)
                    
                    Jac = dN_dxieta @ coords
                    detJ = np.linalg.det(Jac)
                    invJac = np.linalg.inv(Jac)
                    
                    dN_dxdy = invJac @ dN_dxieta
                    
                    B = np.zeros((3, 8))
                    B[0, 0::2] = dN_dxdy[0, :]
                    B[1, 1::2] = dN_dxdy[1, :]
                    B[2, 0::2] = dN_dxdy[1, :]
                    B[2, 1::2] = dN_dxdy[0, :]
                    
                    N_mat = np.zeros((2, 8))
                    N_mat[0, 0::2] = N
                    N_mat[1, 1::2] = N
                    
                    weight = w_xi * w_eta * detJ * self.t
                    ke += B.T @ self.mat.D @ B * weight
                    me += N_mat.T @ N_mat * self.mat.rho * weight
            
            dofs = np.zeros(8, dtype=int)
            dofs[0::2] = 2 * elem
            dofs[1::2] = 2 * elem + 1
            
            for i in range(8):
                for j in range(8):
                    I.append(dofs[i])
                    J.append(dofs[j])
                    V_K.append(ke[i, j])
                    V_M.append(me[i, j])
                    
        self.K = sp.coo_matrix((V_K, (I, J)), shape=(self.ndof, self.ndof)).tocsr()
        self.M = sp.coo_matrix((V_M, (I, J)), shape=(self.ndof, self.ndof)).tocsr()

    def apply_boundary_conditions(self, q):
        """
        Fix left end (x=0) and apply uniform load q on top edge (y=H).
        q is force per unit length in the y-direction (typically negative).
        """
        F = np.zeros(self.ndof)
        
        # Fixed nodes at x = 0
        fixed_nodes = np.where(np.isclose(self.mesh.nodes[:, 0], 0))[0]
        fixed_dofs = np.empty(2 * len(fixed_nodes), dtype=int)
        fixed_dofs[0::2] = 2 * fixed_nodes
        fixed_dofs[1::2] = 2 * fixed_nodes + 1
        
        free_dofs = np.setdiff1d(np.arange(self.ndof), fixed_dofs)
        
        # Load on top edge (y = H)
        for i in range(self.mesh.nx):
            n4 = (self.mesh.ny) * (self.mesh.nx + 1) + i
            n3 = (self.mesh.ny) * (self.mesh.nx + 1) + (i + 1)
            le = self.mesh.L / self.mesh.nx
            # Apply equivalent nodal loads (q * L_elem / 2) to Y-dofs
            F[2 * n4 + 1] += q * le / 2.0
            F[2 * n3 + 1] += q * le / 2.0
            
        return F, free_dofs, fixed_dofs
        
    def solve_static(self, q):
        if self.K is None:
            self.assemble()
            
        F, free_dofs, fixed_dofs = self.apply_boundary_conditions(q)
        
        K_ff = self.K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        U_f = spla.spsolve(K_ff, F_f)
        
        U = np.zeros(self.ndof)
        U[free_dofs] = U_f
        
        # Total strain energy U = 1/2 U^T K U
        strain_energy = 0.5 * U.T @ self.K @ U
        return U, strain_energy

    def compute_stress_strain(self, U):
        """
        Computes stress and strain at the center of each element.
        """
        stresses = []
        strains = []
        
        for e, elem in enumerate(self.mesh.elements):
            coords = self.mesh.nodes[elem]
            
            dofs = np.zeros(8, dtype=int)
            dofs[0::2] = 2 * elem
            dofs[1::2] = 2 * elem + 1
            ue = U[dofs]
            
            # Evaluate at center (xi=0, eta=0)
            _, dN_dxieta = self.shape_functions(0, 0)
            Jac = dN_dxieta @ coords
            invJac = np.linalg.inv(Jac)
            dN_dxdy = invJac @ dN_dxieta
            
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dxdy[0, :]
            B[1, 1::2] = dN_dxdy[1, :]
            B[2, 0::2] = dN_dxdy[1, :]
            B[2, 1::2] = dN_dxdy[0, :]
            
            strain = B @ ue
            stress = self.mat.D @ strain
            
            strains.append(strain)
            stresses.append(stress)
            
        return np.array(strains), np.array(stresses)

    def solve_modal(self, num_modes=5):
        if self.K is None or self.M is None:
            self.assemble()
            
        _, free_dofs, _ = self.apply_boundary_conditions(0)
        
        K_ff = self.K[np.ix_(free_dofs, free_dofs)]
        M_ff = self.M[np.ix_(free_dofs, free_dofs)]
        
        # solve generalized eigenvalue problem K phi = lambda M phi
        eigenvalues, eigenvectors_f = spla.eigsh(K_ff, k=num_modes, M=M_ff, sigma=0, which='LM')
        
        # Sort eigenvalues just in case
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors_f = eigenvectors_f[:, idx]
        
        # Avoid negative eigenvalues due to numerical noise
        eigenvalues = np.abs(eigenvalues)
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        
        eigenvectors = np.zeros((self.ndof, num_modes))
        eigenvectors[free_dofs, :] = eigenvectors_f
        
        return frequencies, eigenvectors
