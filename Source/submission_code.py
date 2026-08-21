import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os

# ==========================================
# 1. FEM CORE CLASSES
# ==========================================

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
        dN_dxi = 0.25 * np.array([-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi),  (1 + xi),  (1 - xi)])
        return N, np.vstack([dN_dxi, dN_deta])

    def assemble(self):
        I, J, V_K, V_M = [], [], [], []
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
        strain_energy = 0.5 * U.T @ self.K @ U
        return U, strain_energy

    def compute_stress_strain(self, U):
        stresses = []
        strains = []
        for e, elem in enumerate(self.mesh.elements):
            coords = self.mesh.nodes[elem]
            dofs = np.zeros(8, dtype=int)
            dofs[0::2] = 2 * elem
            dofs[1::2] = 2 * elem + 1
            ue = U[dofs]
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
        eigenvalues, eigenvectors_f = spla.eigsh(K_ff, k=num_modes, M=M_ff, sigma=0, which='LM')
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors_f = eigenvectors_f[:, idx]
        eigenvalues = np.abs(eigenvalues)
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        eigenvectors = np.zeros((self.ndof, num_modes))
        eigenvectors[free_dofs, :] = eigenvectors_f
        return frequencies, eigenvectors

# ==========================================
# 2. ANALYTICAL SOLUTIONS
# ==========================================
def euler_bernoulli_deflection(x, L, E, I, q):
    return q * (x**2 * (6 * L**2 - 4 * L * x + x**2)) / (24 * E * I)

def timoshenko_deflection(x, L, E, G, I, A, k, q):
    v_eb = euler_bernoulli_deflection(x, L, E, I, q)
    v_shear = q * (L * x - x**2 / 2) / (k * A * G)
    return v_eb + v_shear

def shear_force(x, L, q):
    return q * (L - x)

def timoshenko_shear_stress(y, V, I, t, H):
    y_na = y - H / 2
    Q = t / 2 * ((H / 2)**2 - y_na**2)
    return V * Q / (I * t)

def analytical_strain_energy_eb(L, E, I, q):
    return (q**2 * L**5) / (40 * E * I)

def analytical_strain_energy_timoshenko(L, E, G, I, A, k, q):
    U_b = analytical_strain_energy_eb(L, E, I, q)
    U_s = (q**2 * L**3) / (6 * k * A * G)
    return U_b + U_s

# ==========================================
# 3. ANALYSIS AND PLOTTING ROUTINES
# ==========================================
def generate_plots():
    os.makedirs('plots', exist_ok=True)
    
    # Interactive Base Parameters
    print("="*50)
    print("    2D FEM Cantilever Beam Analysis Simulator        ")
    print("="*50)
    print("\nPlease enter the physical parameters (or press Enter to use default values):")
    try:
        H_in = input("Enter beam height H in meters [default=1.0]: ")
        H = float(H_in) if H_in.strip() else 1.0
        t_in = input("Enter beam thickness t in meters [default=0.1]: ")
        t = float(t_in) if t_in.strip() else 0.1
        E_in = input("Enter Young's Modulus E in Pa [default=210e9]: ")
        E = float(E_in) if E_in.strip() else 210e9
        nu_in = input("Enter Poisson's ratio nu [default=0.3]: ")
        nu = float(nu_in) if nu_in.strip() else 0.3
        rho_in = input("Enter Density rho in kg/m^3 [default=7850.0]: ")
        rho = float(rho_in) if rho_in.strip() else 7850.0
        q_in = input("Enter uniform load q in N/m (negative for downward) [default=-10000.0]: ")
        q = float(q_in) if q_in.strip() else -10000.0
    except ValueError:
        print("\n[!] Invalid input. Using default values.")
        H, t, E, nu, rho, q = 1.0, 0.1, 210e9, 0.3, 7850.0, -10000.0
        
    print("\nStarting analyses...\n")
    
    G = E / (2 * (1 + nu))
    I = t * H**3 / 12
    A_area = t * H
    k_shear = 5/6
    mat = Material(E, nu, rho)

    # ---------------------------------------------------------
    # PART A: Mesh Convergence & Contours (Aspect Ratio = 50)
    # ---------------------------------------------------------
    L = 50.0 * H  # Maintain AR=50
    print(f"Running Validation (AR={L/H})...")
    
    meshes = [(10, 2), (20, 4), (40, 8), (80, 16), (160, 32)]
    fem_defs, fem_energies, num_elems = [], [], []
    
    def_eb = euler_bernoulli_deflection(L/2, L, E, I, q)
    def_timo = timoshenko_deflection(L/2, L, E, G, I, A_area, k_shear, q)
    en_eb = analytical_strain_energy_eb(L, E, I, q)
    en_timo = analytical_strain_energy_timoshenko(L, E, G, I, A_area, k_shear, q)

    for nx, ny in meshes:
        mesh = Mesh(L, H, nx, ny)
        solver = FEMSolver(mesh, mat, thickness=t)
        U, energy = solver.solve_static(q)
        
        node_A = np.argmin(np.sqrt((mesh.nodes[:, 0] - L/2)**2 + (mesh.nodes[:, 1] - 0)**2))
        def_A = U[2 * node_A + 1]
        
        fem_defs.append(def_A)
        fem_energies.append(energy)
        num_elems.append(mesh.nelems)
        
        # Save converged state for contours
        if (nx, ny) == meshes[-1]:
            U_conv = U
            mesh_conv = mesh
            solver_conv = solver

    # Plot 1: Disp at A vs elements
    plt.figure()
    plt.plot(num_elems, np.abs(fem_defs), 'o-', label='FEM')
    plt.axhline(np.abs(def_eb), color='r', ls='--', label='Euler-Bernoulli')
    plt.axhline(np.abs(def_timo), color='g', ls='-.', label='Timoshenko')
    plt.xlabel('Number of Elements')
    plt.ylabel('Absolute Deflection at Point A (m)')
    plt.title('Deflection at A vs No. of Elements')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/disp_at_A_convergence.png')
    plt.close()

    # Plot 2: Strain energy vs elements
    plt.figure()
    plt.plot(num_elems, fem_energies, 'o-', label='FEM')
    plt.axhline(en_eb, color='r', ls='--', label='Euler-Bernoulli')
    plt.axhline(en_timo, color='g', ls='-.', label='Timoshenko')
    plt.xlabel('Number of Elements')
    plt.ylabel('Total Strain Energy (J)')
    plt.title('Strain Energy vs No. of Elements')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/energy_convergence.png')
    plt.close()

    # Contours for converged mesh (AR=50)
    print("Generating Contours...")
    strains, stresses = solver_conv.compute_stress_strain(U_conv)
    
    # 3. Displacement Contour (Y-displacement)
    uy = U_conv[1::2]
    
    # Convert Q4 to triangles for matplotlib tricontourf
    triangles = []
    for elem in mesh_conv.elements:
        triangles.append([elem[0], elem[1], elem[2]])
        triangles.append([elem[0], elem[2], elem[3]])
    triangles = np.array(triangles)
    
    plt.figure(figsize=(10, 2))
    plt.tricontourf(mesh_conv.nodes[:, 0], mesh_conv.nodes[:, 1], triangles, uy, levels=50, cmap='jet')
    plt.colorbar(label='Y-Displacement (m)')
    plt.title('Displacement Contour (y-direction)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('plots/displacement_contour.png')
    plt.close()

    # Helper function for element-wise contour plotting
    def plot_element_data(mesh, data, title, filename, label):
        verts = mesh.nodes[mesh.elements]
        pc = PolyCollection(verts, cmap='jet', edgecolor='none')
        pc.set_array(data)
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, label=label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # 4. Stress Contour (Sigma_xx)
    sigma_xx = stresses[:, 0]
    plot_element_data(mesh_conv, sigma_xx, 'Bending Stress Contour ($\sigma_{xx}$)', 'plots/stress_contour.png', 'Stress (Pa)')

    # 5. Strain Contour (Epsilon_xx)
    eps_xx = strains[:, 0]
    plot_element_data(mesh_conv, eps_xx, 'Bending Strain Contour ($\epsilon_{xx}$)', 'plots/strain_contour.png', 'Strain')

    # ---------------------------------------------------------
    # PART B: Shear Stress Investigation (Aspect Ratio = 5)
    # ---------------------------------------------------------
    L_shear = 5.0 * H  # Maintain AR=5
    print(f"Running Shear Investigation (AR={L_shear/H})...")
    mesh_s = Mesh(L_shear, H, 40, 20)
    solver_s = FEMSolver(mesh_s, mat, thickness=t)
    U_s, _ = solver_s.solve_static(q)
    _, stresses_s = solver_s.compute_stress_strain(U_s)
    
    mid_x = L_shear / 2
    elem_x_centers = np.mean(mesh_s.nodes[mesh_s.elements][:,:,0], axis=1)
    elem_y_centers = np.mean(mesh_s.nodes[mesh_s.elements][:,:,1], axis=1)
    mid_elems = np.where(np.isclose(elem_x_centers, elem_x_centers[np.argmin(np.abs(elem_x_centers - mid_x))]))[0]
    
    y_coords = elem_y_centers[mid_elems]
    tau_xy_fem = stresses_s[mid_elems, 2]
    
    actual_x = elem_x_centers[mid_elems[0]]
    V = shear_force(actual_x, L_shear, q)
    y_ana = np.linspace(0, H, 100)
    tau_ana = timoshenko_shear_stress(y_ana, V, I, t, H)
    
    # Plot 6: Shear stress vs thickness
    plt.figure()
    plt.plot(tau_xy_fem, y_coords, 'o', label='FEM Q4 Elements')
    plt.plot(tau_ana, y_ana, 'k-', label='Timoshenko Theory')
    plt.ylabel('y coordinate across thickness (m)')
    plt.xlabel('Shear Stress $\\tau_{xy}$ (Pa)')
    plt.title(f'Shear Stress Distribution at x={actual_x:.2f} (AR=5)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/shear_stress_vs_thickness.png')
    plt.close()

    # ---------------------------------------------------------
    # PART C: Parametric Investigation (AR 2 to 50)
    # ---------------------------------------------------------
    print("Running Parametric Study...")
    ars = [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
    fem_disp_A = []
    
    for ar in ars:
        L_p = ar * H
        mesh_p = Mesh(L_p, H, int(8 * ar), 8)
        solver_p = FEMSolver(mesh_p, mat, thickness=t)
        U_p, _ = solver_p.solve_static(q)
        node_A = np.argmin(np.sqrt((mesh_p.nodes[:, 0] - L_p/2)**2 + (mesh_p.nodes[:, 1] - 0)**2))
        fem_disp_A.append(np.abs(U_p[2 * node_A + 1]))

    # Plot 7: Disp at A vs Aspect Ratio
    plt.figure()
    plt.plot(ars, fem_disp_A, 'o-b', label='FEM Disp at A')
    plt.xlabel('Aspect Ratio (L/H)')
    plt.ylabel('Absolute Displacement at Point A (m)')
    plt.title('Displacement at Point A vs Aspect Ratio')
    plt.grid(True)
    plt.savefig('plots/disp_at_A_vs_aspect_ratio.png')
    plt.close()

    # ---------------------------------------------------------
    # PART D: Modal Analysis (Aspect Ratio = 50)
    # ---------------------------------------------------------
    L_m = 50.0 * H  # Maintain AR=50
    print(f"Running Modal Analysis (AR={L_m/H})...")
    ny_modal = 8 # Converged thickness elements
    mesh_m = Mesh(L_m, H, 50, ny_modal)
    solver_m = FEMSolver(mesh_m, mat, thickness=t)
    freqs, eigs = solver_m.solve_modal(num_modes=5)
    
    with open('plots/frequencies.txt', 'w') as f:
        f.write("First 5 Natural Frequencies (Hz):\n")
        for i, fr in enumerate(freqs):
            f.write(f"Mode {i+1}: {fr:.3f} Hz\n")
            
    # Plot 8: Mode Shapes
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(5, 1, i+1)
        mode_shape = eigs[:, i]
        scale = 0.2 * H / np.max(np.abs(mode_shape))
        def_nodes = mesh_m.nodes + scale * mode_shape.reshape(-1, 2)
        for elem in mesh_m.elements:
            nodes = np.append(elem, elem[0])
            plt.plot(def_nodes[nodes, 0], def_nodes[nodes, 1], 'b-', lw=0.5)
        plt.title(f"Mode {i+1}: {freqs[i]:.2f} Hz")
        plt.axis('equal'); plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/mode_shapes.png')
    plt.close()

    # Plot 9: Modal Convergence for 5th mode
    m_nx_list = [10, 20, 40, 80, 160]
    freq_5th = []
    num_e_m = []
    for nx in m_nx_list:
        m = Mesh(L_m, H, nx, ny_modal)
        s = FEMSolver(m, mat, thickness=t)
        f, _ = s.solve_modal(num_modes=5)
        freq_5th.append(f[4])
        num_e_m.append(m.nelems)

    plt.figure()
    plt.plot(num_e_m, freq_5th, 'o-r')
    plt.xlabel('Number of Elements')
    plt.ylabel('Frequency of 5th Mode (Hz)')
    plt.title('Modal Convergence (5th Natural Mode)')
    plt.grid(True)
    plt.savefig('plots/modal_convergence.png')
    plt.close()

    print("All analyses completed successfully! Plots are saved in the 'plots/' directory.")

if __name__ == '__main__':
    generate_plots()
