import numpy as np
import matplotlib.pyplot as plt
import os
from fem_solver import Material, Mesh, FEMSolver
from analytical_solutions import timoshenko_shear_stress, shear_force

def run_shear():
    os.makedirs('plots', exist_ok=True)
    
    L = 5.0  # Aspect ratio 5
    H = 1.0
    t = 0.1
    E = 210e9
    nu = 0.3
    rho = 7850
    q = -10000.0
    I = t * H**3 / 12
    
    # High resolution in y-direction to capture parabolic distribution well
    nx, ny = 40, 20
    
    mat = Material(E, nu, rho)
    mesh = Mesh(L, H, nx, ny)
    solver = FEMSolver(mesh, mat, thickness=t)
    U, _ = solver.solve_static(q)
    
    _, stresses = solver.compute_stress_strain(U)
    
    # We want elements around x = L/2
    mid_x = L / 2
    
    # Find x-coordinates of element centers
    elem_x_centers = np.zeros(mesh.nelems)
    elem_y_centers = np.zeros(mesh.nelems)
    
    for i, elem in enumerate(mesh.elements):
        coords = mesh.nodes[elem]
        elem_x_centers[i] = np.mean(coords[:, 0])
        elem_y_centers[i] = np.mean(coords[:, 1])
        
    # Find elements in the column closest to mid_x
    diff = np.abs(elem_x_centers - mid_x)
    min_diff = np.min(diff)
    mid_span_elems = np.where(np.isclose(diff, min_diff))[0]
    
    y_coords = elem_y_centers[mid_span_elems]
    tau_xy_fem = stresses[mid_span_elems, 2]  # tau_xy is the 3rd component of stress vector
    
    # Actual x location of these elements
    actual_x = elem_x_centers[mid_span_elems[0]]
    
    # Analytical solution
    V = shear_force(actual_x, L, q)
    y_analytical = np.linspace(0, H, 100)
    tau_xy_analytical = timoshenko_shear_stress(y_analytical, V, I, t, H)
    
    plt.figure(figsize=(6, 8))
    plt.plot(tau_xy_fem, y_coords, 'o', label='FEM Q4 Elements')
    plt.plot(tau_xy_analytical, y_analytical, 'k-', label='Timoshenko Theory')
    plt.ylabel('y coordinate across thickness (m)')
    plt.xlabel('Shear Stress $\\tau_{xy}$ (Pa)')
    plt.title(f'Shear Stress Distribution at x = {actual_x:.2f} m')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/shear_stress.png')
    plt.close()

if __name__ == '__main__':
    run_shear()
