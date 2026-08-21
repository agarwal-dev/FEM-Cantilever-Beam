import numpy as np
import matplotlib.pyplot as plt
import os
from fem_solver import Material, Mesh, FEMSolver
from analytical_solutions import (
    timoshenko_deflection, euler_bernoulli_deflection, 
    analytical_strain_energy_eb, analytical_strain_energy_timoshenko
)

def run_validation():
    os.makedirs('plots', exist_ok=True)
    
    # Parameters
    L = 50.0  # Aspect ratio 50
    H = 1.0
    t = 0.1
    E = 210e9
    nu = 0.3
    rho = 7850
    q = -10000.0  # N/m
    G = E / (2 * (1 + nu))
    I = t * H**3 / 12
    A = t * H
    k = 5/6

    # Analytical values at Point A (x = L/2)
    deflection_eb = euler_bernoulli_deflection(L/2, L, E, I, q)
    deflection_timo = timoshenko_deflection(L/2, L, E, G, I, A, k, q)
    energy_eb = analytical_strain_energy_eb(L, E, I, q)
    energy_timo = analytical_strain_energy_timoshenko(L, E, G, I, A, k, q)

    # Mesh refinement
    meshes = [(10, 2), (20, 4), (40, 8), (80, 16), (160, 32)]
    fem_deflections = []
    fem_energies = []
    num_elems = []

    mat = Material(E, nu, rho)

    for nx, ny in meshes:
        mesh = Mesh(L, H, nx, ny)
        solver = FEMSolver(mesh, mat, thickness=t)
        U, strain_energy = solver.solve_static(q)
        
        # Deflection at Point A: find node at x=L/2, y=0
        dist = np.sqrt((mesh.nodes[:, 0] - L/2)**2 + (mesh.nodes[:, 1] - 0)**2)
        node_A = np.argmin(dist)
        
        deflection_A = U[2 * node_A + 1]  # Y-displacement
        
        fem_deflections.append(deflection_A)
        fem_energies.append(strain_energy)
        num_elems.append(mesh.nelems)
        
        print(f"Mesh {nx}x{ny}: Point A Deflection = {deflection_A:.5e}, Strain Energy = {strain_energy:.5e}")

    # Plot 1: Point A Deflection Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(num_elems, np.abs(fem_deflections), 'o-', label='FEM Q4 Elements')
    plt.axhline(np.abs(deflection_eb), color='r', linestyle='--', label='Euler-Bernoulli')
    plt.axhline(np.abs(deflection_timo), color='g', linestyle='-.', label='Timoshenko')
    plt.xlabel('Number of Elements')
    plt.ylabel('Absolute Deflection at Point A (m)')
    plt.title('Convergence of Deflection at Point A (x=L/2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/validation_deflection.png')
    plt.close()

    # Plot 2: Strain Energy Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(num_elems, fem_energies, 'o-', label='FEM Q4 Elements')
    plt.axhline(energy_eb, color='r', linestyle='--', label='Euler-Bernoulli')
    plt.axhline(energy_timo, color='g', linestyle='-.', label='Timoshenko')
    plt.xlabel('Number of Elements')
    plt.ylabel('Total Strain Energy (J)')
    plt.title('Convergence of Total Strain Energy')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/validation_energy.png')
    plt.close()

    # Deformed Shape for finest mesh
    nx, ny = meshes[-1]
    mesh = Mesh(L, H, nx, ny)
    solver = FEMSolver(mesh, mat, thickness=t)
    U, _ = solver.solve_static(q)
    
    scale_factor = 0.1 * L / np.max(np.abs(U))  # Scale deformation for visibility
    deformed_nodes = mesh.nodes + scale_factor * U.reshape(-1, 2)
    
    plt.figure(figsize=(12, 3))
    for elem in mesh.elements:
        nodes = np.append(elem, elem[0])  # close the loop
        plt.plot(mesh.nodes[nodes, 0], mesh.nodes[nodes, 1], 'k--', lw=0.5, alpha=0.5)
        plt.plot(deformed_nodes[nodes, 0], deformed_nodes[nodes, 1], 'b-', lw=1.0)
    plt.title(f'Deformed Shape (Scale factor: {scale_factor:.1f})')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.savefig('plots/deformed_shape.png')
    plt.close()

if __name__ == '__main__':
    run_validation()
