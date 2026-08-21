import numpy as np
import matplotlib.pyplot as plt
import os
from fem_solver import Material, Mesh, FEMSolver
from analytical_solutions import timoshenko_deflection, euler_bernoulli_deflection

def run_parametric():
    os.makedirs('plots', exist_ok=True)
    
    aspect_ratios = [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
    H = 1.0
    t = 0.1
    E = 210e9
    nu = 0.3
    rho = 7850
    q = -10000.0
    G = E / (2 * (1 + nu))
    k = 5/6
    I = t * H**3 / 12
    A = t * H

    mat = Material(E, nu, rho)
    
    fem_deflections = []
    eb_deflections = []
    timo_deflections = []

    ny = 8

    for ar in aspect_ratios:
        L = ar * H
        nx = int(ny * ar)
        
        mesh = Mesh(L, H, nx, ny)
        solver = FEMSolver(mesh, mat, thickness=t)
        U, _ = solver.solve_static(q)
        
        # Tip deflection at mid-height
        dist = np.sqrt((mesh.nodes[:, 0] - L)**2 + (mesh.nodes[:, 1] - H/2)**2)
        tip_node = np.argmin(dist)
        v_fem = np.abs(U[2 * tip_node + 1])
        
        v_eb = np.abs(euler_bernoulli_deflection(L, L, E, I, q))
        v_timo = np.abs(timoshenko_deflection(L, L, E, G, I, A, k, q))
        
        fem_deflections.append(v_fem)
        eb_deflections.append(v_eb)
        timo_deflections.append(v_timo)
        print(f"AR={ar}: FEM={v_fem:.5e}, EB={v_eb:.5e}, Timo={v_timo:.5e}")

    fem_deflections = np.array(fem_deflections)
    eb_deflections = np.array(eb_deflections)
    timo_deflections = np.array(timo_deflections)

    # Plot ratios vs Aspect Ratio
    plt.figure(figsize=(8, 5))
    plt.plot(aspect_ratios, fem_deflections / eb_deflections, 'o-', label='FEM / Euler-Bernoulli')
    plt.plot(aspect_ratios, timo_deflections / eb_deflections, 's--', label='Timoshenko / Euler-Bernoulli')
    plt.axhline(1.0, color='k', linestyle=':', label='Euler-Bernoulli Reference')
    plt.xlabel('Aspect Ratio (L/H)')
    plt.ylabel('Normalized Tip Deflection')
    plt.title('Parametric Study: Effect of Aspect Ratio on Deflection')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/parametric_study.png')
    plt.close()

if __name__ == '__main__':
    run_parametric()
