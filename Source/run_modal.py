import numpy as np
import matplotlib.pyplot as plt
import os
from fem_solver import Material, Mesh, FEMSolver

def run_modal():
    os.makedirs('plots', exist_ok=True)
    
    L = 50.0  # Aspect ratio 50
    H = 1.0
    t = 0.1
    E = 210e9
    nu = 0.3
    rho = 7850
    mat = Material(E, nu, rho)
    
    # 1. Base Modal Analysis to get mode shapes
    try:
        ny_input = input("Enter suitable number of elements in thickness direction (ny) from static analysis [default=8]: ")
        ny = int(ny_input) if ny_input.strip() else 8
    except Exception:
        ny = 8
    nx = 50  # reasonable starting nx for AR=50
    mesh = Mesh(L, H, nx, ny)
    solver = FEMSolver(mesh, mat, thickness=t)
    
    frequencies, eigenvectors = solver.solve_modal(num_modes=5)
    
    print("First 5 Natural Frequencies (Hz):")
    for i, freq in enumerate(frequencies):
        print(f"Mode {i+1}: {freq:.3f} Hz")
        
    # Plot Mode Shapes
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(5, 1, i+1)
        mode_shape = eigenvectors[:, i]
        # Scale for visualization
        scale = 0.2 * H / np.max(np.abs(mode_shape))
        deformed_nodes = mesh.nodes + scale * mode_shape.reshape(-1, 2)
        
        # Plot outline only for speed
        plt.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], 'k.', markersize=1, alpha=0.2)
        
        # Plot perimeter of deformed shape
        for elem in mesh.elements:
            nodes = np.append(elem, elem[0])
            plt.plot(deformed_nodes[nodes, 0], deformed_nodes[nodes, 1], 'b-', lw=0.5)
            
        plt.title(f"Mode {i+1}: {frequencies[i]:.2f} Hz")
        plt.axis('equal')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('plots/mode_shapes.png')
    plt.close()
    
    # 2. Convergence study for 5th mode
    print(f"\nPerforming convergence study for 5th mode with fixed ny={ny}")
    meshes = [(10, ny), (20, ny), (40, ny), (80, ny), (160, ny)]
    freq_mode5 = []
    num_elems = []
    
    for m_nx, m_ny in meshes:
        m = Mesh(L, H, m_nx, m_ny)
        s = FEMSolver(m, mat, thickness=t)
        f, _ = s.solve_modal(num_modes=5)
        freq_mode5.append(f[4])
        num_elems.append(m.nelems)
        print(f"Mesh {m_nx}x{m_ny}: Mode 5 Freq = {f[4]:.3f} Hz")
        
    plt.figure(figsize=(8, 5))
    plt.plot(num_elems, freq_mode5, 'o-r')
    plt.xlabel('Number of Elements')
    plt.ylabel('Frequency of 5th Mode (Hz)')
    plt.title('Convergence Study of 5th Natural Frequency')
    plt.grid(True)
    plt.savefig('plots/modal_convergence.png')
    plt.close()

if __name__ == '__main__':
    run_modal()
