import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from fem_solver import Material, Mesh, FEMSolver

def main():
    print("="*50)
    print("   2D FEM Cantilever Beam Solver (Interactive)")
    print("="*50)
    
    # Gathering user input with defaults
    try:
        print("\n--- Geometry Parameters ---")
        L_str = input("Enter beam length (L) in meters [default=10.0]: ")
        L = float(L_str) if L_str.strip() else 10.0
        
        H_str = input("Enter beam height (H) in meters [default=1.0]: ")
        H = float(H_str) if H_str.strip() else 1.0
        
        t_str = input("Enter beam thickness (t) in meters [default=0.1]: ")
        t = float(t_str) if t_str.strip() else 0.1
        
        print("\n--- Mesh Parameters ---")
        nx_str = input("Enter number of elements in x-direction (nx) [default=40]: ")
        nx = int(nx_str) if nx_str.strip() else 40
        
        ny_str = input("Enter number of elements in y-direction (ny) [default=8]: ")
        ny = int(ny_str) if ny_str.strip() else 8
        
        print("\n--- Material & Load Parameters ---")
        E_str = input("Enter Young's Modulus (E) in Pa [default=210e9]: ")
        E = float(E_str) if E_str.strip() else 210e9
        
        nu_str = input("Enter Poisson's ratio (nu) [default=0.3]: ")
        nu = float(nu_str) if nu_str.strip() else 0.3
        
        q_str = input("Enter uniformly distributed load (q) in N/m (use negative for downward) [default=-10000]: ")
        q = float(q_str) if q_str.strip() else -10000.0

    except ValueError:
        print("\n[!] Invalid input detected. Falling back to default values.")
        L, H, t, nx, ny, E, nu, q = 10.0, 1.0, 0.1, 40, 8, 210e9, 0.3, -10000.0
        
    print("\nInitializing solver...")
    mat = Material(E=E, nu=nu)
    mesh = Mesh(L, H, nx, ny)
    solver = FEMSolver(mesh, mat, thickness=t)
    
    print("Assembling matrices and solving static problem...")
    U, strain_energy = solver.solve_static(q)
    
    # Calculate tip deflection at mid-height
    dist = np.sqrt((mesh.nodes[:, 0] - L)**2 + (mesh.nodes[:, 1] - H/2)**2)
    tip_node = np.argmin(dist)
    tip_deflection = U[2 * tip_node + 1]
    
    print("\n" + "="*50)
    print("                     RESULTS")
    print("="*50)
    print(f"Total Number of Elements: {mesh.nelems}")
    print(f"Total Degrees of Freedom: {solver.ndof}")
    print(f"Maximum Tip Deflection:   {tip_deflection:.5e} m")
    print(f"Total Strain Energy:      {strain_energy:.5e} J")
    print("="*50)
    
    # Interactive Plotting
    plot_choice = input("\nDo you want to visualize the deformed shape? (y/n) [default=y]: ")
    if not plot_choice.strip() or plot_choice.lower().startswith('y'):
        # Calculate a reasonable scale factor so deformation is visible
        max_u = np.max(np.abs(U))
        scale_factor = (0.1 * L) / max_u if max_u != 0 else 1.0
        
        deformed_nodes = mesh.nodes + scale_factor * U.reshape(-1, 2)
        
        plt.figure(figsize=(10, 4))
        # Prepare line segments for ultra-fast plotting
        lines_undeformed = [mesh.nodes[np.append(elem, elem[0])] for elem in mesh.elements]
        lines_deformed = [deformed_nodes[np.append(elem, elem[0])] for elem in mesh.elements]
        
        lc_und = LineCollection(lines_undeformed, colors='k', linestyles='--', linewidths=0.5, alpha=0.3)
        lc_def = LineCollection(lines_deformed, colors='b', linestyles='-', linewidths=1.0)
        
        ax = plt.gca()
        ax.add_collection(lc_und)
        ax.add_collection(lc_def)
        ax.autoscale()
            
        plt.title(f'Deformed Shape (Displacement Scaled by {scale_factor:.1f}x)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        print("\nClose the plot window to exit the program.")
        plt.show()

if __name__ == '__main__':
    main()
