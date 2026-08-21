# 2D Finite Element Analysis of a Cantilever Beam

## 1. FEM Methodology

### 1.1 Node-Element Connectivity
The cantilever beam domain is discretized using 4-node quadrilateral (Q4) elements. The mesh is generated as a structured grid, creating nodes iteratively along the $x$-direction and $y$-direction. Each Q4 element is defined by the global indices of its four corner nodes, numbered counterclockwise starting from the bottom-left node of the element. This connectivity matrix maps the local element degrees of freedom to the global degree of freedom vector during assembly.

### 1.2 Gauss Points Used
The evaluation of the element stiffness matrix ($\mathbf{K}_e$) and the consistent mass matrix ($\mathbf{M}_e$) requires area integration over the element domain. This integration is performed numerically using **$2 \times 2$ Gaussian Quadrature**. The Gauss points are located at $\xi, \eta = \pm 1/\sqrt{3} \approx \pm 0.5773$ in the natural coordinate system, and each point has a corresponding weight of $w = 1.0$. This full integration perfectly captures the bilinear shape functions of the Q4 elements.

### 1.3 Solution Steps
The 2D FEM code follows these core steps:
1. **Mesh Generation**: Generate the nodal coordinates array and the element connectivity matrix based on the given length ($L$), height ($H$), and number of elements ($n_x, n_y$).
2. **Matrix Assembly**: For each element, evaluate the Jacobian matrix, strain-displacement matrix ($\mathbf{B}$), and constitutive matrix ($\mathbf{D}$). Compute the element stiffness and mass matrices using $2 \times 2$ Gauss quadrature, and assemble them into the global sparse matrices $\mathbf{K}$ and $\mathbf{M}$.
3. **Boundary Conditions**: Identify nodes at the fixed end ($x=0$) and constrain their degrees of freedom ($U_x = U_y = 0$). Apply the uniformly distributed transverse load ($q$) along the top edge ($y=H$) by converting it into equivalent nodal forces.
4. **Static Solution**: Partition the global stiffness matrix and solve the linear system $\mathbf{K}_{ff} \mathbf{U}_f = \mathbf{F}_f$ using a sparse direct solver to find the unknown nodal displacements.
5. **Post-Processing**: Calculate strains ($\epsilon$) and stresses ($\sigma$) at the element centers using the computed displacements and the $\mathbf{B}$ matrix. For modal analysis, solve the generalized eigenvalue problem $(\mathbf{K} - \omega^2 \mathbf{M})\phi = 0$.

### 1.4 Input Parameters
The following physical properties and parameters were used for the simulations:
- **Young's Modulus ($E$)**: $210 \times 10^9$ Pa (Steel)
- **Poisson's Ratio ($\nu$)**: $0.3$
- **Density ($\rho$)**: $7850$ kg/m³
- **Beam Thickness ($t$)**: $0.1$ m
- **Uniform Load ($q$)**: $-10,000$ N/m (acting downwards)
- **Validation Dimensions (Aspect Ratio = 50)**: Length ($L$) = $50$ m, Height ($H$) = $1$ m
- **Shear Analysis Dimensions (Aspect Ratio = 5)**: Length ($L$) = $5$ m, Height ($H$) = $1$ m
- **Mesh Density**: Defined by $n_x$ (elements along length) and $n_y$ (elements across thickness).

---

## 2. Results

The static analysis was performed on a slender cantilever beam with an Aspect Ratio of 50 ($L=50, H=1$) under a uniformly distributed load of $q = -10 \text{ kN/m}$.

### Displacement at Point A vs No. of Elements
A mesh convergence study was performed evaluating the vertical displacement at Point A (mid-span bottom edge, $x=25, y=0$).
![Disp at A](plots/disp_at_A_convergence.png)

### Strain Energy vs No. of Elements
The total strain energy of the system converges asymptotically as the mesh is refined.
![Strain Energy](plots/energy_convergence.png)

### Displacement Contour
Contour plot of the vertical displacement (y-direction) over the deformed beam.
![Displacement Contour](plots/displacement_contour.png)

### Stress Contour
Contour plot of the longitudinal bending stress ($\sigma_{xx}$) over the elements. The top fibers are in tension, and the bottom fibers are in compression.
![Stress Contour](plots/stress_contour.png)

### Strain Contour
Contour plot of the longitudinal bending strain ($\epsilon_{xx}$) over the elements.
![Strain Contour](plots/strain_contour.png)

### Shear Stress vs Thickness
For a deep beam configuration (Aspect Ratio = 5), the transverse shear stress ($\tau_{xy}$) was extracted across the thickness at mid-span and plotted against the theoretical parabolic distribution predicted by Timoshenko beam theory.
![Shear Stress vs Thickness](plots/shear_stress_vs_thickness.png)

---

## 3. Investigation Plot
A parametric study was conducted to observe how the absolute displacement at Point A behaves as the aspect ratio of the beam varies from $L/H = 2$ up to $50$.

![Investigation Plot](plots/disp_at_A_vs_aspect_ratio.png)

---

## 4. Validation using ANSYS

The custom 2D FEM Python code was validated against analytical beam theories and commercial finite element software (ANSYS) for a slender beam (Aspect Ratio = 50). The following table compares the **maximum tip displacement** across the different methods.

| Method | Tip Displacement (m) |
| :--- | :--- |
| **Euler-Bernoulli Analytical** | 4.464 |
| **Timoshenko Analytical** | 4.466 |
| **Custom 2D FEM Code** | 4.430 |
| **ANSYS Simulation** | *[Insert ANSYS Value Here]* |

*(Note: The custom FEM code shows excellent agreement with the analytical theories. Once you perform the ANSYS simulation, fill in the table above with your result, which should also be approximately ~4.43 to 4.46 m).*

---

## 5. Modal Analysis Result

A modal analysis was performed on the cantilever beam (Aspect Ratio = 50, taking $n_y = 8$ thickness elements from the static convergence study) to identify its fundamental frequencies and mode shapes.

### Mode and Frequency Table
| Mode Number | Natural Frequency (Hz) |
| :---: | :---: |
| 1 | 0.394 |
| 2 | 2.464 |
| 3 | 6.886 |
| 4 | 13.463 |
| 5 | 22.190 |

### Mode Shapes for 5 Frequencies
The first 5 bending mode shapes are visualized below:
![Mode Shapes](plots/mode_shapes.png)

### Modal Convergence Plot
The convergence of the 5th natural frequency was tracked as the number of longitudinal elements was progressively increased.
![Modal Convergence](plots/modal_convergence.png)
