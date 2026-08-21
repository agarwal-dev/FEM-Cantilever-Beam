# 2D FEM Cantilever Beam Analysis

Structural analysis and dynamic simulation of a 2D continuum cantilever beam using Q4 quadrilateral finite elements, developed as a B.Tech FEM project.

## Project Overview

The project implements a custom 2D Finite Element Method (FEM) solver in Python using 4-node bilinear quadrilateral (Q4) isoparametric elements with $2 \times 2$ Gauss Quadrature numerical integration under plane stress conditions.

The solver performs static deflection analysis, stress/strain field visualization, transverse shear evaluation, aspect-ratio parametric studies, dynamic modal analysis, and analytical theory validation.

## Key Features

- 4-node bilinear isoparametric quadrilateral (Q4) elements
- $2 \times 2$ Gaussian Quadrature numerical integration
- Sparse matrix assembly (`scipy.sparse`) for high efficiency
- Static deflection and total strain energy calculation
- Bending stress ($\sigma_{xx}$) and strain ($\epsilon_{xx}$) contour field visualization
- Transverse shear stress profile extraction across beam thickness
- Parametric aspect ratio investigation ($L/H = 2$ to $50$)
- Dynamic modal analysis (first 5 natural frequencies & mode shapes)
- Benchmark validation against Euler-Bernoulli and Timoshenko beam theories

## Design & Calculations

- Element type: **Q4 Isoparametric Quadrilateral**
- Integration points: **$2 \times 2$ Gauss Quadrature** ($\xi, \eta = \pm 1/\sqrt{3}$)
- Young's Modulus ($E$): **210 GPa** (Steel)
- Poisson's Ratio ($\nu$): **0.3**
- Density ($\rho$): **7850 kg/m³**
- Applied load ($q$): **-10 kN/m** (Uniform transverse load)
- Slender beam dimensions: **$L = 50\text{ m}$, $H = 1\text{ m}$, $t = 0.1\text{ m}$** (Aspect Ratio = 50)
- Deep beam dimensions: **$L = 5\text{ m}$, $H = 1\text{ m}$, $t = 0.1\text{ m}$** (Aspect Ratio = 5)
- Calculated tip deflection: **4.430 m** (Custom FEM) vs **4.464 m** (Euler-Bernoulli Analytical)

## Simulation & Code Structure

The code is organized into modular scripts as well as a monolithic submission script:

- `submission_code.py` — Monolithic script executing all simulation modules
- `main.py` — Interactive CLI solver for custom geometry, materials, and mesh
- `fem_solver.py` — Core OOP classes (`Material`, `Mesh`, `FEMSolver`)
- `analytical_solutions.py` — Euler-Bernoulli & Timoshenko closed-form solutions
- `report.md` — Detailed academic project report

## Tools Used

- Python 3.x
- NumPy (Matrix operations)
- SciPy (Sparse solvers & eigenvalue analysis)
- Matplotlib (Contour plotting & visualization)
- SolidWorks & ANSYS (CAD modeling & commercial benchmarking)

## Applications

- Structural beam analysis in civil and mechanical engineering
- Validation of 2D continuum elements against 1D beam theories
- Educational FEM implementation and mesh convergence studies
- Dynamic free-vibration mode shape visualization

## Preview

Result plots, convergence graphs, stress/strain contours, and mode shapes are included in this repository under the `plots/` directory.

## Project Status

**Completed**
