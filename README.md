# 2D Finite Element Analysis of a Cantilever Beam (Q4 Quad Elements)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![SciPy](https://img.shields.io/badge/SciPy-Sparse%20Solvers-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end, high-performance **2D Finite Element Method (FEM) Solver** written in Python using 4-node bilinear quadrilateral (Q4) elements and $2 \times 2$ Gaussian Quadrature integration. 

This repository provides full static deflection analysis, stress/strain field visualization, transverse shear stress evaluation across thick/thin beam limits, parametric aspect-ratio studies, dynamic modal analysis (natural frequencies & mode shapes), and benchmark validation against **Euler-Bernoulli Beam Theory**, **Timoshenko Beam Theory**, and **ANSYS Commercial Simulation**.

---

## 📑 Table of Contents
- [Overview & Methodology](#-overview--methodology)
- [Mathematical Formulation](#-mathematical-formulation)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Features & Workflow](#-features--workflow)
- [Results & Visualizations](#-results--visualizations)
  - [1. Static Deflection & Energy Convergence](#1-static-deflection--energy-convergence)
  - [2. Stress & Strain Contours](#2-stress--strain-contours)
  - [3. Transverse Shear Stress Distribution](#3-transverse-shear-stress-distribution)
  - [4. Parametric Study (Aspect Ratio L/H)](#4-parametric-study-aspect-ratio-lh)
  - [5. Modal Dynamic Analysis](#5-modal-dynamic-analysis)
- [Validation & Benchmarks](#-validation--benchmarks)
- [ANSYS Verification Guide](#-ansys-verification-guide)
- [License](#-license)

---

## 📌 Overview & Methodology

The custom Python solver models a 2D continuum cantilever beam subjected to a uniformly distributed transverse load $q$ along its top edge ($y = H$), with fixed boundary conditions at $x = 0$.

Key simulation characteristics:
* **Element Type**: 4-Node Bilinear Quadrilateral (Q4) isoparametric elements.
* **Integration Scheme**: Full $2 \times 2$ Gauss Quadrature ($\xi, \eta = \pm 1/\sqrt{3}$).
* **Material Model**: Linear Isotropic Elasticity under Plane Stress formulation.
* **Sparse Matrix Solvers**: Efficient assembly using COO sparse representation and direct linear solve via `scipy.sparse.linalg.spsolve`.
* **Dynamic Eigenvalue Solver**: Generalized sparse eigenvalue solver (`scipy.sparse.linalg.eigsh`) for undamped free-vibration modal analysis.

---

## 📐 Mathematical Formulation

### 1. Q4 Isoparametric Shape Functions
For natural coordinates $\xi, \eta \in [-1, 1]$:
$$N_i(\xi, \eta) = \frac{1}{4}(1 + \xi_i \xi)(1 + \eta_i \eta), \quad i = 1, 2, 3, 4$$

### 2. Element Stiffness & Mass Matrices
$$\mathbf{K}_e = \int_{-1}^{1} \int_{-1}^{1} \mathbf{B}^T \mathbf{D} \mathbf{B} \cdot t \cdot \det(\mathbf{J}) \, d\xi \, d\eta$$

$$\mathbf{M}_e = \rho \int_{-1}^{1} \int_{-1}^{1} \mathbf{N}^T \mathbf{N} \cdot t \cdot \det(\mathbf{J}) \, d\xi \, d\eta$$

where $\mathbf{B}$ is the strain-displacement matrix, $\mathbf{D}$ is the plane stress constitutive matrix, $\mathbf{J}$ is the Jacobian matrix, $t$ is beam thickness, and $\rho$ is material density.

### 3. Constitutive Relations (Plane Stress)
$$\mathbf{D} = \frac{E}{1-\nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}$$

---

## 📁 Repository Structure

```text
├── README.md                      # Project documentation and user guide
├── report.md                      # Comprehensive academic technical report
├── submission_code.py             # Single monolithic script (All-in-one runner)
├── main.py                        # Interactive CLI solver for custom inputs
├── fem_solver.py                  # Core OOP modules: Material, Mesh, FEMSolver
├── analytical_solutions.py        # Analytical Euler-Bernoulli & Timoshenko functions
├── run_validation.py              # Convergence & deflection validation script
├── run_shear.py                   # Transverse shear stress investigation script
├── run_parametric.py             # Aspect ratio parametric study script
├── run_modal.py                   # Dynamic modal analysis script
├── Test bar.IGS                   # CAD Model (IGES format for ANSYS import)
├── Test bar.SLDPRT                # SolidWorks Part CAD Model
└── plots/                         # High-resolution output graphics & data
    ├── disp_at_A_convergence.png
    ├── energy_convergence.png
    ├── displacement_contour.png
    ├── stress_contour.png
    ├── strain_contour.png
    ├── shear_stress_vs_thickness.png
    ├── disp_at_A_vs_aspect_ratio.png
    ├── mode_shapes.png
    ├── modal_convergence.png
    └── frequencies.txt
