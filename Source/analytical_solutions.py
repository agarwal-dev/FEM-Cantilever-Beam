import numpy as np

def euler_bernoulli_deflection(x, L, E, I, q):
    """
    Computes the transverse deflection of a cantilever beam under uniformly distributed load
    using Euler-Bernoulli beam theory.
    
    Parameters:
    x (float or ndarray): Position along the beam (0 <= x <= L)
    L (float): Length of the beam
    E (float): Young's modulus
    I (float): Area moment of inertia
    q (float): Uniformly distributed load (force per unit length, positive downwards)
    
    Returns:
    float or ndarray: Deflection (positive downwards)
    """
    return q * (x**2 * (6 * L**2 - 4 * L * x + x**2)) / (24 * E * I)

def timoshenko_deflection(x, L, E, G, I, A, k, q):
    """
    Computes the transverse deflection of a cantilever beam under uniformly distributed load
    using Timoshenko beam theory.
    
    Parameters:
    x (float or ndarray): Position along the beam (0 <= x <= L)
    L (float): Length of the beam
    E (float): Young's modulus
    G (float): Shear modulus
    I (float): Area moment of inertia
    A (float): Cross-sectional area
    k (float): Shear correction factor (usually 5/6 for rectangular section)
    q (float): Uniformly distributed load (force per unit length, positive downwards)
    
    Returns:
    float or ndarray: Deflection (positive downwards)
    """
    v_eb = euler_bernoulli_deflection(x, L, E, I, q)
    v_shear = q * (L * x - x**2 / 2) / (k * A * G)
    return v_eb + v_shear

def bending_moment(x, L, q):
    """
    Computes the bending moment of a cantilever beam under uniformly distributed load.
    
    Parameters:
    x (float or ndarray): Position along the beam
    L (float): Length of the beam
    q (float): Uniformly distributed load
    
    Returns:
    float or ndarray: Bending moment
    """
    return -q * (L - x)**2 / 2

def shear_force(x, L, q):
    """
    Computes the shear force of a cantilever beam under uniformly distributed load.
    """
    return q * (L - x)

def timoshenko_shear_stress(y, V, I, t, H):
    """
    Computes the shear stress across the cross-section using Timoshenko theory.
    
    Parameters:
    y (float or ndarray): Vertical position from bottom (0 <= y <= H)
    V (float): Shear force at the section
    I (float): Area moment of inertia
    t (float): Thickness
    H (float): Height
    
    Returns:
    float or ndarray: Shear stress
    """
    # Distance from neutral axis
    y_na = y - H / 2
    # First moment of area Q
    Q = t / 2 * ((H / 2)**2 - y_na**2)
    return V * Q / (I * t)

def analytical_strain_energy_eb(L, E, I, q):
    """
    Computes the total strain energy using Euler-Bernoulli beam theory.
    U = \int_0^L (M^2 / (2EI)) dx
    M(x) = -q(L-x)^2 / 2
    M^2 = q^2 (L-x)^4 / 4
    \int (L-x)^4 dx = L^5 / 5
    U = q^2 L^5 / (40 EI)
    """
    return (q**2 * L**5) / (40 * E * I)

def analytical_strain_energy_timoshenko(L, E, G, I, A, k, q):
    """
    Computes the total strain energy using Timoshenko beam theory.
    U = U_bending + U_shear
    U_shear = \int_0^L (V^2 / (2kAG)) dx
    V(x) = q(L-x)
    V^2 = q^2(L-x)^2
    \int (L-x)^2 dx = L^3 / 3
    U_shear = q^2 L^3 / (6kAG)
    """
    U_b = analytical_strain_energy_eb(L, E, I, q)
    U_s = (q**2 * L**3) / (6 * k * A * G)
    return U_b + U_s
