# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:22:18 2025

@author: MBESSJM1
"""

import numpy as np
from scipy.optimize import least_squares

def argon_visc(T, P):
    """
    Calculates the viscosity of argon at a given temperature (T) and pressure (P).

    Parameters:
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pa.

    Returns:
        float: Viscosity in microPascal-seconds (uPa.s).
    """
    # Convert Pressure to MPa
    P=P/1e6
    # Coefficients from Younglove and Hanley (1986)
    gv = np.array([
        -0.8973188257E+05,
        0.8259113473E+05,
        -0.2766475915E+05,
        0.3068539784E+04,
        0.4553103615E+03,
        -0.1793443839E+03,
        0.2272225106E+02,
        -0.1350672796E+1,
        0.3183693230E-01
    ])

    # Calculate mu0
    mu0 = (
        gv[0] * T**(-1) +
        gv[1] * T**(-2/3) +
        gv[2] * T**(-1/3) +
        gv[3] +
        gv[4] * T**(1/3) +
        gv[5] * T**(2/3) +
        gv[6] * T +
        gv[7] * T**(4/3) +
        gv[8] * T**(5/3)
    )

    # Solve for rho
    def EOS(rho, T, P):
        R = 8.3144621 / 1000  # Gas constant in L MPa K^-1 mol^-1
        G = np.array([
            -0.65697312940E-04,
            0.18229578010E-01,
            -0.36494701410E+00,
            0.12320121070E+02,
            -0.86135782740E+03,
            0.79785796910E-05,
            -0.29114891100E-02,
            0.75818217580E+00,
            0.87804881690E+03,
            0.14231459890E-07,
            0.16741461310E-03,
            -0.32004479090E-01,
            0.25617663720E-05,
            -0.54759349410E-04,
            -0.45050320580E-01,
            0.20132546530E-05,
            -0.16789412730E-07,
            0.42073292710E-04,
            -0.54442129960E-06,
            -0.80048550110E+03,
            -0.13193042010E+05,
            -0.49549239300E+01,
            0.80921321770E+04,
            -0.98701040610E-02,
            0.20204415620E+00,
            -0.16374172050E-04,
            -0.70389441360E-01,
            -0.11543245390E-07,
            0.15559901170E-05,
            -0.14921785360E-10,
            -0.10013560710E-08,
            0.29339632160E-07
        ])
        g = -0.0055542372

        return (
            P - (
                rho * R * T +
                rho**2 * (G[0] * T + G[1] * T**(1/2) + G[2] + G[3] / T + G[4] / T**2) +
                rho**3 * (G[5] * T + G[6] + G[7] / T + G[8] / T**2) +
                rho**4 * (G[9] * T + G[10] + G[11] / T) +
                rho**5 * G[12] +
                rho**6 * (G[13] / T + G[14] / T**2) +
                rho**7 * (G[15] / T) +
                rho**8 * (G[16] / T + G[17] / T**2) +
                rho**9 * (G[18] / T**2) +
                rho**3 * (G[19] / T**2 + G[20] / T**3) * np.exp(g * rho**2) +
                rho**5 * (G[21] / T**2 + G[22] / T**4) * np.exp(g * rho**2) +
                rho**7 * (G[23] / T**2 + G[24] / T**3) * np.exp(g * rho**2) +
                rho**9 * (G[25] / T**2 + G[26] / T**4) * np.exp(g * rho**2) +
                rho**11 * (G[27] / T**2 + G[28] / T**3) * np.exp(g * rho**2) +
                rho**13 * (G[29] / T**2 + G[30] / T**3 + G[31] / T**4) * np.exp(g * rho**2)
            )
        )

    rho0 = 10  # Initial guess
    bounds = (0, 100)
    result = least_squares(EOS, rho0, bounds=bounds, args=(T, P))
    rho = result.x[0]

    # Additional coefficients
    XV = np.zeros(13)
    XV[0] = 0.5927733783E+00
    XV[2] = -0.2698417165E-01
    XV[4] = -0.3958508120E+04
    XV[6] = -0.2633471347E+01
    XV[8] = -0.3811869019E-04
    XV[10] = -0.53858111481E+01
    XV[12] = -0.13288934114E+01
    XV[1] = -0.42512211698E+02
    XV[3] = 0.3727762288E+02
    XV[5] = 0.36361308111E-02
    XV[7] = 0.2936563322E+03
    XV[9] = 0.445191111164E-01
    XV[11] = -0.1115054926E-01

    f1 = XV[0] + XV[1] / T
    f2 = XV[2] + XV[3] / T + XV[4] / T**2
    f3 = XV[5] + XV[6] / T + XV[7] / T**2
    f4 = XV[8] + XV[9] / T + XV[10] / T**2
    f5 = XV[11] + XV[12] / T

    # Calculate viscosity
    mu = (
        mu0 +
        f1 * rho / (1 + f5 * rho) +
        f2 * rho**2 / (1 + f5 * rho) +
        f3 * rho**3 / (1 + f5 * rho) +
        f4 * rho**4 / (1 + f5 * rho)
    )

    return mu, rho



def Argon_Z(T, P):
    """
    Calculates density (rho), compressibility factor (Z), dZdP, 
    and isothermal compressibility (C) using Gosman et al. (1969).

    Parameters:
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pascal.

    Returns:
        tuple: Z, rho, dZdP, C
    """
    # Convert Pa to atm
    P_atm = P * 9.8692e-6
    P_plus=(P+1)*9.8692e-6
    P_minus=(P-1)*9.8692e-6
    # Gas constant in atm L / (g mol K)
    R = 0.0820535

    # Coefficients for the equations from Gosman et al. (1969)
    n = np.array([
        0.25978374e-2,
        -0.89735867,
        -0.67273638e2,
        -0.26494177e4,
        0.97631231e7,
        0.70478556e-4,
        -0.46767764e-2,
        0.22640765e-5,
        0.48141071e3,
        0.64565346e5,
        -0.11485282e8,
        -0.64835488,
        0.46524812e3,
        0.10933578e5,
        0.69439530e-6,
        0.48e-2
    ])

    def EOS(rho, T, n, R, P):
        """Equation 39 from Gosman et al. (1969)."""
        term1 = rho * R * T
        term2 = rho**2 * (n[0] * T + n[1] + n[2] / T + n[3] / T**2 + n[4] / T**4)
        term3 = rho**3 * (n[5] * T + n[6])
        term4 = rho**4 * n[7] * T
        term5 = rho**3 * (n[8] / T**2 + n[9] / T**3 + n[10] / T**4) * np.exp(-n[15] * rho**2)
        term6 = rho**5 * (n[11] / T**2 + n[12] / T**3 + n[13] / T**4) * np.exp(-n[15] * rho**2)
        term7 = rho**6 * n[14]
        return term1 + term2 + term3 + term4 + term5 + term6 + term7 - P

    def z_fact(T, rho, n):
        """Equation 40 from Gosman et al. (1969)."""
        term1 = 1
        term2 = (rho / R) * (n[0] + n[1] / T + n[2] / T**2 + n[3] / T**3 + n[4] / T**5)
        term3 = (rho**2 / R) * (n[5] + n[6] / T)
        term4 = (rho**3 / R) * n[7]
        term5 = (rho**2 / R) * (n[8] / T**3 + n[9] / T**4 + n[10] / T**5) * np.exp(-n[15] * rho**2)
        term6 = (rho**4 / R) * (n[11] / T**3 + n[12] / T**4 + n[13] / T**5) * np.exp(-n[15] * rho**2)
        term7 = (rho**5 / R) * n[14] / T
        return term1 + term2 + term3 + term4 + term5 + term6 + term7

    # Solve for rho
    rho0 = 10  # Initial guess
    bounds = (0, 100)
    result = least_squares(EOS, rho0, bounds=bounds, args=(T, n, R, P_atm))
    rho = result.x[0]

    # Perturbed densities for dZdP
    result_plus = least_squares(EOS, rho, bounds=bounds, args=(T, n, R, P_plus))
    rho_plus = result_plus.x[0]

    result_minus = least_squares(EOS, rho, bounds=bounds, args=(T, n, R, P_minus))
    rho_minus = result_minus.x[0]

    # Calculate Z and dZdP
    Z = z_fact(T, rho, n)
    Z_plus = z_fact(T, rho_plus, n)
    Z_minus = z_fact(T, rho_minus, n)
    dZdP = (Z_plus - Z_minus) / (2)  # in 1/Pa

    # Isothermal compressibility
    C = (1 / P) - (1 / Z) * dZdP  # in 1/Pa

    return Z, rho, dZdP, C
