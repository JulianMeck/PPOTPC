# -*- coding: utf-8 -*-


"""
Created on Mon Jan 20 15:48:58 2025

@author: Julian Mecklenburgh (University of Manchester)



PPO Permeability Processing
============================

Version history
---------------
v3  (original)
    - Simplified algebraic Bernabé forward model 
    - Nearest-neighbour lookup for Bernabé solver starting values
    - Nelder-Mead simplex optimizer for sinusoidal fitting
    - Parameter errors from bootstrap standard deviation only
    - Error on η when ξ=0 via analytical formula (diverges when A → 1)
    - Downstream storage capacity computed internally as bd = Dv × C(T,P)
Changes made by Pier-Carlo Giacomel
v4  
    [0] Bernabé forward model corrected: replaced the 
        algebraic approximation used in v3:
            A = sqrt((1+(2η+ξ)²) / ((1+η²)(1+(η+ξ)²)))
            φ = atan((ξ+η)/(1+η(η+ξ))) − atan(η)
        with the exact distributed (sinh/cosh) formulation from Bernabé
        (2006), matching the MATLAB reference (singlek_JMv2_5.m):
            val = ((1+i)/sqrt(η·ξ)·sinh((1+i)·sqrt(ξ/η))
                   + cosh((1+i)·sqrt(ξ/η)))⁻¹
            A = |val|,  φ = −angle(val)
        The approximation is only valid when in-sample storage (ξ)
        is negligible, explaining the discrepancies between
        v3 and MATLAB especially for the post-tested samples. Starting-value search updated from nearest-neighbour
        to griddata linear interpolation; cost function updated to the
        MATLAB log-ratio form: w·(log(A_th)/log(A_exp)−1)² + (1−w)·(φ_th−φ_exp)²

    [1] Sinusoidal fitting: Nelder-Mead → L-BFGS-B. Primary parameter
        errors now from Hessian at the optimum (sqrt(diag(inv(H)))),
        matching MATLAB fmincon. Bootstrap retained as fallback and for
        distribution plots.

    [2] η/ξ error estimation — hybrid Hessian/bootstrap strategy
        (function bern_hessian_errors + fallback in main loop):

        Primary (always tried first): numerical Hessian of the Berna
        cost function at the optimum.
          ξ=0: 1-D Hessian w.r.t. log(η) only, ξ pinned at boundary.
                Replaces analytical formula η_err ∝ 1/(1−A²) which
                diverges as A→1 (high-gain pretest samples).
          ξ>0: 2-D Hessian w.r.t. [log(η), log(ξ)].

        Fallback (if Hessian gives η_err/η > 100%, i.e. cost surface
        too flat — typically posttest samples at low gain):
          ξ=0: analytical formula (better than diverging Hessian)
          ξ>0: bootstrap std of η/ξ distributions, using
                the bootstrap samples already computed for the plots.

        Console prints which path was taken for each measurement.
        This replaces the bootstrap-only approach, which returned
        NaN for most posttest measurements in VS.3 due to bootstrap samples
        collapsing onto the ξ=0 boundary.

    [3] Downstream storage: user chooses bd (direct input, recommended for
        water) or Dv (bd = Dv×C(T,P) per measurement, recommended for
        argon).

    [4] ROI selection: two independent figure windows instead of reusing
        the same axes, preventing label overlap on repeated runs.
        Nomogram legend added.
v6 (this file)
  [1] simplified the fitting of pressure waves
  
  [2] Added a linear function to the fit of downstream to fit to account for leaks temperature drift
  
"""
import os
from tkinter import Tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from astropy.timeseries import LombScargle
import scipy.io
from scipy.spatial import KDTree
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
# from scipy.stats import norm  # kept for possible future use – not used in main calculation
from iapws import IAPWS95
from iapws import _iapws
from argon import Argon_Z
from argon import argon_visc
import numdifftools as nd

# Force-close any figures left over from a previous run in the same session,
# then give the event loop a moment to fully release tkinter resources before
# we open new dialogs.  Without this pause the filedialog can hang on re-runs.


plt.close('all')
try:
    plt.pause(0.2)          # let the old event loop drain
except Exception:
    pass

# Switch to an interactive backend only when necessary.
# NOTE: matplotlib.use() cannot change the backend after it has already been
# initialised in this session, so we guard the call carefully.
_backend = matplotlib.get_backend().lower()
if not (_backend.startswith('qt') or _backend.startswith('tk')):
    try:
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt   # re-import after backend switch
    except Exception:
        pass  # keep whatever backend is already active

print("=" * 60)
print("PPO Permeability Processing - Modified Version")
print("Using downstream storage capacity (bd) as direct input")
print("Dialogs and plots will appear ON TOP of other windows")
print("=" * 60)

# Helper function to create a topmost Tk window
def create_topmost_root():
    """Create a Tk root window that appears on top of all other windows"""
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Force window to top
    root.lift()  # Lift to top
    root.focus_force()  # Force focus
    return root

def make_figure_topmost(fig):
    """Force a matplotlib figure to appear on top"""
    try:
        fig.canvas.manager.window.attributes('-topmost', True)
        fig.canvas.manager.window.lift()
        fig.canvas.manager.window.focus_force()
    except:
        pass  # If this fails, continue anyway

# Function to read data file
def read_datafile():
    root = create_topmost_root()
    filename = filedialog.askopenfilename(title="Pick a datafile", filetypes=[("All files", "*.*")])
    root.destroy()
    return filename
def lookup_table():
    mat_data = scipy.io.loadmat('lookup.mat')
    A_lookup = mat_data['A']
    phi_lookup=mat_data['phi']
    eta_lookup=mat_data['eta']
    xi_lookup=mat_data['xi']
    return A_lookup,phi_lookup,eta_lookup,xi_lookup
def fitboth(b0,f,xm,yup,ydwn):
    gup=b0[0]*np.sin(2*np.pi*f*xm+b0[1])+b0[2]
    gdwn=b0[3]*np.sin(2*np.pi*f*xm+b0[4])+b0[5]
    E=np.sum(np.abs(gup-yup)**2)+np.sum(np.abs(gdwn-ydwn)**2)
    return E
def get_freq(y,t,Tmax,Tmin):
    fs=1/np.mean(np.diff(t)) # calculate sampling frequency
    # look at freqs between 1/10,000 Hz and 1/10 Hz
    freqs=np.linspace(1/Tmax,1/Tmin,10000)
    ls = LombScargle(t,y)
    power = ls.power(freqs)
    peaks,properties = find_peaks(power, width=True)
    pw=power[peaks]
    inds=np.argmax(pw)
    fw=freqs[peaks]
    fw=fw[inds]
    width=properties['widths']/2/fs
    width=width[inds]
    return fw,width

def ls_sin_fit(y, t, fw):
    omega = 2 * np.pi * fw
    cos_part = np.cos(omega * t)
    sin_part = np.sin(omega * t)
    X = np.array([np.ones(np.size(t)), sin_part, cos_part])
    X = X.T
    
    # Core linear least squares solution
    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ X.T @ y
    
    offset = beta[0]
    alpha  = beta[1] # coefficient of sine term (a)
    bta    = beta[2] # coefficient of cosine term (b)
    
    # ----------------------------------------------------
    # NEW: ERROR CALCULATION
    # ----------------------------------------------------
    # 1. Calculate the model predictions and residuals (noise variance)
    y_fit = X @ beta
    residuals = y - y_fit
    degrees_of_freedom = np.size(t) - 3 # N minus 3 fitted parameters
    
    if degrees_of_freedom <= 0:
        raise ValueError("Not enough data points to compute errors.")
    
    # Mean squared error of the residuals
    s_sq = np.sum(residuals**2) / degrees_of_freedom
    
    # 2. Compute the parameter covariance matrix
    cov_matrix = s_sq * XTX_inv
    
    # Extract variances (diagonal elements) and covariances (off-diagonal elements)
    sigma_offset_sq = cov_matrix[0, 0]
    sigma_alpha_sq  = cov_matrix[1, 1]
    sigma_beta_sq   = cov_matrix[2, 2]
    cov_alpha_beta  = cov_matrix[1, 2] # Correlation between alpha and beta
    
    # 3. Calculate standard errors using exact error propagation
    # Error in Offset
    err_offset = np.sqrt(sigma_offset_sq)
    
    # Error in Amplitude: from Query 2 (A = sqrt(a^2 + b^2))
    # Formula updated with cross-term for flawless precision
    amp_error_sq = (alpha**2 * sigma_alpha_sq + bta**2 * sigma_beta_sq + 2 * alpha * bta * cov_alpha_beta) / (alpha**2 + bta**2)
    err_amp = np.sqrt(amp_error_sq)
    
    # Error in Phase: from Query 1 and 3 (phi = atan(b/a))
    # Formula updated with cross-term for flawless precision
    phase_error_sq = (bta**2 * sigma_alpha_sq + alpha**2 * sigma_beta_sq - 2 * alpha * bta * cov_alpha_beta) / (alpha**2 + bta**2)**2
    err_phase = np.sqrt(phase_error_sq)
    # ----------------------------------------------------
    
    # Clean Phase calculation (replaces your 180-degree loop)
    phase = np.arctan2(bta, alpha)
    amp = np.sqrt(alpha**2 + bta**2)
    
    # Bound phase strictly between -pi and pi
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    
    # Returns parameters alongside their mathematically exact standard deviations
    return amp, phase, offset, err_amp, err_phase, err_offset





def fit_sines2(up, dwn, t_raw, Tmax, Tmin, return_errors=False):
    """
    Fit sinusoids to upstream and downstream pressure data.
    The upstream data ('up') is fitted with a pure sine wave.
    The downstream data ('dwn') is fitted with a linear trend + sine wave (shared period).

    Returns
    -------
    updat : list of [amp, period, phase, offset]
    dwndat : list of [amp, period, phase, offset, slope]
    up_err, dwn_err : (if return_errors=True) Standard errors matching the data layouts
    """
    # Shift time to start at 0 for accurate phase tracking
    t = t_raw - t_raw.min()
    N = len(t)
    
    # 1. Use Lomb-Scargle to get a robust initial guess for the frequency
    ls_up = LombScargle(t, up)
    ls_dwn = LombScargle(t, dwn)
    frequency, power = ls_up.autopower(minimum_frequency=1/Tmax, maximum_frequency=1/Tmin)
    best_idx = np.argmax(power)
    guess_freq = frequency[best_idx]
    
    # 2. Extract basic estimates for remaining initial guesses
    theta_up = ls_up.model_parameters(guess_freq)
       
    theta_dwn = ls_dwn.model_parameters(guess_freq)
        
    guess_amp_up = np.sqrt(theta_up[1]**2 + theta_up[2]**2)
    guess_amp_dwn = np.sqrt(theta_dwn[1]**2 + theta_dwn[2]**2)
    guess_offset_up = theta_up[0]+np.mean(up)
    guess_offset_dwn = theta_dwn[0]+np.mean(dwn)
    guess_phase_up = np.arctan2(theta_up[2], theta_up[1])
    guess_phase_dwn = np.arctan2(theta_dwn[2], theta_dwn[1])
    
    # 3. Define the combined model for curve_fit
    # Stacking up and dwn into one array enforces a SHARED frequency parameter
    def combined_model(t_combined, freq, amp_up, phase_up, offset_up, amp_dwn, phase_dwn, offset_dwn, slope_dwn):
        # Split the concatenated time array back into two halves
        t_half = t_combined[:N]
        
        # Upstream: Pure Sine wave + Offset
        model_up = amp_up * np.sin(2 * np.pi * freq * t_half + phase_up) + offset_up
        # Downstream: Sine wave + Linear Trend (slope * t + offset)
        model_dwn = amp_dwn * np.sin(2 * np.pi * freq * t_half + phase_dwn) + (slope_dwn * t_half + offset_dwn)
        
        return np.concatenate([model_up, model_dwn])

    # 4. Prepare data arrays and execute curve_fit
    t_combined = np.concatenate([t, t])
    data_combined = np.concatenate([up, dwn])
    
    # [freq, amp_up, phase_up, offset_up, amp_dwn, phase_dwn, offset_dwn, slope_dwn]
    initial_guesses = [guess_freq, guess_amp_up, guess_phase_up, guess_offset_up, guess_amp_dwn, guess_phase_dwn, guess_offset_dwn, 0.0]
    
    # Set explicit bounds to keep frequency and amplitudes strictly positive
    lower_bounds = [1/Tmax, 0, -np.pi, -np.inf, 0, -np.pi, -np.inf, -np.inf]
    upper_bounds = [1/Tmin, np.inf, np.pi, np.inf, np.inf, np.pi, np.inf, np.inf]
    
    popt, pcov = curve_fit(
        combined_model, t_combined, data_combined, 
        p0=initial_guesses, bounds=(lower_bounds, upper_bounds)
    )
    
    # 5. Extract optimal fitted parameters
    f_fit, amp_up, phase_up, offset_up, amp_dwn, phase_dwn, offset_dwn, slope_dwn = popt
    period_fit = 1 / f_fit
    
    # 6. Extract standard errors from the Covariance Matrix
    perr = np.sqrt(np.diag(pcov))
    err_f, err_amp_up, err_phase_up, err_offset_up, err_amp_dwn, err_phase_dwn, err_offset_dwn, err_slope_dwn = perr
    err_period = err_f / (f_fit ** 2) # Propagation of error for 1/f
    
    # Pack up data arrays matching your exact required layout
    updat  = [amp_up, period_fit, phase_up, offset_up]
    dwndat = [amp_dwn, period_fit, phase_dwn, offset_dwn, slope_dwn]
    
    if not return_errors:
        return updat, dwndat
        
    up_err  = [err_amp_up, err_period, err_phase_up, err_offset_up]
    dwn_err = [err_amp_dwn, err_period, err_phase_dwn, err_offset_dwn, err_slope_dwn]
    
    return updat, dwndat, up_err, dwn_err






def sin_fits_bootstrap(up, dwn, t, Num, Tmin, Tmax):
    """
    Fit sinusoids and estimate parameter uncertainties.

    Primary uncertainty source: Hessian of the objective at the optimum,
    identical to MATLAB's  e = sqrt(diag(inv(hessian)))  from fmincon.
    Bootstrap is still performed for eta/xi error propagation and for
    the distribution plots, but Aerr / phierr are taken from the Hessian.

    If the Hessian is ill-conditioned (singular or negative diagonal),
    the function falls back to bootstrap standard errors automatically.

    Returns
    -------
    upfit, dwnfit        : best-fit parameters [amp, period, phase, offset]
    up_err, dwn_err      : Hessian-based errors (same layout), or bootstrap
                           fallback if Hessian fails
    params_up, params_dwn: bootstrap parameter samples (N x 4)
    """
    # ── 1. Fit original data and compute Hessian errors ──────────────────────
    upfit, dwnfit, up_err_hess, dwn_err_hess = fit_sines2(
        up, dwn, t, Tmax, Tmin, return_errors=True
    )

    hessian_ok = (
        not any(np.isnan(up_err_hess))
        and not any(np.isnan(dwn_err_hess))
    )
    if hessian_ok:
        print("  [Hessian errors: OK — using as primary uncertainty estimate]")
    else:
        print("  [Hessian ill-conditioned — will fall back to bootstrap errors]")

    # ── 2. Bootstrap (needed for eta/xi distribution even if Hessian is OK) ──
    freq     = 1 / upfit[1]
    print(f"Period: {upfit[1]:.3e} s")
    print(f"Frequecy: {freq:.3e} Hz")
    yfit_up  = upfit[0]  * np.sin(2 * np.pi * freq * t + upfit[2])  + upfit[3]
    yfit_dwn = dwnfit[0] * np.sin(2 * np.pi * freq * t + dwnfit[2]) + dwnfit[3] + dwnfit[4] * t
    res_up   = up  - yfit_up
    res_dwn  = dwn - yfit_dwn


    params_up  = np.zeros((Num, 4))
    params_dwn = np.zeros((Num, 5))

    for i in range(Num):
        indx   = np.random.randint(0, len(t), len(t))
        up_bs  = yfit_up  + res_up[indx]
        dwn_bs = yfit_dwn + res_dwn[indx]
        try:
            up_fit, dwn_fit = fit_sines2(up_bs, dwn_bs, t, Tmax, Tmin,
                                        return_errors=False)
            params_up[i, :]  = up_fit
            params_dwn[i, :] = dwn_fit
        except (ValueError, TypeError):
            params_up[i, :]  = np.nan
            params_dwn[i, :] = np.nan

    params_up  = params_up[~np.isnan(params_up).any(axis=1)]
    params_dwn = params_dwn[~np.isnan(params_dwn).any(axis=1)]

    # Bootstrap std (ddof=1, as MATLAB normfit)
    up_err_boot  = np.std(params_up,  axis=0, ddof=1)
    dwn_err_boot = np.std(params_dwn, axis=0, ddof=1)

    # ── 3. Choose error source: Hessian (primary) or Bootstrap (fallback) ────
    #up_err  = up_err_hess  if up_err_hess[1]<up_err_boot[1] else up_err_boot
    #dwn_err = dwn_err_hess if up_err_hess[1]<up_err_boot[1] else dwn_err_boot
    up_err  = up_err_hess  if hessian_ok else up_err_boot
    dwn_err = dwn_err_hess if hessian_ok else dwn_err_boot

    return upfit, dwnfit, up_err, dwn_err, params_up, params_dwn




def plot_bootstrap_distributions(params_up, params_dwn, original_up, original_dwn,plt_num):
    param_names = ['Amplitude', 'Period', 'Phase', 'Offset']
    plt.figure(plt_num).clf()
    fig, axes = plt.subplots(4, 2, figsize=(8, 9),num=plt_num)
    
    #axes = fig.subplots(4, 2)
    params_dwn = np.where(np.isinf(params_dwn), np.nan, params_dwn)
    params_up = np.where(np.isinf(params_up), np.nan, params_up)
    for i in range(4):
        # Upstream
        ax_up = axes[i, 0]
        sns.histplot(params_up[:, i], kde=False, stat='density', color=(0.4, 0.7, 1), ax=ax_up)
        sns.kdeplot(params_up[:, i], color='blue', linewidth=1.5, ax=ax_up)
        ax_up.axvline(original_up[i], color='red', linestyle='--', linewidth=1.5)
        ax_up.set_xlabel(f'Upstream {param_names[i]}')
        ax_up.legend(['KDE','Original Fit', 'Bootstrap'], loc='upper left')

        # Downstream
        ax_dwn = axes[i, 1]
        sns.histplot(params_dwn[:, i], kde=False, stat='density', color=(0.6, 1, 0.6), ax=ax_dwn)
        sns.kdeplot(params_dwn[:, i], color='green', linewidth=1.5, ax=ax_dwn)
        ax_dwn.axvline(original_dwn[i], color='red', linestyle='--', linewidth=1.5)
        ax_dwn.set_xlabel(f'Downstream {param_names[i]}')
        ax_dwn.legend(['KDE','Original Fit', 'Bootstrap'], loc='upper left')

    fig.suptitle('Bootstrap Distributions of Fit Parameters', fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 1], h_pad=2.0)
    #plt.subplots_adjust(top=0.95, bottom=0.05)
    make_figure_topmost(fig)  # Force figure to top
    plt.show()


# ── CHANGE v4 [0]: exact Bernabé forward model ───────────────────────────────
# v3 used a simplified algebraic approximation:
#   A   = sqrt((1 + (2η+ξ)²) / ((1+η²)(1+(η+ξ)²)))
#   φ   = atan((ξ+η)/(1+η(η+ξ))) − atan(η)
# v4 uses the exact complex hyperbolic formulation from Bernabé (2006),
# matching the MATLAB reference (singlek_JMv2_5.m) line for line:
#   val = ((1+i)/sqrt(η·ξ)·sinh((1+i)·sqrt(ξ/η)) + cosh((1+i)·sqrt(ξ/η)))⁻¹
#   A = |val|,  φ = −angle(val)
# The cost function was also updated to use the MATLAB log-ratio form:
#   C = w·(log(A_th)/log(A_exp)−1)² + (1−w)·(φ_th−φ_exp)²
# Starting-value search updated from nearest-neighbour to linear griddata
# interpolation (see solve_bern_eq), also matching MATLAB.
# ── END CHANGE v4 [0] ────────────────────────────────────────────────────────

def _bern_complex(eta, xi):
    """
    Exact Bernabe (2006) forward model using complex hyperbolic functions.
    Matches MATLAB singlek_JMv2_5.m exactly.

    Returns A_i (amplitude ratio) and phi_i (phase shift, positive convention).
    """
    # Guard against xi=0 or eta=0 which cause division by zero in sqrt(eta*xi)
    # When xi→0 the solution degenerates to the xi=0 approximation:
    #   A = 1/sqrt(1 + eta^2/4)   phi = atan(eta/2)  (Bernabe 2006 Eq. A3)
    if xi <= 0 or eta <= 0:
        raise ValueError(f"_bern_complex: xi={xi}, eta={eta} must be > 0")
    if eta * xi < 1e-30:
        # Use limiting form for xi→0
        A_i   = 1.0 / np.sqrt(1.0 + (eta / 2.0) ** 2)
        phi_i = np.arctan(eta / 2.0)
        return A_i, phi_i
    s = (1 + 1j) * np.sqrt(xi / eta)
    val = ((1 + 1j) / np.sqrt(eta * xi) * np.sinh(s) + np.cosh(s)) ** (-1)
    A_i   = np.abs(val)
    phi_i = np.angle(val)
    # Range correction: MATLAB uses phi_i*-1 after ensuring phi_i<=0
    if phi_i > 0:
        phi_i -= 2 * np.pi
    phi_i = -phi_i          # flip sign → phi_i is now positive (0 … pi)
    return A_i, phi_i


def bern_eq(x, *data):
    """
    Cost function for Bernabe equation inversion (exact formulation).

    Parameters
    ----------
    x : [log10(eta), log10(xi)]
    data : (Aexp, phiexp, w)

    Returns
    -------
    float  – weighted residual C (same form as MATLAB bern_eq)
    """
    Aexp, phiexp, w = data
    eta = 10 ** x[0]
    xi  = 10 ** x[1]
    A_i, phi_i = _bern_complex(eta, xi)
    # MATLAB cost: w*(log(A_i)/log(A)-1)^2 + (1-w)*(phi_i-phi)^2
    C = w * (np.log(A_i) / np.log(Aexp) - 1) ** 2 + (1 - w) * (phi_i - phiexp) ** 2
    return C


def bern_fwd(eta, xi):
    """
    Exact Bernabe (2006) forward model — convenience wrapper.
    Returns (A, phi) using the same complex sinh/cosh formula as MATLAB.
    """
    return _bern_complex(eta, xi)



def solve_bern_eq(A, phi, w):
    """
    Solve the Bernabe equation to find eta and xi.

    CHANGE v4 [0]: aligned with MATLAB Solve_Bern_Eq (singlek_JMv2_5.m):
      - griddata linear interpolation for starting values, replacing the
        nearest-neighbour lookup used in v3
      - exact _bern_complex() forward model (see above)
      - phi < phi_xi0 boundary check → xi=0 branch (unchanged from v3)
      - L-BFGS-B optimizer with tighter tolerances (replaces Nelder-Mead)

    Parameters
    ----------
    A, phi : float  – experimental amplitude ratio and phase difference
    w      : float  – weighting (0=A only, 1=phi only, 0.5=equal)

    Returns
    -------
    xi, eta, Afit, phifit, A0, phi0
    """
    from scipy.optimize import fsolve

    # Load lookup table
    A_lookup, phi_lookup, eta_lookup, xi_lookup = lookup_table()

    # MATLAB uses phi values as-is (can be negative); normalise to positive
    # for the griddata call only (phi_lookup may contain negative values)
    pml = np.where(phi_lookup < 0, phi_lookup + 2 * np.pi, phi_lookup)

    # ── Starting values via linear interpolation (matches MATLAB griddata) ──
    eta0 = griddata(
        (np.log10(A_lookup.ravel()), pml.ravel()),
        eta_lookup.ravel(),
        (np.log10(A), phi),
        method='linear'
    )
    xi0 = griddata(
        (np.log10(A_lookup.ravel()), pml.ravel()),
        xi_lookup.ravel(),
        (np.log10(A), phi),
        method='linear'
    )
    # print(f"eta0:  {eta0}")
    # print(f"xi0:  {xi0}")

    # Use approximate formula for eta when xi is negligible (MATLAB xi0<0.1)
    if xi0 is None or np.isnan(xi0) or xi0 < 0.01:
        eta0 = (2 * A) / np.sqrt(1 - A ** 2)

    # Guard against NaN/None from griddata (outside convex hull)
    if eta0 is None or np.isnan(eta0):
        eta0 = (2 * A) / np.sqrt(1 - A ** 2)
    if xi0 is None or np.isnan(xi0):
        xi0 = 0
    
    print(f"eta0:  {eta0}")
    print(f"xi0:  {xi0}")
    # Forward model at interpolated starting point
    # MATLAB: if xi0 < 0.1 use analytical approximation (avoids division by zero)
    if xi0 is None or np.isnan(xi0) or xi0 < 0.01:
        # Analytical solution for xi=0 (Bernabe 2006)
        A0   = eta0 / np.sqrt(eta0**2+4)
        phi0 = np.arctan(np.sqrt(1-A0**2)/A0)
        print(" Used Approximation to where xi ~ 0")
    else:
        A0, phi0 = bern_fwd(float(eta0), float(xi0))
        print(" Used full calculation to get A0 and phi0")

    # ── Boundary check: is phi inside the solution space? ──────────────────
    # phi_xi0 = phase at xi→0 (lower bound of solution space)
    # MATLAB: phi_xi0 = -atan(sqrt(-(A-1)*(A+1))/A) then negated
    phi_xi0 = np.arctan(np.sqrt((1 - A ** 2)) / A)   # positive value, 0…π/2

    if phi < phi_xi0:
        # Data lies to the left of the solution space → xi = 0 branch
        eta    = (2 * A) / np.sqrt(1 - A ** 2)
        xi     = 0
        Afit   = A
        phifit = phi_xi0
        # x_sol: keep log_eta at solution; pin log_xi to boundary value
        x_sol  = np.array([np.log10(eta), np.log10(1e-4)])
    else:
        # ── Solve using Levenberg-Marquardt (matches MATLAB fsolve LM) ──────
        x0 = [np.log10(float(eta0)), np.log10(float(xi0))]

        def cost_vec(x):
            """Return scalar cost as 1-element array so fsolve drives it to 0."""
            return [bern_eq(x, A, phi, w)]

        try:
            x_sol, _, ier, _ = fsolve(
                cost_vec, x0,
                full_output=True,
                xtol=1e-12, ftol=1e-12, maxfev=1000
            )
            if ier not in (1, 2, 3, 4):   # fsolve failed – fall back to minimize
                raise RuntimeError("fsolve did not converge")
        except Exception:
            result = minimize(bern_eq, x0, args=(A, phi, w),
                              method='L-BFGS-B',
                              bounds=[(-2, 6), (-2, 4)],
                              options={'ftol': 1e-12, 'gtol': 1e-12})
            x_sol = result.x

        eta  = 10 ** x_sol[0]
        xi   = 10 ** x_sol[1]
        Afit, phifit = bern_fwd(eta, xi)

        # Mirror MATLAB: if xi is negligible treat as zero
        if xi < 0.1:
            xi = 0

    return xi, eta, Afit, phifit, A0, phi0, x_sol


# ── CHANGE v4 [2]: new function — replaces analytical eta_err formula ────────
# v3 computed eta_err = eta*sqrt((δA/A)² + (A·δA/(1−A²))²) which diverges
# when A → 1 (denominator 1−A² → 0).  This function uses the curvature of
# the Bernabé cost function at the optimum instead, which is well-behaved
# across the full gain range. 
def bern_hessian_errors(x_sol, A, phi, w, eta, xi):
    """
    Hessian-based error estimation — consistent for xi=0 and xi>0.

    xi = 0  (boundary case)
    ─────────────────────
    The solution lives on the xi→0 wall, so xi is not a free parameter.
    We compute the 1-D Hessian of the cost w.r.t. log_eta only, with
    log_xi pinned to the boundary value stored in x_sol[1].
    This avoids the diverging analytical formula  η_err ∝ 1/(1-A²)
    that blows up when A → 1.

    xi > 0  (interior case)
    ────────────────────────
    Full 2-D Hessian w.r.t. [log_eta, log_xi], same as before.

    In both cases:  σ_p = p · ln(10) · σ_log   (log→linear conversion)

    Returns
    -------
    eta_err : float
    xi_err  : float  (nan when xi = 0, parameter not resolved)
    """
    LN10 = np.log(10)

    if xi == 0:
        # ── 1-D Hessian: vary only log_eta, pin log_xi ────────────────────
        log_xi_pin = float(x_sol[1])   # boundary value set in solve_bern_eq
        def cost_1d(log_eta):
            return bern_eq([float(np.asarray(log_eta).flat[0]), log_xi_pin], A, phi, w)
        try:
            H1 = float(nd.Hessian(cost_1d)(np.array([x_sol[0]]))[0, 0])
            if H1 <= 0:
                raise ValueError("Non-positive 1-D Hessian")
            eta_err = float(eta * LN10 * np.sqrt(1.0 / H1))
        except Exception:
            eta_err = np.nan
        xi_err = np.nan   # xi is not a free parameter on the boundary

    else:
        # ── 2-D Hessian: both log_eta and log_xi free ─────────────────────
        try:
            H2   = nd.Hessian(lambda x: bern_eq(x, A, phi, w))(x_sol)
            cond = np.linalg.cond(H2)
            if cond > 1e12:
                raise np.linalg.LinAlgError(f"ill-conditioned (cond={cond:.2e})")
            H_inv = np.linalg.inv(H2)
            diag  = np.diag(H_inv)
            e_log = np.where(diag > 0, np.sqrt(diag), np.nan)
            eta_err = float(eta * LN10 * e_log[0])
            xi_err  = float(xi  * LN10 * e_log[1])
        except Exception:
            eta_err = np.nan
            xi_err  = np.nan

    return float(eta_err), float(xi_err) if not np.isnan(xi_err) else np.nan


def plot_nomo(plt_num):
    """
    Nomogram for Bernabe (2006) equation — replica of MATLAB figure.

    Red lines   = iso-eta contours (log10 eta: -1.6 to +0.2, step 0.2)
    Green lines = iso-xi contours  (xi: 0.002 ... 16)

    Curves computed directly from the Bernabe complex equation —
    independent of lookup table structure.
    """
    fig = plt.figure(plt_num)
    plt.clf()
    ax = fig.add_subplot(111)

    xi_scan  = np.logspace(-5, 1.5, 3000)
    eta_scan = np.logspace(2, -2, 3000)

    # ── RED: iso-eta (fix eta, vary xi small→large) ───────────────────────────
    for log_eta in np.arange(-1.6, 0.21, 0.2):
        eta = 10 ** log_eta
        phi_c, A_c, prev_phi = [], [], None
        for xi in xi_scan:
            try:
                A_i, phi_i = _bern_complex(eta, xi)
                if not (0.001 < A_i < 0.9995 and 0 < phi_i < np.pi):
                    continue
                if prev_phi is not None and abs(phi_i - prev_phi) > 0.15:
                    continue
                phi_c.append(phi_i); A_c.append(A_i); prev_phi = phi_i
            except:
                continue
        if len(phi_c) > 2:
            idx = np.argsort(phi_c)
            ph = np.array(phi_c)[idx]
            ac = np.log10(np.array(A_c)[idx])
            ax.plot(ph, ac, color='red', lw=0.9)
            ax.text(ph[0], ac[0], f'{log_eta:.1f}',
                    color='red', fontsize=7, va='bottom', ha='right', clip_on=True)

    # ── GREEN: iso-xi (fix xi, vary eta large→small) ──────────────────────────
    xi_list = [0.002, 0.004, 0.008, 0.016, 0.032,
               0.064, 0.128, 0.256, 0.512, 1, 2, 4, 8, 16]
    for xi in xi_list:
        phi_c, A_c, prev_phi = [], [], None
        for eta in eta_scan:
            try:
                A_i, phi_i = _bern_complex(eta, xi)
                if not (0.001 < A_i < 0.9995 and 0 < phi_i < np.pi):
                    continue
                if prev_phi is not None and abs(phi_i - prev_phi) > 0.15:
                    continue
                phi_c.append(phi_i); A_c.append(A_i); prev_phi = phi_i
            except:
                continue
        if len(phi_c) > 2:
            idx = np.argsort(phi_c)
            ph = np.array(phi_c)[idx]
            ac = np.log10(np.array(A_c)[idx])
            ax.plot(ph, ac, color='green', lw=0.9)
            if ph[-1] > 1.5:
                ax.text(ph[-1], ac[-1], f'  {xi}',
                        color='green', fontsize=7, va='center', ha='left', clip_on=True)
            else:
                ax.text(ph[len(ph)//2], -2.02, f'{xi}',
                        color='green', fontsize=7, va='top', ha='center', clip_on=False)

    ax.set_xlabel('Phase Shift (radians)', fontsize=11)
    ax.set_ylabel('Log (Gain)', fontsize=11)
    ax.set_title('Nomogram — Bernabé (2006) Red = iso-η   |   Green = iso-ξ', fontsize=11)
    ax.set_xlim([0, 3.9])
    ax.set_ylim([-2.05, 0.05])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Phase Shift rad', fontsize=11)

    # --- Legend for data points (proxy artists so it works across loops) ---
    import matplotlib.lines as mlines
    leg_data  = mlines.Line2D([], [], color='blue',  marker='o', linestyle='None',
                              markersize=6, label='Measured (φ, log A) ± errors')
    leg_start = mlines.Line2D([], [], color='green', marker='^', linestyle='None',
                              markersize=7, label='Starting estimate (φ₀, log A₀)')
    leg_fit   = mlines.Line2D([], [], color='red',   marker='x', linestyle='None',
                              markersize=8, markeredgewidth=2,
                              label='Bernabé fit (φ_fit, log A_fit)')
    ax.legend(handles=[leg_data, leg_start, leg_fit],
              loc='lower right', fontsize=8, framealpha=0.8)

    make_figure_topmost(fig)
    plt.tight_layout()
    plt.show()
    return ax   # return main axis so caller can plot points on it


# Viscosity and compressibility calculations
def argon(Temp,P):
    # T in K and P in Pa
    # outputs Compressibility (1/Pa) and viscosity Pa.s
    _, _, _, C0=Argon_Z(Temp,P)
    mu, _=argon_visc(Temp,P*1e6)
    mu=mu/1e6
    return C0, mu


def water(Temp,Press):
    # T in K and P in MPa
    #outputs density in kg/m^3 C (1/MPa) and viscosity Pa.s
    prop=IAPWS95(T=Temp,P=Press)
    mu=_iapws._Viscosity(rho=prop.rho, T=Temp)
    C0=(1/prop.rho)*(1/prop.dpdrho_T)
    C0=C0/1e6
    return C0,mu

def rheolube(Temp,P):
    # Temp in K and P in MPa
    K00=6.292 # GPa
    betak=0.0052
    K0p=12.051
    K0=K00*np.exp(-betak*Temp)
    C0=1/(K0*((P/1000*(K0p + 1))/K0 + 1))
    C0=C0/1e9
    A1=134.9376
    A2=0.3128
    Tg0=-93.4602+273
    B1=7.1564
    B2=-0.4888
    C1=16.0511
    C2=19.6526
    mug=1e12
    Tg=Tg0+A1*np.log(1+A2*P/1000)
    F=(1+B1*P/1000)**B2
    mu=mug*np.exp(-np.log(10)*(C1*(Temp-Tg)*F)/(C2+(Temp-Tg)*F))
    return C0,mu


def ask_to_continue():
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Force to top
    root.lift()
    root.focus_force()
    root.update()
    result = messagebox.askyesno("Continue?", "Process another dataset?")
    root.destroy()
    return result



# Pore pressure oscillation permeability processing

# ---------------------------------------------------------------------------
# Helper: prompt user for a value, showing a default in brackets.
# Pressing Enter without typing anything accepts the default.
# ---------------------------------------------------------------------------
def prompt(message, default, cast=float):
    """
    Prompt the user for a value with a default fallback.

    Parameters
    ----------
    message : str
        Text shown to the user (without the default hint).
    default : any
        Value used when the user just presses Enter.
    cast : callable
        Type-casting function applied to the raw string input
        (e.g. float, int, str).

    Returns
    -------
    Value of type `cast`, or `default` if nothing was entered.
    """
    raw = input(f"  {message} [{default}]: ").strip()
    if raw == "":
        return default
    return cast(raw)


print("\n" + "=" * 60)
print("SAMPLE PARAMETERS")
print("=" * 60)
print("(Press Enter to accept the default value shown in brackets)\n")

# -- Sample geometry --
l        = prompt("Sample length (mm)",                  100)
l_err    = prompt("Error on length (m)",                 5e-4)
dia      = prompt("Sample diameter (mm)",                20)
dia_err  = prompt("Error on diameter (mm)",              0.5)

# -- Downstream storage mode --
# ── CHANGE v4 [3]: downstream storage — bd direct or Dv × C(T,P) ────────────
# v3 always computed bd = Dv × C(T,P) internally.  v4 lets the user choose:
#   bd  — enter bd directly (water / incompressible fluids)
#   Dv  — enter downstream volume; bd recomputed per measurement from the
#          actual pore pressure (argon / compressible fluids where C(P) varies)
print()
print("  Downstream storage capacity mode:")
print("    bd  — enter bd directly (recommended for incompressible fluids, e.g. water)")
print("    Dv  — enter downstream volume; bd = Dv × C(T,P) computed per measurement")
print("          (recommended for compressible fluids, e.g. argon)")
print()
while True:
    bd_mode = input("  Choose mode [bd / Dv]: ").strip().lower()
    if bd_mode in ('bd', 'dv'):
        break
    print("   Please type  bd  or  Dv")

if bd_mode == 'bd':
    bd      = prompt("Downstream storage capacity bd (m³/Pa)", 2.2522378352e-15)
    bd_err  = prompt("Error on bd (m³/Pa)",                    5e-17)
    Dv      = None   # not used
    Dv_err  = None
else:
    Dv      = prompt("Downstream volume Dv (m³)",   9.6085e-6)
    Dv_err  = prompt("Error on Dv (m³)",            0.01e-6)
    bd      = None   # will be computed per measurement
    bd_err  = None
# ── END CHANGE v4 [3] ────────────────────────────────────────────────────────

# -- Experiment conditions --
Temp     = prompt("Temperature (K)",                     423.15)

valid_permeants = ('water', 'argon', 'rheolube')
while True:
    permeant = prompt("Permeant fluid [water / argon / rheolube]", "water", cast=str)
    if permeant in valid_permeants:
        break
    print(f"  ⚠  Invalid choice '{permeant}'. Please enter one of: {valid_permeants}")

print("\n" + "=" * 60)
print("DATA FILE COLUMN INDICES  (0-based)")
print("=" * 60)
HeaderRows = prompt("Number of header rows to skip",     3,  cast=int)
time_col   = prompt("Time column index",                 0,  cast=int)
Pup_col    = prompt("Upstream pressure column index",    1,  cast=int)
Pdwn_col   = prompt("Downstream pressure column index",  2,  cast=int)
Pc_col     = prompt("Confining pressure column index",   3,  cast=int)

print("\n" + "=" * 60)
print("FITTING PARAMETERS")
print("=" * 60)
N    = prompt("Number of bootstrap resamples",           20,     cast=int)
w    = prompt("A/phi weighting factor (0=A only, 1=phi only, 0.5=equal)", 0.5)
Tmin = prompt("Minimum oscillation period to search (s)", 1,   cast=float)
Tmax = prompt("Maximum oscillation period to search (s)", 10000, cast=int)

# -- Unit conversions (done after user input) --
l    = l / 1000                            # mm → m
area = np.pi * (dia / 2000) ** 2          # mm diameter → m² area

print("\n" + "=" * 60)
print("Input summary")
print("=" * 60)
print(f"  Sample:      l = {l*1000:.3f} mm  |  dia = {dia:.3f} mm")
if bd_mode == 'bd':
    print(f"  Storage:     bd = {bd:.3e} m³/Pa  ±  {bd_err:.3e}  [direct input]")
else:
    print(f"  Storage:     Dv = {Dv:.3e} m³  ±  {Dv_err:.3e}  →  bd = Dv×C(T,P) per measurement")
print(f"  Conditions:  T = {Temp} K  |  permeant = {permeant}")
print(f"  Columns:     time={time_col}  Pup={Pup_col}  Pdwn={Pdwn_col}  Pc={Pc_col}  header_rows={HeaderRows}")
print(f"  Fitting:     N={N}  w={w}  Tmin={Tmin} s  Tmax={Tmax} s")
print("=" * 60)


# Ensure no matplotlib figures are open before spawning the tkinter dialog.
# On re-runs inside the same session (Spyder / IPython) stale figure windows
# hold a tkinter event loop that makes filedialog hang indefinitely.
plt.close('all')
try:
    plt.pause(0.3)
except Exception:
    pass

# Prompt user to select output file location once
print("\n" + "=" * 60)
print("STEP 1: Please select where to SAVE the output CSV file")
print("=" * 60)
root = create_topmost_root()
outfile = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xls"), ("all files", "*.*")],
    title="Output File",
    initialfile="datafile_proc.csv"
)
root.destroy()

if not outfile:
    print("Save operation cancelled.")
else:
    print(f"\nOutput will be saved to: {outfile}")
    first_loop = True
    while True:
        # Prompt for input data file
        print("\n" + "=" * 60)
        print("STEP 2: Please select an INPUT DATA file to process")
        print("=" * 60)
        root = create_topmost_root()
        datafile = filedialog.askopenfilename(title="Select Data File")
        root.destroy()
        if not datafile:
            print("No file selected. Exiting loop.")
            break

        print(f"\nLoading data from: {os.path.basename(datafile)}")
        # Load and extract data
        all_file = np.loadtxt(datafile, delimiter='\t', skiprows=HeaderRows)
        # May need to add calibrations here
        time = all_file[:, time_col]
        pup = all_file[:, Pup_col]
        pdwn = all_file[:, Pdwn_col]
        pc=np.mean(all_file[:, Pc_col])
        # NOTE: lowpass filter removed — with fs=1Hz and oscillation periods
        # of ~20-60s, any filter aggressive enough to remove aircon noise
        # also kills the PPO signal. No filtering applied to downstream.

        print("\nSTEP 3: Interactive plot will appear")
        print("        Click 2 points to select COARSE region of interest")
        # ── CHANGE v4 [4]: clear figure before recreating axes ───────────────
        # v3 reused the same axes on repeated runs, causing axis labels from
        # the previous run to accumulate (overlap) on the new ones.
        # Calling .clf() before plt.subplots() ensures a clean slate each time.
        # Stage 1: coarse ROI selection
        _fig1 = plt.figure(1)
        _fig1.clf()
        fig, ax = plt.subplots(num=1)
        ax.plot(time, pup, 'r', label='Upstream Pressure')
        ax.plot(time, pdwn, 'b', label='Downstream Pressure')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (MPa)')
        ax.set_title('Click start and end points for coarse ROI')
        ax.legend()
        ax.grid()
        make_figure_topmost(fig)  # Force figure to top
        plt.show()

        # Get coarse ROI
        pts = plt.ginput(2, timeout=-1)
        tree = KDTree(np.column_stack((time, pup)))
        idx = [tree.query(pt)[1] for pt in pts]

        # Stage 2: refine ROI — open a completely new figure window
        plt.close(fig)   # close figure 1 entirely
        
        mask_roi = (time >= time[idx[0]]) & (time <= time[idx[1]])
        # Clear figure 10 before recreating axes (prevents label overlap on re-runs)
        _fig10 = plt.figure(10)
        _fig10.clf()
        fig2, ax2 = plt.subplots(num=10, figsize=(12, 5))
        ax2.plot(time[mask_roi], pup[mask_roi],  'r', lw=0.9, label='Upstream Pressure')
        ax2.plot(time[mask_roi], pdwn[mask_roi], 'b', lw=0.9, label='Downstream Pressure')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pressure (MPa)')
        ax2.set_title('Click START and END points for REFINED ROI')
        ax2.legend()
        ax2.grid()
        make_figure_topmost(fig2)
        plt.tight_layout()
        plt.show()

        # Get refined ROI
        pts = fig2.ginput(2, timeout=-1)
        tree = KDTree(np.column_stack((time, pup)))
        idx = [tree.query(pt)[1] for pt in pts]

        

        # Plot selected range
        plt.figure(2)
        plt.clf()
        plt.plot(time, pup, 'r', label='Upstream Pressure')
        plt.plot(time, pdwn, 'b', label='Downstream Pressure')
        plt.axvline(time[idx[0]], color='g', linestyle='--', label='Start Point')
        plt.axvline(time[idx[1]], color='m', linestyle='--', label='End Point')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (MPa)')
        plt.title('Selected Data Range')
        plt.legend()
        plt.show()

        # Extract selected data
        time = time[idx[0]:idx[1] + 1]
        time=time-np.min(time)
        pup = pup[idx[0]:idx[1] + 1]
        pdwn = pdwn[idx[0]:idx[1] + 1]

        # Fit and model
        updata, dwndata, up_err, dwn_err, up_params_bs, dwn_params_bs = sin_fits_bootstrap(pup, pdwn, time, N,Tmin,Tmax)
        plot_bootstrap_distributions(up_params_bs, dwn_params_bs, updata, dwndata, 3)
        A = np.abs(dwndata[0] / updata[0])
        Aerr = A * np.sqrt((up_err[0] / (2 * updata[0]))**2 + (dwn_err[0] / (2 * dwndata[0]))**2)
        logAerr = np.abs(Aerr / A / np.log(10))
        phi = updata[2]-dwndata[2]
        phierr = np.sqrt((up_err[2] / 2)**2 + (dwn_err[2] / 2)**2)
        T = updata[1]
        if phi > np.pi:
            phi -= 2 * np.pi
        if phi < -np.pi:
            phi += 2*np.pi
        
            

        upmodel = updata[3] + updata[0] * np.sin(time * 2 * np.pi / updata[1] + updata[2])
        dwnmodel = dwndata[3] + updata[0]*A * np.sin(time * 2 * np.pi / updata[1] + updata[2]-phi)+dwndata[4]*time

        plt.figure(4)
        plt.clf()
        plt.plot(time, pup, 'r', label='Upstream Pressure')
        plt.plot(time, pdwn, 'b', label='Downstream Pressure')
        plt.plot(time, upmodel, 'g', label='Up Model')
        plt.plot(time, dwnmodel, 'm', label='Down Model')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (MPa)')
        plt.title('Model Fit')
        plt.legend()
        plt.show()

        # Permeant properties + downstream storage capacity
        if permeant == 'water':
            C, visc = water(Temp, updata[3])
        elif permeant == 'argon':
            C, visc = argon(Temp, updata[3])
        elif permeant == 'rheolube':
            C, visc = rheolube(Temp, updata[3])
        else:
            C, visc = argon(Temp, updata[3])

        # ── CHANGE v4 [3]: compute bd per measurement when in Dv mode ─────────
        if bd_mode == 'bd':
            # bd was entered directly — use as-is (fixed for all measurements)
            pass   # bd and bd_err already set
        else:
            # Dv mode: bd = Dv × C(T, P)  — computed from fluid compressibility
            bd      = float(Dv * C)
            bd_err  = float(C * Dv_err)   # dominant term; C uncertainty neglected
        
        
        
    

        xi, eta, Afit, phifit, A0, phi0, x_sol = solve_bern_eq(A, phi, w)
        if first_loop == True:
            nomo_ax = plot_nomo(5)   # returns main axis, store for all loops
        # Always plot on the stored main axis (not twin ax2)
        nomo_ax.errorbar(abs(phi), np.log10(A), xerr=phierr, yerr=logAerr,
                         fmt='o', color='blue', zorder=5,
                         label='_nolegend_')
        nomo_ax.plot(phi0, np.log10(A0), '^', color='g', zorder=5,
                     label='_nolegend_')
        nomo_ax.plot(phifit, np.log10(Afit), 'x', color='r',
                     markersize=8, markeredgewidth=2, zorder=5,
                     label='_nolegend_')
        # Refresh figure layout cleanly
        plt.figure(5).tight_layout()
        plt.figure(5).canvas.draw_idle()
        plt.pause(0.05)

        # ── CHANGE v4 [2]: unified Hessian errors, replaces old split logic ───
        # Primary: Hessian-based errors (curvature of cost function).
        # Fallback: bootstrap std when Hessian gives relative error > 100%
        # (flat cost surface — measurement poorly constrained). Last resort:
        # analytical formula for xi=0 boundary.
        eta_err, xi_err = bern_hessian_errors(x_sol, A, phi, w, eta, xi)

        # Fallback to bootstrap if Hessian error is implausibly large (>100%)
        if np.isnan(eta_err) or eta_err / eta > 1.0:
            if xi == 0:
                _ae = eta * np.sqrt((Aerr / A)**2 + (A * Aerr / (1 - A**2))**2)
                eta_err = float(_ae) if np.isfinite(_ae) else eta_err
                xi_err  = np.nan
            else:
                Adist   = dwn_params_bs[:, 0] / up_params_bs[:, 0]
                phidist = up_params_bs[:, 2] - dwn_params_bs[:, 2]
                phidist[phidist < 0] += 2 * np.pi
                eta_dist = np.zeros_like(Adist)
                xi_dist  = np.zeros_like(Adist)
                for _p, (_Ai, _phi_i) in enumerate(zip(Adist, phidist)):
                    xi_dist[_p], eta_dist[_p], *_ = solve_bern_eq(_Ai, _phi_i, w)
                _ind = np.where((xi_dist < 16) & (Adist > 0))[0]
                if len(_ind) > 1:
                    _ae = np.std(eta_dist[_ind], ddof=1)
                    _xe = np.std(xi_dist[_ind],  ddof=1)
                    if np.isfinite(_ae): eta_err = _ae
                    if np.isfinite(_xe): xi_err  = _xe
            print("  [η/ξ errors: Hessian >100% — using bootstrap as fallback]")
        else:
            print("  [Hessian errors: OK — using as primary uncertainty estimate]")
        # ── END CHANGE v4 [2] ────────────────────────────────────────────────

        k    = float((eta * np.pi * l * visc * bd) / (area * T))
        kerr = float(np.abs(k) * np.sqrt(
            (eta_err / eta)      ** 2 +
            (l_err   / l)        ** 2 +
            (bd_err  / bd)       ** 2 +
            (2 * dia_err / dia)  ** 2 +
            (up_err[1] / T)      ** 2
        ))

        if xi == 0:
            bc     = 0.0
            bc_err = np.nan
        else:
            bc     = float((xi * bd) / (area * l))
            bc_err = float(np.abs(bc) * np.sqrt(
                (xi_err  / xi)       ** 2 +
                (bd_err  / bd)       ** 2 +
                (2 * dia_err / dia)  ** 2 +
                (l_err   / l)        ** 2
            ))

        # Create and append DataFrame
        file=os.path.basename(datafile)
        output = pd.DataFrame([{
            'File': file,
            'start index': idx[0],
            'end index': idx[1],
            'ConfP': pc,
            'PoreP': updata[3],
            'UpAmp': updata[0],
            'Gain': A,
            'delA': Aerr,
            'Phase': phi,
            'delphi': phierr,
            'Period': T,
            'delT': up_err[1],
            'eta': eta,
            'deleta': eta_err,
            'xi': xi,
            'delxi': xi_err,
            'Permeability': k,
            'delk': kerr,
            'Storage Capacity': bc,
            'delbeta': bc_err 
        }])
        file_name = os.path.basename(datafile)
        output.to_csv(outfile, mode='w' if first_loop else 'a', header=first_loop, index=False)
        print(f"\n{'=' * 60}")
        print(f"Results {'saved' if first_loop else 'appended'} to {os.path.basename(outfile)}")
        print(f"Permeability: {k:.3e} ± {kerr:.3e} m²")
        print(f"Storage Capacity: {bc:.3e} ± {bc_err:.3e} Pa⁻¹")
        print(f"{'=' * 60}")
        first_loop = False
        if not ask_to_continue():
            break
        #cont = input("Process another dataset? (y/n): ").strip().lower()
        #if cont != 'y':
            #break