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
v6 
  [1] simplified the fitting of pressure waves
  
  [2] Added a linear function to the fit of downstream to fit to account for leaks temperature drift
  
v7 

    [1] new branch to do continuous processing so you can have time dependent perm
    [2] tidied up some of the procedures

  
"""
import os

# WSL/default workflow: use the same Tk-based GUI path as the stable main branch.
os.environ.setdefault('MPLBACKEND', 'TkAgg')
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

from tkinter import Tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib
# ROI selection uses ginput(), which requires an interactive GUI backend.
# Select Tk before importing pyplot so VS Code does not leave us on an inline backend.
try:
    matplotlib.use('TkAgg')
except (ImportError, RuntimeError) as exc:
    raise RuntimeError(
        "The TkAgg plotting backend could not be enabled. Run this script in "
        "VS Code's terminal (not the Jupyter/Interactive window) and ensure "
        "Tkinter is installed."
    ) from exc
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.timeseries import LombScargle
import scipy.io
from scipy.spatial import KDTree
from scipy.signal import find_peaks #, butter, filtfilt
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
# from scipy.stats import norm  # kept for possible future use – not used in main calculation
from iapws import IAPWS95
from iapws import _iapws
from argon import Argon_Z
from argon import argon_visc
import configparser
from pathlib import Path
import numdifftools as nd
#import time as time_py
from tqdm import tqdm

# Force-close any figures left over from a previous run in the same session,
# then give the event loop a moment to fully release tkinter resources before
# we open new dialogs.  Without this pause the filedialog can hang on re-runs.


plt.close('all')
try:
    plt.pause(0.2)          # let the old event loop drain
except Exception:
    pass

print("=" * 60)
print("PPO Permeability Processing - Modified Version")
print("Using config-file workflow with dat/mat input and bd/Dv storage")
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
    root_folder = Path(__file__).resolve().parent
    mat_data = scipy.io.loadmat(root_folder / 'lookup.mat')
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
    if len(t_raw) < 8:
        raise ValueError("Not enough samples to fit sine waves.")
    if not (np.all(np.isfinite(up)) and np.all(np.isfinite(dwn)) and np.all(np.isfinite(t_raw))):
        raise ValueError("Input window contains non-finite values.")

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
        p0=initial_guesses, bounds=(lower_bounds, upper_bounds),
        max_nfev=20000
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
    #if hessian_ok:
        
        #print("  [Hessian errors: OK — using as primary uncertainty estimate]")
    #else:
        #print("  [Hessian ill-conditioned — will fall back to bootstrap errors]")
        
    # ── 2. Bootstrap (needed for eta/xi distribution even if Hessian is OK) ──
    freq     = 1 / upfit[1]
    #print(f"Period: {upfit[1]:.3e} s")
    #print(f"Frequecy: {freq:.3e} Hz")
    yfit_up  = upfit[0]  * np.sin(2 * np.pi * freq * t + upfit[2])  + upfit[3]
    yfit_dwn = dwnfit[0] * np.sin(2 * np.pi * freq * t + dwnfit[2]) + dwnfit[3] + dwnfit[4] * t
    res_up   = up  - yfit_up
    res_dwn  = dwn - yfit_dwn


    params_up  = np.full((Num, 4), np.nan)
    params_dwn = np.full((Num, 5), np.nan)

    for i in range(Num):
        indx   = np.random.randint(0, len(t), len(t))
        up_bs  = yfit_up  + res_up[indx]
        dwn_bs = yfit_dwn + res_dwn[indx]
        try:
            up_fit, dwn_fit = fit_sines2(up_bs, dwn_bs, t, Tmax, Tmin,
                                        return_errors=False)
            params_up[i, :]  = up_fit
            params_dwn[i, :] = dwn_fit
        except (RuntimeError, ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            params_up[i, :]  = np.nan
            params_dwn[i, :] = np.nan

    valid_bs = ~np.isnan(params_up).any(axis=1) & ~np.isnan(params_dwn).any(axis=1)
    params_up  = params_up[valid_bs]
    params_dwn = params_dwn[valid_bs]

    # Bootstrap std (ddof=1, as MATLAB normfit)
    if len(params_up) > 1:
        up_err_boot  = np.std(params_up,  axis=0, ddof=1)
        dwn_err_boot = np.std(params_dwn, axis=0, ddof=1)
    else:
        up_err_boot  = np.full(4, np.nan)
        dwn_err_boot = np.full(5, np.nan)

    # ── 3. Choose error source: Hessian (primary) or Bootstrap (fallback) ────
    #up_err  = up_err_hess  if up_err_hess[1]<up_err_boot[1] else up_err_boot
    #dwn_err = dwn_err_hess if up_err_hess[1]<up_err_boot[1] else dwn_err_boot
    up_err  = up_err_hess  if hessian_ok else up_err_boot
    dwn_err = dwn_err_hess if hessian_ok else dwn_err_boot

    return upfit, dwnfit, up_err, dwn_err, params_up, params_dwn




def plot_bootstrap_distributions(params_up, params_dwn, original_up, original_dwn,plt_num):
    param_names = ['Amplitude', 'Period', 'Phase', 'Offset']
    plt.close(plt_num)
    fig, axes = plt.subplots(4, 2, figsize=(8, 9), num=plt_num, clear=True)
    
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
    tile_figure(plt_num, row=0, col=2, row_span=2)
    make_figure_topmost(fig)  # Force figure to top
    plt.show(block=False)
    plt.pause(0.1)


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
    if not np.isfinite(A) or A <= 0 or A >= 1:
        raise ValueError(f"Gain A={A:.6g} is outside the Bernabé domain 0 < A < 1.")
    if not np.isfinite(phi):
        raise ValueError("Phase shift is not finite.")

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
    
    #print(f"eta0:  {eta0}")
    #print(f"xi0:  {xi0}")
    # Forward model at interpolated starting point
    # MATLAB: if xi0 < 0.1 use analytical approximation (avoids division by zero)
    if xi0 is None or np.isnan(xi0) or xi0 < 0.01:
        # Analytical solution for xi=0 (Bernabe 2006)
        A0   = eta0 / np.sqrt(eta0**2+4)
        phi0 = np.arctan(np.sqrt(1-A0**2)/A0)
        # print(" Used Approximation to where xi ~ 0")
    else:
        A0, phi0 = bern_fwd(float(eta0), float(xi0))
        # print(" Used full calculation to get A0 and phi0")

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

def bern_errors(x_sol, A,Aerr, phi, w, eta, xi,up_params_bs,dwn_params_bs):
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
    
    Calculates errors from bootstrapping if Hessian errors not acceptable

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

    # ── Fallback to bootstrap if Hessian error is implausibly large (>100%) ──
    # MOVED OUTSIDE ELSE BLOCK: This allows xi == 0 case to use it
    if np.isnan(eta_err) or eta_err / eta > 1.0:
        if xi == 0:
            _ae = eta * np.sqrt((Aerr / A)**2 + (A * Aerr / (1 - A**2))**2)
            eta_err = float(_ae) if np.isfinite(_ae) else eta_err
            xi_err  = np.nan
        else:
            Adist   = dwn_params_bs[:, 0] / up_params_bs[:, 0]
            phidist = up_params_bs[:, 2] - dwn_params_bs[:, 2]
            phidist[phidist < 0] += 2 * np.pi
            eta_dist = np.full_like(Adist, np.nan)
            xi_dist  = np.full_like(Adist, np.nan)
            for _p, (_Ai, _phi_i) in enumerate(zip(Adist, phidist)):
                if not (np.isfinite(_Ai) and 0 < _Ai < 1 and np.isfinite(_phi_i)):
                    continue
                try:
                    xi_dist[_p], eta_dist[_p], *_ = solve_bern_eq(_Ai, _phi_i, w)
                except (RuntimeError, ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
                    continue
            _ind = np.where((xi_dist < 16) & (Adist > 0) & (Adist < 1) & np.isfinite(eta_dist) & np.isfinite(xi_dist))[0]
            if len(_ind) > 1:
                _ae = np.std(eta_dist[_ind], ddof=1)
                _xe = np.std(xi_dist[_ind],  ddof=1)
                if np.isfinite(_ae): eta_err = _ae
                if np.isfinite(_xe): xi_err  = _xe

    return float(eta_err), float(xi_err) if not np.isnan(xi_err) else np.nan

    

def plot_nomo(plt_num, no_leg='no'):
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
    if no_leg=='yes':
        pass
    else:
        ax.legend(handles=[leg_data, leg_start, leg_fit],
                  loc='lower right', fontsize=8, framealpha=0.8)

    tile_figure(plt_num, row=1, col=1)
    make_figure_topmost(fig)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
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

def permeant_props(permeant,Temp,Press,bd_mode,bd,bd_err,Dv,Dv_err):
    # Usage:
    #  C,visc,bd,bd_err=permeant_props(permeant,Temp,Press,bd_mode,bd,bd_err,Dv,Dv_err)    
    if permeant == 'water':
        C, visc = water(Temp, Press)
    elif permeant == 'argon':
        C, visc = argon(Temp, Press)
    elif permeant == 'rheolube':
        C, visc = rheolube(Temp, Press)
    else:
        C, visc = argon(Temp, Press)
    
        # ── CHANGE v4 [3]: compute bd per measurement when in Dv mode ─────────
    if bd_mode == 'bd':
        # bd was entered directly — use as-is (fixed for all measurements)
        pass   # bd and bd_err already set
    else:
        # Dv mode: bd = Dv × C(T, P)  — computed from fluid compressibility
        bd      = float(Dv * C)
        bd_err  = float(C * Dv_err)   # dominant term; C uncertainty neglected
    return C,visc,bd,bd_err

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


def tile_figure(fig_num, row, col, tot_rows=2, tot_cols=3, row_span=1, col_span=1):
    """
    Position a Matplotlib figure in a screen tile when the active GUI backend
    exposes a window geometry API. This preserves the v3 multi-window layout
    without making the script backend-specific.
    """
    try:
        fig = plt.figure(fig_num)
        manager = fig.canvas.manager
        window = getattr(manager, 'window', None)
        if window is None:
            return

        # Tk backends
        if hasattr(window, 'winfo_screenwidth') and hasattr(window, 'geometry'):
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            tile_w = int(screen_w / tot_cols)
            tile_h = int((screen_h - 60) / tot_rows)
            x_pos = col * tile_w
            y_pos = row * tile_h
            window_w = tile_w * col_span
            window_h = tile_h * row_span
            window.geometry(f"{window_w}x{window_h}+{x_pos}+{y_pos}")
            return

        # Qt backends
        if hasattr(window, 'screen') and hasattr(window, 'setGeometry'):
            screen = window.screen().availableGeometry()
            tile_w = int(screen.width() / tot_cols)
            tile_h = int((screen.height() - 60) / tot_rows)
            x_pos = int(screen.x() + col * tile_w)
            y_pos = int(screen.y() + row * tile_h)
            window_w = tile_w * col_span
            window_h = tile_h * row_span
            window.setGeometry(x_pos, y_pos, window_w, window_h)
    except Exception:
        pass


def _config_has(config, sections, option):
    """Return True if option exists in any of the supplied config sections."""
    if isinstance(sections, str):
        sections = (sections,)
    return any(config.has_option(section, option) for section in sections)


def _config_get(config, sections, option, cast=str, default=None, required=False):
    """Read one value from the first section containing option."""
    if isinstance(sections, str):
        sections = (sections,)

    for section in sections:
        if not config.has_option(section, option):
            continue
        if cast is float:
            return config.getfloat(section, option)
        if cast is int:
            return config.getint(section, option)
        if cast is bool:
            return config.getboolean(section, option)
        value = config.get(section, option)
        return value.strip() if isinstance(value, str) else value

    if required:
        section_list = ", ".join(sections)
        raise KeyError(f"Missing required config value: {section_list}.{option}")
    return default


def prompt_processing_parameters():
    """Interactive fallback used when no matching .ini file is available."""
    print("\n" + "=" * 60)
    print("SAMPLE PARAMETERS")
    print("=" * 60)
    print("(Press Enter to accept the default value shown in brackets)\n")

    params = {}
    params['l']       = prompt("Sample length (mm)",                  100)
    params['l_err']   = prompt("Error on length (m)",                 5e-4)
    params['dia']     = prompt("Sample diameter (mm)",                20)
    params['dia_err'] = prompt("Error on diameter (mm)",              0.5)
    params['thickness_mode'] = 'fixed'
    params['thickness_col'] = None
    params['thickness_var'] = None
    params['thickness_scale'] = 1.0
    params['thickness_min_mm'] = 0.0
    params['thickness_max_mm'] = 5.0

    print()
    print("  Downstream storage capacity mode:")
    print("    bd  — enter bd directly (recommended for incompressible fluids, e.g. water)")
    print("    Dv  — enter downstream volume; bd = Dv × C(T,P) computed per measurement")
    print("          (recommended for compressible fluids, e.g. argon)")
    print()
    while True:
        bd_mode = input("  Choose mode [bd / Dv]: ").strip().lower()
        if bd_mode == "":
            bd_mode = "dv"
        if bd_mode in ('bd', 'dv'):
            break
        print("   Please type  bd  or  Dv")

    params['bd_mode'] = bd_mode
    if bd_mode == 'bd':
        params['bd']     = prompt("Downstream storage capacity bd (m³/Pa)", 2.2522378352e-15)
        params['bd_err'] = prompt("Error on bd (m³/Pa)",                    5e-17)
        params['Dv']     = None
        params['Dv_err'] = None
    else:
        params['Dv']     = prompt("Downstream volume Dv (m³)",   9.6085e-6)
        params['Dv_err'] = prompt("Error on Dv (m³)",            0.01e-6)
        params['bd']     = None
        params['bd_err'] = None

    params['Temp'] = prompt("Temperature (K)", 423.15)

    valid_permeants = ('water', 'argon', 'rheolube')
    while True:
        permeant = prompt("Permeant fluid [water / argon / rheolube]", "water", cast=str).strip().lower()
        if permeant in valid_permeants:
            params['permeant'] = permeant
            break
        print(f"  ⚠  Invalid choice '{permeant}'. Please enter one of: {valid_permeants}")

    print("\n" + "=" * 60)
    print("DATA FILE INPUT")
    print("=" * 60)
    while True:
        file_mode = prompt("Data file mode [dat / mat]", "dat", cast=str).strip().lower()
        if file_mode in ('dat', 'mat'):
            params['file_mode'] = file_mode
            break
        print("  ⚠  Invalid choice. Please enter 'dat' or 'mat'.")

    if params['file_mode'] == 'dat':
        print("\nDATA FILE COLUMN INDICES  (0-based)")
        params['HeaderRows'] = prompt("Number of header rows to skip",     3,  cast=int)
        params['time_col']   = prompt("Time column index",                 0,  cast=int)
        params['Pup_col']    = prompt("Upstream pressure column index",    1,  cast=int)
        params['Pdwn_col']   = prompt("Downstream pressure column index",  2,  cast=int)
        params['Pc_col']     = prompt("Confining pressure column index",   3,  cast=int)
    else:
        print("\nMATLAB VARIABLE NAMES")
        params['time_var']   = prompt("Time variable", "Time", cast=str).strip()
        params['Pup_var']    = prompt("Upstream pressure variable", "PumpPressure", cast=str).strip()
        params['Pdwn_var']   = prompt("Downstream pressure variable", "Pf", cast=str).strip()
        params['Pc_var']     = prompt("Confining pressure variable", "Normal", cast=str).strip()
        params['time_scale'] = prompt("Time scale factor", 0.001)
        params['Pup_scale']  = prompt("Upstream pressure scale factor", 1.0)
        params['Pdwn_scale'] = prompt("Downstream pressure scale factor", 1.0)
        params['Pc_scale']   = prompt("Confining pressure scale factor", 1.0)

    print("\n" + "=" * 60)
    print("FITTING PARAMETERS")
    print("=" * 60)
    params['N']    = prompt("Number of bootstrap resamples",           20,     cast=int)
    params['w']    = prompt("A/phi weighting factor (0=A only, 1=phi only, 0.5=equal)", 0.5)
    params['Tmin'] = prompt("Minimum oscillation period to search (s)", 100,    cast=float)
    params['Tmax'] = prompt("Maximum oscillation period to search (s)", 10000,  cast=float)

    valid_proc_type = ('sin', 'cont')
    while True:
        proc_type = prompt("Is data a single perm measurement [sin] or continuous [cont]?", "sin", cast=str).strip().lower()
        if proc_type in valid_proc_type:
            params['proc_type'] = proc_type
            break
        print(f"  ⚠  Invalid choice '{proc_type}'. Please enter one of: {valid_proc_type}")

    params['periods_2_proc'] = None
    if params['proc_type'] == 'cont':
        params['periods_2_proc'] = prompt("How many periods do you want to process?", 5, cast=int)

    while True:
        thickness_mode = prompt("Thickness mode [fixed/mean]", "fixed", cast=str).strip().lower()
        if thickness_mode in ('fixed', 'mean'):
            params['thickness_mode'] = thickness_mode
            break
        print("  ⚠  Invalid choice. Please enter 'fixed' or 'mean'.")
    if params['thickness_mode'] == 'mean':
        if params['file_mode'] == 'dat':
            params['thickness_col'] = prompt("Thickness column index (raw values converted to mm)", 4, cast=int)
        else:
            params['thickness_var'] = prompt("Thickness variable name (raw values converted to mm)", "Thickness", cast=str).strip()
        params['thickness_scale'] = prompt("Thickness scale factor to convert raw values to mm", 1.0, cast=float)
        params['thickness_min_mm'] = prompt("Minimum valid thickness after scaling (mm)", 0.0, cast=float)
        params['thickness_max_mm'] = prompt("Maximum valid thickness after scaling (mm)", 5.0, cast=float)

    params['config_file'] = None
    return params


def load_processing_config(datafile):
    """
    Load processing parameters using the config-file workflow.

    Search order:
      1. <selected-data-file>.ini
      2. config.ini beside this script
      3. interactive prompts

    Storage fields belong in [Storage]: mode plus either Dv/Dv_err or
    bd/bd_err. Legacy configs using bd_mode, or with Dv/bd under [Sample],
    are still accepted as fallbacks. New configs should keep all
    downstream-storage settings together in [Storage].
    """
    script_root = Path(__file__).resolve().parent
    data_path = Path(datafile)
    candidates = [data_path.with_suffix('.ini'), script_root / 'config.ini']

    config_file = next((candidate for candidate in candidates if candidate.exists()), None)
    if config_file is None:
        print("\nNo matching .ini file found; falling back to interactive parameter prompts.")
        return prompt_processing_parameters()

    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_file)

    params = {
        'config_file': str(config_file),
        'l':          _config_get(config, 'Sample',     'l',          float, required=True),
        'l_err':      _config_get(config, 'Sample',     'l_err',      float, required=True),
        'dia':        _config_get(config, 'Sample',     'dia',        float, required=True),
        'dia_err':    _config_get(config, 'Sample',     'dia_err',    float, required=True),
        'Temp':       _config_get(config, 'Experiment', 'Temp',       float, required=True),
        'permeant':   _config_get(config, 'Experiment', 'permeant',   str,   required=True).strip().lower(),
        'N':          _config_get(config, 'Fitting',    'N',          int,   required=True),
        'w':          _config_get(config, 'Fitting',    'w',          float, required=True),
        'Tmin':       _config_get(config, 'Fitting',    'Tmin',       float, required=True),
        'Tmax':       _config_get(config, 'Fitting',    'Tmax',       float, required=True),
    }

    proc_type = _config_get(config, ('Processing', 'Fitting'), 'proc_type', str, default=None)
    if proc_type is None:
        proc_type = _config_get(config, ('Processing', 'Fitting'), 'type', str, default='sin')
    proc_type = proc_type.strip().lower()
    if proc_type not in ('sin', 'cont'):
        raise ValueError(f"Invalid processing type '{proc_type}' in {config_file}; use 'sin' or 'cont'.")
    params['proc_type'] = proc_type
    params['periods_2_proc'] = _config_get(config, ('Processing', 'Fitting'), 'periods_2_proc', int, default=None)

    thickness_mode = _config_get(config, ('Sample', 'Processing'), 'thickness_mode', str, default=None)
    if thickness_mode is None:
        thickness_mode = _config_get(config, ('Sample', 'Processing'), 'length_mode', str, default='fixed')
    thickness_mode = thickness_mode.strip().lower()
    if thickness_mode in ('constant', 'const', 'l'):
        thickness_mode = 'fixed'
    if thickness_mode in ('data', 'window_mean', 'moving_mean'):
        thickness_mode = 'mean'
    if thickness_mode not in ('fixed', 'mean'):
        raise ValueError(f"Invalid thickness mode '{thickness_mode}' in {config_file}; use 'fixed' or 'mean'.")
    params['thickness_mode'] = thickness_mode
    params['thickness_col'] = None
    params['thickness_var'] = None
    params['thickness_scale'] = _config_get(config, ('File', 'Sample'), 'thickness_scale', float, default=1.0)
    params['thickness_min_mm'] = _config_get(config, ('Sample', 'File'), 'thickness_min_mm', float, default=0.0)
    params['thickness_max_mm'] = _config_get(config, ('Sample', 'File'), 'thickness_max_mm', float, default=5.0)
    if params['thickness_min_mm'] >= params['thickness_max_mm']:
        raise ValueError("thickness_min_mm must be smaller than thickness_max_mm.")

    file_mode = _config_get(config, 'File', 'mode', str, default=None)
    if file_mode is None:
        # Legacy fallback: infer from the selected data-file extension.
        file_mode = 'mat' if data_path.suffix.lower() == '.mat' else 'dat'
    file_mode = file_mode.strip().lower()
    if file_mode not in ('dat', 'mat'):
        raise ValueError(f"Invalid file mode '{file_mode}' in {config_file}; use 'dat' or 'mat'.")
    params['file_mode'] = file_mode

    if file_mode == 'dat':
        params.update({
            'HeaderRows': _config_get(config, 'File', 'HeaderRows', int, required=True),
            'time_col':   _config_get(config, 'File', 'time_col',   int, required=True),
            'Pup_col':    _config_get(config, 'File', 'Pup_col',    int, required=True),
            'Pdwn_col':   _config_get(config, 'File', 'Pdwn_col',   int, required=True),
            'Pc_col':     _config_get(config, 'File', 'Pc_col',     int, required=True),
        })
        if thickness_mode == 'mean':
            params['thickness_col'] = _config_get(config, ('File', 'Sample'), 'thickness_col', int, required=True)
    else:
        params.update({
            'time_var':   _config_get(config, 'File', 'time_var',   str,   required=True),
            'Pup_var':    _config_get(config, 'File', 'Pup_var',    str,   required=True),
            'Pdwn_var':   _config_get(config, 'File', 'Pdwn_var',   str,   required=True),
            'Pc_var':     _config_get(config, 'File', 'Pc_var',     str,   required=True),
            'time_scale': _config_get(config, 'File', 'time_scale', float, default=1.0),
            'Pup_scale':  _config_get(config, 'File', 'Pup_scale',  float, default=1.0),
            'Pdwn_scale': _config_get(config, 'File', 'Pdwn_scale', float, default=1.0),
            'Pc_scale':   _config_get(config, 'File', 'Pc_scale',   float, default=1.0),
        })
        if thickness_mode == 'mean':
            params['thickness_var'] = _config_get(config, ('File', 'Sample'), 'thickness_var', str, required=True)

    storage_sections = ('Storage', 'Sample')
    bd_mode = _config_get(config, storage_sections, 'mode', str, default=None)
    if bd_mode is None:
        bd_mode = _config_get(config, storage_sections, 'bd_mode', str, default=None)

    has_bd = _config_has(config, storage_sections, 'bd')
    if bd_mode is None:
        bd_mode = 'bd' if has_bd else 'dv'

    bd_mode = bd_mode.strip().lower()
    if bd_mode in ('downstream_volume', 'volume'):
        bd_mode = 'dv'
    if bd_mode not in ('bd', 'dv'):
        raise ValueError(f"Invalid downstream storage mode '{bd_mode}' in {config_file}; use 'bd' or 'Dv'.")

    params['bd_mode'] = bd_mode
    if bd_mode == 'bd':
        params['bd']     = _config_get(config, storage_sections, 'bd',     float, required=True)
        params['bd_err'] = _config_get(config, storage_sections, 'bd_err', float, default=0.0)
        params['Dv']     = None
        params['Dv_err'] = None
    else:
        params['Dv']     = _config_get(config, storage_sections, 'Dv',     float, required=True)
        params['Dv_err'] = _config_get(config, storage_sections, 'Dv_err', float, default=0.0)
        params['bd']     = None
        params['bd_err'] = None

    return params


def print_input_summary(params, datafile):
    """Print a compact per-file processing summary."""
    print("\n" + "=" * 60)
    print("Input summary")
    print("=" * 60)
    if params.get('config_file'):
        print(f"  Config:      {params['config_file']}")
    else:
        print("  Config:      interactive prompts")
    print(f"  Data file:   {os.path.basename(datafile)}")
    print(f"  Processing:  {params['proc_type']}")
    print(f"  Sample:      l = {params['l']:.3f} mm  |  dia = {params['dia']:.3f} mm")
    if params.get('thickness_mode') == 'mean':
        if params['file_mode'] == 'dat':
            source = f"column {params['thickness_col']}"
        else:
            source = f"variable {params['thickness_var']}"
        scope = "selected ROI" if params.get('proc_type') == 'sin' else "each moving window"
        print(f"  Thickness:   mean in {scope} from {source}  |  scale to mm = {params['thickness_scale']}")
        print(f"               valid range: {params['thickness_min_mm']} < thickness <= {params['thickness_max_mm']} mm")
    else:
        print("  Thickness:   fixed sample length from [Sample] l")
    if params['bd_mode'] == 'bd':
        print(f"  Storage:     bd = {params['bd']:.3e} m³/Pa  ±  {params['bd_err']:.3e}  [direct input]")
    else:
        print(f"  Storage:     Dv = {params['Dv']:.3e} m³  ±  {params['Dv_err']:.3e}  →  bd = Dv×C(T,P) per measurement")
    print(f"  Conditions:  T = {params['Temp']} K  |  permeant = {params['permeant']}")
    print(f"  File mode:   {params['file_mode']}")
    if params['file_mode'] == 'dat':
        print(
            "  Columns:     "
            f"time={params['time_col']}  Pup={params['Pup_col']}  "
            f"Pdwn={params['Pdwn_col']}  Pc={params['Pc_col']}  "
            f"header_rows={params['HeaderRows']}"
        )
    else:
        print(
            "  Variables:   "
            f"time={params['time_var']}  Pup={params['Pup_var']}  "
            f"Pdwn={params['Pdwn_var']}  Pc={params['Pc_var']}"
        )
    print(f"  Fitting:     N={params['N']}  w={params['w']}  Tmin={params['Tmin']} s  Tmax={params['Tmax']} s")
    if params['proc_type'] == 'cont' and params.get('periods_2_proc') is not None:
        print(f"  Continuous:  periods per window = {params['periods_2_proc']}")
    print("=" * 60)


def _mat_vector(mat_data, var_name, scale=1.0):
    """Return one MATLAB variable as a 1-D float array with optional scaling."""
    if var_name not in mat_data:
        available = sorted(k for k in mat_data if not k.startswith('__'))
        raise KeyError(
            f"Variable '{var_name}' not found in .mat file. Available variables: {available}"
        )

    values = np.asarray(mat_data[var_name]).squeeze()
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Variable '{var_name}' is not numeric.")
    if values.ndim != 1:
        raise ValueError(
            f"Variable '{var_name}' must be a vector after squeezing; got shape {values.shape}."
        )
    return values.astype(float) * scale


def load_experiment_data(datafile, params):
    """Load time, pressures, confining pressure, and optional thickness series."""
    if params['file_mode'] == 'dat':
        all_file = np.loadtxt(datafile, delimiter='\t', skiprows=params['HeaderRows'])
        time = all_file[:, params['time_col']]
        pup = all_file[:, params['Pup_col']]
        pdwn = all_file[:, params['Pdwn_col']]
        pc = np.mean(all_file[:, params['Pc_col']])
        thickness = None
        if params.get('thickness_mode') == 'mean':
            thickness = all_file[:, params['thickness_col']] * params.get('thickness_scale', 1.0)
        return time, pup, pdwn, pc, thickness

    try:
        mat_data = scipy.io.loadmat(datafile, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            "This .mat file appears to be MATLAB v7.3/HDF5. "
            "scipy.io.loadmat cannot read it; save as v7.2 or add an h5py/mat73 reader."
        ) from exc

    time = _mat_vector(mat_data, params['time_var'], params['time_scale'])
    pup = _mat_vector(mat_data, params['Pup_var'], params['Pup_scale'])
    pdwn = _mat_vector(mat_data, params['Pdwn_var'], params['Pdwn_scale'])
    pc_values = _mat_vector(mat_data, params['Pc_var'], params['Pc_scale'])
    thickness = None
    if params.get('thickness_mode') == 'mean':
        thickness = _mat_vector(mat_data, params['thickness_var'], params.get('thickness_scale', 1.0))

    lengths = {len(time), len(pup), len(pdwn), len(pc_values)}
    if thickness is not None:
        lengths.add(len(thickness))
    if len(lengths) != 1:
        raise ValueError(
            "Configured .mat variables do not have the same length: "
            f"time={len(time)}, Pup={len(pup)}, Pdwn={len(pdwn)}, Pc={len(pc_values)}, "
            f"thickness={len(thickness) if thickness is not None else 'not used'}"
        )

    pc = np.mean(pc_values)
    return time, pup, pdwn, pc, thickness


def window_length_m(params, thickness, start, stop):
    """Return sample length in m and window thickness std in m."""
    if params.get('thickness_mode') != 'mean':
        return params['l'] / 1000, 0.0

    validate_thickness_range(params, thickness, start, stop, context=f"window {start}:{stop}")

    window = np.asarray(thickness[start:stop], dtype=float)
    length_m = float(np.nanmean(window) / 1000)
    if window.size > 1:
        length_std_m = float(np.nanstd(window, ddof=1) / 1000)
    else:
        length_std_m = 0.0
    return length_m, length_std_m


def roi_bounds(idx):
    """Return sorted inclusive ROI bounds from two clicked indices."""
    start, end = sorted(int(i) for i in idx)
    if start == end:
        raise ValueError("Selected ROI has zero length; click two different points.")
    return start, end


def validate_thickness_range(params, thickness, start, stop, context="selected ROI"):
    """Validate configured thickness data in mm before using it for length."""
    if params.get('thickness_mode') != 'mean':
        return
    if thickness is None:
        raise ValueError("thickness_mode = mean requires a configured thickness column/variable.")

    window = np.asarray(thickness[start:stop], dtype=float)
    if window.size == 0:
        raise ValueError(f"No thickness values found in {context}.")

    min_mm = params.get('thickness_min_mm', 0.0)
    max_mm = params.get('thickness_max_mm', 5.0)
    valid = np.isfinite(window) & (window > min_mm) & (window <= max_mm)
    if np.all(valid):
        return

    bad = np.where(~valid)[0]
    examples = []
    for i in bad[:5]:
        value = window[i]
        value_text = f"{value:.6g}" if np.isfinite(value) else str(value)
        examples.append(f"{start + int(i)}:{value_text}")
    raise ValueError(
        f"Invalid thickness values in {context}: {len(bad)}/{window.size} values are outside "
        f"the valid range {min_mm} < thickness <= {max_mm} mm. "
        f"Example index:value = {', '.join(examples)}"
    )


def positive_error_band(values, errors):
    """Return lower/upper arrays for log-scale error shading."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    lower = values - errors
    upper = values + errors
    valid = (
        np.isfinite(values) & np.isfinite(errors) &
        np.isfinite(lower) & np.isfinite(upper) &
        (values > 0) & (errors >= 0) & (lower > 0) & (upper > 0)
    )
    return np.where(valid, lower, np.nan), np.where(valid, upper, np.nan)


def has_positive_values(values):
    """Return True if an array has at least one finite positive value."""
    values = np.asarray(values, dtype=float)
    return bool(np.any(np.isfinite(values) & (values > 0)))


def format_value_error(value, error, unit):
    """Format a value ± error pair, handling undefined/NaN uncertainties."""
    if np.isfinite(error):
        return f"{value:.3e} ± {error:.3e} {unit}"
    return f"{value:.3e} {unit} (uncertainty undefined)"


def select_roi(time, pup, pdwn):
    """Interactively select and refine a region of interest."""
    print("\nSTEP 3: Interactive plot will appear")
    print("        Click 2 points to select COARSE region of interest")

    _fig1 = plt.figure(1)
    _fig1.clf()
    fig, ax = plt.subplots(num=1, clear=True)
    ax.plot(time, pup, 'r', label='Upstream Pressure')
    ax.plot(time, pdwn, 'b', label='Downstream Pressure')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_title('Click start and end points for coarse ROI')
    ax.legend()
    ax.grid()
    tile_figure(1, row=0, col=0)
    make_figure_topmost(fig)
    plt.show(block=False)
    plt.pause(0.1)

    pts = fig.ginput(2, timeout=-1)
    tree = KDTree(np.column_stack((time, pup)))
    idx = [tree.query(pt)[1] for pt in pts]
    idx = list(roi_bounds(idx))

    plt.close(fig)
    mask_roi = (time >= time[idx[0]]) & (time <= time[idx[1]])
    plt.close(10)
    fig2, ax2 = plt.subplots(num=10, figsize=(12, 5), clear=True)
    ax2.plot(time[mask_roi], pup[mask_roi],  'r', lw=0.9, label='Upstream Pressure')
    ax2.plot(time[mask_roi], pdwn[mask_roi], 'b', lw=0.9, label='Downstream Pressure')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pressure (MPa)')
    ax2.set_title('Click START and END points for REFINED ROI')
    ax2.legend()
    ax2.grid()
    tile_figure(10, row=0, col=0)
    make_figure_topmost(fig2)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    pts = fig2.ginput(2, timeout=-1)
    tree = KDTree(np.column_stack((time, pup)))
    idx = [tree.query(pt)[1] for pt in pts]
    idx = list(roi_bounds(idx))

    fig_selected = plt.figure(2)
    fig_selected.clf()
    plt.plot(time, pup, 'r', label='Upstream Pressure')
    plt.plot(time, pdwn, 'b', label='Downstream Pressure')
    plt.axvline(time[idx[0]], color='g', linestyle='--', label='Start Point')
    plt.axvline(time[idx[1]], color='m', linestyle='--', label='End Point')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (MPa)')
    plt.title('Selected Data Range')
    plt.legend()
    tile_figure(2, row=0, col=1)
    make_figure_topmost(fig_selected)
    plt.show(block=False)
    plt.pause(0.1)
    return idx


def fit_selected_window(time, pup, pdwn, idx, params):
    """Fit upstream/downstream signals in the selected ROI."""
    time_sel = time[idx[0]:idx[1] + 1]
    time_sel = time_sel - np.min(time_sel)
    pup_sel = pup[idx[0]:idx[1] + 1]
    pdwn_sel = pdwn[idx[0]:idx[1] + 1]

    updata, dwndata, up_err, dwn_err, up_params_bs, dwn_params_bs = sin_fits_bootstrap(
        pup_sel, pdwn_sel, time_sel, params['N'], params['Tmin'], params['Tmax']
    )
    plot_bootstrap_distributions(up_params_bs, dwn_params_bs, updata, dwndata, 3)

    A = np.abs(dwndata[0] / updata[0])
    Aerr = A * np.sqrt((up_err[0] / (2 * updata[0]))**2 + (dwn_err[0] / (2 * dwndata[0]))**2)
    logAerr = np.abs(Aerr / A / np.log(10))
    phi = updata[2] - dwndata[2]
    phierr = np.sqrt((up_err[2] / 2)**2 + (dwn_err[2] / 2)**2)
    T = updata[1]
    if phi > np.pi:
        phi -= 2 * np.pi
    if phi < -np.pi:
        phi += 2 * np.pi

    upmodel = updata[3] + updata[0] * np.sin(time_sel * 2 * np.pi / updata[1] + updata[2])
    dwnmodel = dwndata[3] + updata[0] * A * np.sin(time_sel * 2 * np.pi / updata[1] + updata[2] - phi) + dwndata[4] * time_sel

    fig_model = plt.figure(4)
    fig_model.clf()
    plt.plot(time_sel, pup_sel, 'r', label='Upstream Pressure')
    plt.plot(time_sel, pdwn_sel, 'b', label='Downstream Pressure')
    plt.plot(time_sel, upmodel, 'g', label='Up Model')
    plt.plot(time_sel, dwnmodel, 'm', label='Down Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (MPa)')
    plt.title('Model Fit')
    plt.legend()
    tile_figure(4, row=1, col=0)
    make_figure_topmost(fig_model)
    plt.show(block=False)
    plt.pause(0.1)

    return {
        'time': time_sel,
        'pup': pup_sel,
        'pdwn': pdwn_sel,
        'updata': updata,
        'dwndata': dwndata,
        'up_err': up_err,
        'dwn_err': dwn_err,
        'up_params_bs': up_params_bs,
        'dwn_params_bs': dwn_params_bs,
        'A': A,
        'Aerr': Aerr,
        'logAerr': logAerr,
        'phi': phi,
        'phierr': phierr,
        'T': T,
    }


def compute_bernabe_outputs(A, Aerr, phi, up_params_bs, dwn_params_bs, params, l, area, up_pressure, T, up_period_err):
    """Compute eta/xi, permeability, storage capacity, and uncertainties."""
    if not np.isfinite(A) or A <= 0 or A >= 1:
        raise ValueError(f"Gain A={A:.6g} is outside the Bernabé domain 0 < A < 1.")
    if not np.isfinite(phi):
        raise ValueError("Phase shift is not finite.")
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"Period T={T:.6g} is invalid.")

    C, visc, bd, bd_err = permeant_props(
        params['permeant'], params['Temp'], up_pressure,
        params['bd_mode'], params['bd'], params['bd_err'], params['Dv'], params['Dv_err']
    )
    xi, eta, Afit, phifit, A0, phi0, x_sol = solve_bern_eq(A, phi, params['w'])
    eta_err, xi_err = bern_errors(x_sol, A, Aerr, phi, params['w'], eta, xi, up_params_bs, dwn_params_bs)

    k = float((eta * np.pi * l * visc * bd) / (area * T))
    kerr = float(np.abs(k) * np.sqrt(
        (eta_err / eta)      ** 2 +
        (params['l_err'] / l) ** 2 +
        (bd_err  / bd)       ** 2 +
        (2 * params['dia_err'] / params['dia']) ** 2 +
        (up_period_err / T)  ** 2
    ))

    if xi == 0:
        bc = 0.0
        bc_err = np.nan
    else:
        bc = float((xi * bd) / (area * l))
        bc_err = float(np.abs(bc) * np.sqrt(
            (xi_err  / xi)       ** 2 +
            (bd_err  / bd)       ** 2 +
            (2 * params['dia_err'] / params['dia']) ** 2 +
            (params['l_err'] / l) ** 2
        ))

    return {
        'C': C, 'visc': visc, 'bd': bd, 'bd_err': bd_err,
        'xi': xi, 'eta': eta, 'Afit': Afit, 'phifit': phifit,
        'A0': A0, 'phi0': phi0, 'x_sol': x_sol,
        'eta_err': eta_err, 'xi_err': xi_err,
        'k': k, 'kerr': kerr, 'bc': bc, 'bc_err': bc_err,
    }


def process_single_measurement(datafile, outfile, params, first_loop, nomo_ax):
    time, pup, pdwn, pc, thickness = load_experiment_data(datafile, params)
    idx = select_roi(time, pup, pdwn)
    validate_thickness_range(params, thickness, idx[0], idx[1] + 1, context="selected ROI")
    fit = fit_selected_window(time, pup, pdwn, idx, params)

    l, l_std = window_length_m(params, thickness, idx[0], idx[1] + 1)
    area = np.pi * (params['dia'] / 2000) ** 2
    bern = compute_bernabe_outputs(
        fit['A'], fit['Aerr'], fit['phi'], fit['up_params_bs'], fit['dwn_params_bs'],
        params, l, area, fit['updata'][3], fit['T'], fit['up_err'][1]
    )

    if first_loop:
        nomo_ax = plot_nomo(5)
    nomo_ax.errorbar(abs(fit['phi']), np.log10(fit['A']), xerr=fit['phierr'], yerr=fit['logAerr'],
                     fmt='o', color='blue', zorder=5, label='_nolegend_')
    nomo_ax.plot(bern['phi0'], np.log10(bern['A0']), '^', color='g', zorder=5, label='_nolegend_')
    nomo_ax.plot(bern['phifit'], np.log10(bern['Afit']), 'x', color='r',
                 markersize=8, markeredgewidth=2, zorder=5, label='_nolegend_')
    tile_figure(5, row=1, col=1)
    plt.figure(5).tight_layout()
    plt.figure(5).canvas.draw_idle()
    plt.pause(0.05)

    file = os.path.basename(datafile)
    output = pd.DataFrame([{
        'File': file,
        'start index': idx[0],
        'end index': idx[1],
        'ConfP': pc,
        'Thickness_mm': l * 1000,
        'ThicknessStd_mm': l_std * 1000,
        'PoreP': fit['updata'][3],
        'UpAmp': fit['updata'][0],
        'Gain': fit['A'],
        'delA': fit['Aerr'],
        'Phase': fit['phi'],
        'delphi': fit['phierr'],
        'Period': fit['T'],
        'delT': fit['up_err'][1],
        'eta': bern['eta'],
        'deleta': bern['eta_err'],
        'xi': bern['xi'],
        'delxi': bern['xi_err'],
        'Permeability': bern['k'],
        'delk': bern['kerr'],
        'Storage Capacity': bern['bc'],
        'delbeta': bern['bc_err']
    }])
    output.to_csv(outfile, mode='w' if first_loop else 'a', header=first_loop, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results {'saved' if first_loop else 'appended'} to {os.path.basename(outfile)}")
    print(f"Thickness used: {l * 1000:.3f} mm")
    print(f"Permeability: {format_value_error(bern['k'], bern['kerr'], 'm²')}")
    print(f"Storage Capacity: {format_value_error(bern['bc'], bern['bc_err'], 'Pa⁻¹')}")
    print(f"{'=' * 60}")
    return nomo_ax


def process_continuous(datafile, outfile, params, first_loop, nomo_ax):
    time_all, pup_all, pdwn_all, pc, thickness_all = load_experiment_data(datafile, params)
    idx = select_roi(time_all, pup_all, pdwn_all)
    validate_thickness_range(params, thickness_all, idx[0], idx[1] + 1, context="selected ROI")
    fit = fit_selected_window(time_all, pup_all, pdwn_all, idx, params)

    roi_start, roi_end = idx[0], idx[1]
    roi_stop = roi_end + 1
    time_roi = time_all[roi_start:roi_stop]
    pup_roi = pup_all[roi_start:roi_stop]
    pdwn_roi = pdwn_all[roi_start:roi_stop]
    thickness_roi = thickness_all[roi_start:roi_stop] if thickness_all is not None else None

    T_main = fit['T']
    N = len(time_roi)
    if N < 2:
        raise ValueError("Selected ROI is too short for continuous processing.")
    del_t = (time_roi[-1] - time_roi[0]) / (N - 1)
    periods_2_proc = params.get('periods_2_proc')
    if periods_2_proc is None:
        periods_2_proc = prompt("How many periods do you want to process?", 5, cast=int)
    N2proc = int(np.floor(periods_2_proc * T_main / del_t))
    step = int(np.floor(T_main / del_t))
    if step < 1:
        raise ValueError("Continuous processing step is <1 sample; check period/time units.")
    if N2proc + 1 >= N:
        raise ValueError(
            "Selected ROI is too short for continuous processing with "
            f"periods_2_proc={periods_2_proc}. Select a longer ROI or reduce periods_2_proc."
        )

    print(
        f"Continuous processing limited to selected ROI: indices {roi_start}:{roi_end} "
        f"({N} samples)."
    )

    A = np.empty(0, dtype=float)
    phi = np.empty(0, dtype=float)
    Aerr = np.empty(0, dtype=float)
    phierr = np.empty(0, dtype=float)
    T = np.empty(0, dtype=float)
    T_err = np.empty(0, dtype=float)
    time2 = np.empty(0, dtype=float)
    xi = np.empty(0, dtype=float)
    eta = np.empty(0, dtype=float)
    eta_err = np.empty(0, dtype=float)
    xi_err = np.empty(0, dtype=float)
    k = np.empty(0, dtype=float)
    bc = np.empty(0, dtype=float)
    k_err = np.empty(0, dtype=float)
    bc_err = np.empty(0, dtype=float)
    Pp = np.empty(0, dtype=float)
    up_amp = np.empty(0, dtype=float)
    thickness_mm = np.empty(0, dtype=float)
    thickness_std_mm = np.empty(0, dtype=float)

    area = np.pi * (params['dia'] / 2000) ** 2

    m = 0
    n = N2proc + 1
    skipped_windows = 0
    with tqdm(total=N, desc="Continuous processing") as pbar:
        while n < N:
            try:
                updata, dwndata, up_err, dwn_err, up_params_bs, dwn_params_bs = sin_fits_bootstrap(
                    pup_roi[m:n], pdwn_roi[m:n], time_roi[m:n], params['N'], params['Tmin'], params['Tmax']
                )
                Ai = np.abs(dwndata[0] / updata[0])
                Aerri = Ai * np.sqrt((up_err[0] / (2 * updata[0]))**2 + (dwn_err[0] / (2 * dwndata[0]))**2)
                phii = updata[2] - dwndata[2]
                phierri = np.sqrt((up_err[2] / 2)**2 + (dwn_err[2] / 2)**2)
                Ti = updata[1]
                if phii > np.pi:
                    phii -= 2 * np.pi
                if phii < -np.pi:
                    phii += 2 * np.pi

                l, l_std = window_length_m(params, thickness_roi, m, n)

                bern = compute_bernabe_outputs(
                    Ai, Aerri, phii, up_params_bs, dwn_params_bs,
                    params, l, area, updata[3], Ti, up_err[1]
                )
            except (RuntimeError, ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                skipped_windows += 1
                if skipped_windows <= 5:
                    tqdm.write(f"Skipping continuous window {roi_start + m}:{roi_start + n}: {exc}")
                m += step
                n += step
                pbar.update(step)
                continue

            A = np.append(A, Ai)
            phi = np.append(phi, phii)
            Aerr = np.append(Aerr, Aerri)
            phierr = np.append(phierr, phierri)
            T = np.append(T, Ti)
            T_err = np.append(T_err, up_err[1])
            time2 = np.append(time2, (time_roi[n] + time_roi[m]) / 2)
            xi = np.append(xi, bern['xi'])
            eta = np.append(eta, bern['eta'])
            eta_err = np.append(eta_err, bern['eta_err'])
            xi_err = np.append(xi_err, bern['xi_err'])
            k = np.append(k, bern['k'])
            bc = np.append(bc, bern['bc'])
            k_err = np.append(k_err, bern['kerr'])
            bc_err = np.append(bc_err, bern['bc_err'])
            Pp = np.append(Pp, updata[3])
            up_amp = np.append(up_amp, updata[0])
            thickness_mm = np.append(thickness_mm, l * 1000)
            thickness_std_mm = np.append(thickness_std_mm, l_std * 1000)

            m += step
            n += step
            pbar.update(step)

    if skipped_windows:
        print(f"Skipped {skipped_windows} continuous window(s) because the fit or Bernabé solve was invalid.")

    plt.figure(5)
    plt.clf()
    ax1 = plt.gca()
    ax1.plot(time2, A, 'r', label='Gain')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Gain', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.plot(time2, phi, 'b', label='Phase Shift rad')
    ax2.set_ylabel('Phase Shift Rad', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Phase and Gain')
    tile_figure(5, row=1, col=1)
    plt.show(block=False)
    plt.pause(0.01)

    plt.figure(6)
    plt.clf()
    ax1 = plt.gca()
    k_plot = np.where(np.isfinite(k) & (k > 0), k, np.nan)
    bc_plot = np.where(np.isfinite(bc) & (bc > 0), bc, np.nan)
    k_low, k_high = positive_error_band(k, k_err)
    bc_low, bc_high = positive_error_band(bc, bc_err)
    ax1.plot(time2, k_plot, 'r', label='Perm')
    ax1.fill_between(time2, k_low, k_high, color='r', alpha=0.18, linewidth=0, label='Perm ± err')
    ax1.set_xlabel('Time (s)')
    if has_positive_values(k_plot):
        ax1.set_yscale('log', base=10)
        ax1.set_ylabel('Permeability m² (log10)', color='r')
    else:
        ax1.plot(time2, k, 'r', alpha=0.35, label='Perm (non-positive)')
        ax1.set_ylabel('Permeability m²', color='r')
        ax1.text(0.02, 0.95, 'No positive permeability values for log10 scale',
                 transform=ax1.transAxes, color='r', va='top', fontsize=9)
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.plot(time2, bc_plot, 'b', label='Storage')
    ax2.fill_between(time2, bc_low, bc_high, color='b', alpha=0.15, linewidth=0, label='Storage ± err')
    if has_positive_values(bc_plot):
        ax2.set_yscale('log', base=10)
        ax2.set_ylabel('Storage Pa⁻¹ (log10)', color='b')
    else:
        ax2.plot(time2, bc, 'b', alpha=0.35, label='Storage (non-positive)')
        ax2.set_ylabel('Storage Pa⁻¹', color='b')
        ax2.text(0.98, 0.95, 'No positive storage values for log10 scale',
                 transform=ax2.transAxes, color='b', va='top', ha='right', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='b')
    tile_figure(6, row=1, col=2)
    plt.show(block=False)
    plt.pause(0.01)

    logAerr = np.abs(Aerr / A / np.log(10))
    if first_loop:
        nomo_ax = plot_nomo(7, no_leg='yes')
    nomo_ax.errorbar(abs(phi), np.log10(A), xerr=phierr, yerr=logAerr,
                     fmt='o', color='blue', zorder=5, label='_nolegend_')
    tile_figure(7, row=1, col=1)
    plt.figure(7).tight_layout()
    plt.figure(7).canvas.draw_idle()
    plt.pause(0.05)

    output = pd.DataFrame({
        'Time': time2,
        'Thickness_mm': thickness_mm,
        'ThicknessStd_mm': thickness_std_mm,
        'PoreP': Pp,
        'UpAmp': up_amp,
        'Gain': A,
        'delA': Aerr,
        'Phase': phi,
        'delphi': phierr,
        'Period': T,
        'delT': T_err,
        'eta': eta,
        'deleta': eta_err,
        'xi': xi,
        'delxi': xi_err,
        'Permeability': k,
        'delk': k_err,
        'Storage Capacity': bc,
        'delbeta': bc_err
    })
    output.to_csv(outfile, mode='w' if first_loop else 'a', header=first_loop, index=False)

    print(f"\n{'=' * 60}")
    print(f"Results {'saved' if first_loop else 'appended'} to {os.path.basename(outfile)}")
    if len(k):
        print(f"Last permeability: {format_value_error(k[-1], k_err[-1], 'm²')}")
        print(f"Last storage capacity: {format_value_error(bc[-1], bc_err[-1], 'Pa⁻¹')}")
        if np.all(np.isfinite(bc)) and np.all(bc == 0) and np.all(~np.isfinite(bc_err)):
            print("Storage remained on the xi=0 boundary; delbeta is undefined and stored as NaN in the CSV.")
    print(f"{'=' * 60}")
    return nomo_ax


def main():
    plt.close('all')
    try:
        plt.pause(0.3)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("STEP 1: Please select where to SAVE the output CSV file")
    print("=" * 60)
    root = create_topmost_root()
    try:
        outfile = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xls"), ("all files", "*.*")],
            title="Output File",
            initialfile="datafile_proc.csv"
        )
    finally:
        root.destroy()

    if not outfile:
        print("Save operation cancelled.")
        return

    print(f"\nOutput will be saved to: {outfile}")
    first_loop = True
    nomo_ax = None
    plt.ion()

    while True:
        print("\n" + "=" * 60)
        print("STEP 2: Please select an INPUT DATA file to process")
        print("=" * 60)
        root = create_topmost_root()
        try:
            datafile = filedialog.askopenfilename(
                defaultextension=".dat",
                filetypes=[("Data files", "*.dat *.mat"), ("DAT files", "*.dat"), ("MAT files", "*.mat"), ("all files", "*.*")],
                title="Select Data File"
            )
        finally:
            root.destroy()

        if not datafile:
            print("No file selected. Exiting loop.")
            break

        print(f"\nLoading data from: {os.path.basename(datafile)}")
        params = load_processing_config(datafile)
        print_input_summary(params, datafile)

        if params['proc_type'] == 'cont':
            nomo_ax = process_continuous(datafile, outfile, params, first_loop, nomo_ax)
        else:
            nomo_ax = process_single_measurement(datafile, outfile, params, first_loop, nomo_ax)

        first_loop = False
        if not ask_to_continue():
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
