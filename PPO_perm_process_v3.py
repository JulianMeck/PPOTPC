# -*- coding: utf-8 -*-


"""
Created on Mon Jan 20 15:48:58 2025

@author: MBESSJM1
"""
import os
from tkinter import Tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.timeseries import LombScargle
import scipy.io
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import norm
from iapws import IAPWS95
from iapws import _iapws
from argon import Argon_Z
from argon import argon_visc



plt.close('all')

# Function to read data file
def read_datafile():
    Tk().withdraw()  # Close the root window
    filename = filedialog.askopenfilename(title="Pick a datafile", filetypes=[("All files", "*.*")])
    return filename
def lookup_table():
    mat_data = scipy.io.loadmat('lookup.mat')
    A_lookup = mat_data['A']
    phi_lookup=mat_data['phi']
    eta_lookup=mat_data['eta']
    xi_lookup=mat_data['xi']
    return A_lookup,phi_lookup,eta_lookup,xi_lookup
def fitboth(b0,xm,yup,ydwn):
    gup=b0[0]*np.sin(2*np.pi/b0[6]*xm+b0[1])+b0[2]
    gdwn=b0[3]*np.sin(2*np.pi/b0[6]*xm+b0[4])+b0[5]
    E=np.sum(np.abs(gup-yup)**2)+np.sum(np.abs(gdwn-ydwn)**2)
    return E
def get_freq(y,t):
    fs=1/np.mean(np.diff(t)) # calculate sampling frequency
    # look at freqs between 1/10,000 Hz and 1/100 Hz
    freqs=np.linspace(1/10000,1/100,10000)
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

def ls_sin_fit(y,t,fw):
    omega = 2 * np.pi * fw
    cos_part = np.cos(omega * t)
    sin_part = np.sin(omega * t)
    X=np.array([np.ones(np.size(t)), sin_part, cos_part])
    X=X.T
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    #Calculate phase
    phase = np.arctan(beta[2]/beta[1])
    amp = np.sqrt(beta[1]**2+beta[2]**2)
    offset = beta[0]
    #checks if LSSA fit has got result 180° out of phase
    yLSSA1=offset+amp*np.sin((omega*t)+phase)
    yLSSA2=offset+amp*np.sin((omega*t)+phase+np.pi)
    Xi1=np.sum(np.abs(y-yLSSA1)**2)
    Xi2=np.sum(np.abs(y-yLSSA2)**2)
    if Xi2<Xi1:
        phase=phase+np.pi
        # make phi =+/-pi
        if phase>np.pi:
            phase=phase-2*np.pi
        elif phase<-np.pi:
            phase=phase+2*np.pi
    return amp,phase,offset

def fit_sines(up,dwn,t):
    # Get main freq from upstream
    fw,width=get_freq(up,t)
    # Use linear least-squares to fit sin waves
    amp_up,phase_up,offset_up=ls_sin_fit(up, t, fw)
    amp_dwn,phase_dwn,offset_dwn=ls_sin_fit(dwn, t, fw)
    Tw=1/fw
    # Refine fit of sin waves using mimimization fitting both upstream and downstream together
    # use ls as tsrrting point
    b0=np.array([amp_up,phase_up,offset_up,amp_dwn,phase_dwn,offset_dwn,Tw])
    # Set boundaries for amp, phase, offest, period
    lb = [0, -np.pi, min(up), 0, -np.pi, min(dwn), Tw/2]
    ub = [max(up) - min(up),np.pi,max(up),max(dwn) - min(dwn),np.pi,max(dwn),Tw *2]
    bnds = list(zip(lb, ub))
    def mx(b0):
        return fitboth(b0,t,up,dwn)
    # Optimization options
    opts = {"disp": False}
    # Run the optimization
    result = minimize(mx, b0, method='Nelder-Mead', bounds=bnds, options=opts,tol=1e-10)
    # Extract results
    d = result.x
    #d=np.array([amp_up,phase_up,offset_up,amp_dwn,phase_dwn,offset_dwn,T])

    updat=[d[0], d[6], d[1], d[2]]    
    dwndat=[d[3], d[6], d[4], d[5]]
    return updat,dwndat


def sin_fits_bootstrap(up, dwn, t, Num):
    """
    Perform bootstrap resampling to estimate the uncertainty of sinusoidal model parameters.

    Parameters:
    - up, dwn: input data arrays
    - time: time vector
    - N: number of bootstrap iterations
    - fit_sines: function to fit the sinusoidal model

    Returns:
    - updata, dwndata: original fit parameters
    - up_err_boot, dwn_err_boot: bootstrap standard errors
    - params_up, params_dwn: bootstrap parameter samples
    """
    # Fit the original data
    upfit, dwnfit= fit_sines(up, dwn, t)

    # Generate model predictions
    freq = 1/upfit[1]
    yfit_up = upfit[0] * np.sin(2 * np.pi * freq * t + upfit[2]) + upfit[3]
    yfit_dwn = dwnfit[0] * np.sin(2 * np.pi * freq * t + dwnfit[2]) + dwnfit[3]

    # Compute residuals
    res_up = up - yfit_up
    res_dwn = dwn - yfit_dwn

    # Preallocate arrays for bootstrap parameters
    params_up = np.zeros((Num, 4))
    params_dwn = np.zeros((Num, 4))

    for i in range(Num):
        # Resample residuals with replacement
        indx = np.random.randint(0, len(t), len(t))
        up_bs = yfit_up + res_up[indx]
        dwn_bs = yfit_dwn + res_dwn[indx]

        try:
            up_fit, dwn_fit = fit_sines(up_bs, dwn_bs, t)
            params_up[i, :] = up_fit
            params_dwn[i, :] = dwn_fit
        except (ValueError, TypeError):
            params_up[i, :] = np.nan
            params_dwn[i, :] = np.nan

    # Remove failed fits
    params_up = params_up[~np.isnan(params_up).any(axis=1)]
    params_dwn = params_dwn[~np.isnan(params_dwn).any(axis=1)]

    # Compute bootstrap standard errors
    up_err_boot = np.std(params_up, axis=0, ddof=0)
    dwn_err_boot = np.std(params_dwn, axis=0, ddof=0)

    return upfit, dwnfit, up_err_boot, dwn_err_boot, params_up, params_dwn




def plot_bootstrap_distributions(params_up, params_dwn, original_up, original_dwn,plt_num):
    param_names = ['Amplitude', 'Period', 'Phase', 'Offset']
    fig, axes = plt.subplots(4, 2, figsize=(8, 9),num=plt_num)
    plt.figure(plt_num).clf()
    axes = fig.subplots(4, 2)
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
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.show()


# Equation to solve
def bern_eq(x, *data):

    Aexp, phiexp, wie=data
    """
    Equation 1 from Bernabe et al. (2006)    
    Parameters:
    x : list or array
        [log(eta), log(xi)] (logarithmic scales of eta and xi)
    A : float
        Amplitude Gain
    phi : float
        Phase angle
    w : float
        Weighting factor
    
    Returns:
    C : float
        Computed value based on the input parameters
    """
    # Convert from logarithmic scales to original values
    x = 10 ** np.array(x)
    
    # Compute the intermediate terms
    sqrt_x_ratio = np.sqrt(x[1] / x[0])
    term = (1 + 1j) / np.sqrt(x[0] * x[1]) * np.sinh((1 + 1j) * sqrt_x_ratio) + \
        np.cosh((1 + 1j) * sqrt_x_ratio)
    term_inv = 1 / term    
    # Compute A_i and phi_i
    A_i = np.abs(term_inv)
    phi_i = np.angle(term_inv)
    # Adjust phi_i if greater than 0
    if phi_i > 0:
        phi_i -= 2 * np.pi
    phi_i *= -1 
    # Compute the output C
    Cfun = w * (np.log(A_i) / np.log(Aexp) - 1) ** 2 + (1 - wie) * (phi_i - phiexp) ** 2
    return Cfun

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

def plot_nomo(plt_num):
    # Load data for nomogram and to look up starting values of eta and xi
    xis=np.array([0,0.002,0.004,0.008,0.016,0.032,0.064,.128,.256,.512,1,2,4,8,16])
    etas=np.array([1.6,0.8,0.4,0.2,0,-.2,-.4,-0.6,-.8,-1,-1.2,-1.4,-1.6,-1.8])
    A_lookup,phi_lookup,eta_lookup,xi_lookup=lookup_table()
    # Plot nomogram
    plt.figure(plt_num)
    plt.clf()
    green_color = (0, 0.5, 0)
    if first_loop==True:
        for n in range(len(xis)): 
            ind=np.where(xi_lookup==xis[n])[0]
            plt.plot(phi_lookup[ind],np.log10(A_lookup[ind]),color=green_color,linewidth=.8)
            for n in range(len(etas)): 
                ind=np.where(np.round(np.log10(eta_lookup),decimals=1)==etas[n])[0]
                plt.plot(phi_lookup[ind],np.log10(A_lookup[ind]),color='r',linewidth=.8)
    
        plt.xlabel('Phase Shift (rad)')
        plt.ylabel('log$_{10}$ Amplitude Ratio')   
        plt.xlim(0,2*np.pi*0.6)
        plt.ylim(-2,0)
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.text(0.2,-.25,r'$log_{10} \eta = 0.2$',color='red')
        plt.text(0.9,-.37,'0.0',color='red')
        plt.text(1,-.52,'-0.2',color='red')
        plt.text(1.1,-.72,'-0.4',color='red')
        plt.text(1.2,-.92,'-0.6',color='red')
        plt.text(1.25,-1.12,'-0.8',color='red')
        plt.text(1.3,-1.32,'-1.0',color='red')
        plt.text(1.3,-1.52,'-1.2',color='red')
        plt.text(1.3,-1.72,'-1.4',color='red')
        plt.text(1.3,-1.92,'-1.6',color='red')

        # Add text annotations with rotation and color
        plt.text(1.20, -2.25, r'$\xi = 0.002$', rotation=45, color=green_color)
        plt.text(1.42, -2.18, '0.004', rotation=45, color=green_color)
        plt.text(1.55, -2.18, '0.008', rotation=45, color=green_color)
        plt.text(1.68, -2.18, '0.016', rotation=45, color=green_color)
        plt.text(1.85, -2.18, '0.032', rotation=45, color=green_color)
        plt.text(2.20, -2.18, '0.064', rotation=45, color=green_color)
        plt.text(2.65, -2.18, '0.128', rotation=45, color=green_color)
        plt.text(3.15, -2.18, '0.256', rotation=45, color=green_color)
        plt.text(3.55, -2.18, '0.512', rotation=45, color=green_color)

        # Compute 0.605 * 2 * pi
        x_val = 0.605 * 2 * np.pi

        # Add vertical text annotations at the computed x position
        plt.text(x_val, -1.78, '1', color=green_color)
        plt.text(x_val, -1.6, '2', color=green_color)
        plt.text(x_val, -1.48, '4', color=green_color)
        plt.text(x_val, -1.4, '8', color=green_color)
        plt.text(x_val, -1.32, '16', color=green_color)

    plt.show()
    
def solve_bern_eq(A,phi,wie):
    A_lookup,phi_lookup,eta_lookup,xi_lookup=lookup_table()
    # lookup starting values of eta and xi with linear interpolation
    points=np.column_stack((np.log10(A_lookup),phi_lookup))
    # linearly interpolate from values in lookup table
    eta0=griddata(points,eta_lookup,(np.log10(A),phi),method='linear')
    xi0=griddata(points,xi_lookup,(np.log10(A),phi),method='linear')
    if np.size(eta0)>1:
        eta0=eta0[0]
    if np.size(xi0)>1:
        xi0=xi0[0]
    eta0=float(eta0[0])
    xi0=float(xi0[0])
    if xi0<0.1:
        eta0=(2*A)/((1-A**2)**0.5)
    A0=np.abs(((1+1j)/np.sqrt(eta0*xi0)*np.sinh((1+1j)*np.sqrt(xi0/eta0))+\
               np.cosh((1+1j)*np.sqrt(xi0/eta0)))**(-1))
    phi0=np.angle(((1+1j)/np.sqrt(eta0*xi0)*np.sinh((1+1j)*np.sqrt(xi0/eta0))+\
                   np.cosh((1+1j)*np.sqrt(xi0/eta0)))**(-1))
    # Checks phi is in correct range
    if phi0>0:
        phi0=phi0-2*np.pi

    phi0=phi0*-1

    phi_xi0=np.arctan((-(A-1)*(A+1))**0.5/A)
    if phi<phi_xi0:
        eta=(2*A)/((1-A**2)**0.5)
        xi=0
        Afit=A
        phifit=phi
    else:
        # Calculate A and phi from looked up values of eta and xi
        # Starting values for eta and xi
        x0 =np.array([np.log10(eta0), np.log10(xi0)])
        # Set bounds
        lb = [-2, np.log10(0.0001)]
        ub = [1.6, np.log10(32)]
        bounds = list(zip(lb, ub))
        def mx(x):
            return bern_eq(x,A,phi,wie)
        # Optimization options
        options = {"disp": False}
        # Run the optimization
        result = minimize(mx, x0, method="L-BFGS-B", bounds=bounds, options=options)

        # Extract results
        d = result.x
        #hessian_inv=result.hess_inv
        #cov_matrix=hessian_inv.todense()
        #e=np.sqrt(np.diag(cov_matrix))
        eta=10**d[0]
        xi=10**d[1]
        #eta_err=np.abs(eta)*np.log(10)*e[0]
        #xi_err=np.abs(xi)*np.log(10)*e[1]
        # Calculate A and phi from values of eta and xi
        
        Afit=np.abs(((1+1j)/np.sqrt(eta*xi)*np.sinh((1+1j)*np.sqrt(xi/eta))+\
                     np.cosh((1+1j)*np.sqrt(xi/eta)))**(-1))
        phifit=np.angle(((1+1j)/np.sqrt(eta*xi)*np.sinh((1+1j)*np.sqrt(xi/eta))+\
                         np.cosh((1+1j)*np.sqrt(xi/eta)))**(-1))        
           
        # Checks phi is in correct range
        if phifit>0:
            phifit=phifit-2*np.pi
        
        phifit=phifit*-1
       
    return xi,eta,Afit,phifit, A0,phi0

def ask_to_continue():
    root = Tk()
    root.withdraw()  # Hide the main window
    root.update()
    result = messagebox.askyesno("Continue?", "Process another dataset?")
    root.destroy()
    return result



# Pore pressure oscillation permeability processing

# Select data file
#datafile = read_datafile()

# User inputs
# Sample info
l = 12.68  # sample length in mm
l_err = 0.01e-3
dia = 18.83  # sample diameter in mm
dia_err = 0.01
Dv = 9.6085e-6  # downstream vol m^3
Dv_err = 0.01e-6  # error in downstream vol
Temp=293
permeant='water' # 'water' ,'argon' or 'rheolube'
# File info
HeaderRows=3
time_col=0 # time column
Pup_col=1 # upstream Pressure column
Pdwn_col=2 # downstream pressure column
Pc_col=3 # Confining pressure column
# fiting params
N=20 # No of remaples for bootstrapping
w=0.5 # wieght 0.5 equal wiegth of A and phi
# Convert units
l = l / 1000  # length in m
area = np.pi * (dia / 2000) ** 2  # area in m^2


# Prompt user to select output file location once
Tk().withdraw()
outfile = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xls"), ("all files", "*.*")],
    title="Output File",
    initialfile="datafile_proc.csv"
)

if not outfile:
    print("Save operation cancelled.")
else:
    first_loop = True
    while True:
        # Prompt for input data file
        datafile = filedialog.askopenfilename(title="Select Data File")
        if not datafile:
            print("No file selected. Exiting loop.")
            break

        # Load and extract data
        all_file = np.loadtxt(datafile, delimiter='\t', skiprows=HeaderRows)
        # May need to add calibrations here
        time = all_file[:, time_col]
        pup = all_file[:, Pup_col]
        pdwn = all_file[:, Pdwn_col]
        pc=np.mean(all_file[:, Pc_col])

        # Plot for data selection
        plt.figure(1)
        plt.clf()
        plt.plot(time, pup, 'r', label='Upstream Pressure')
        plt.plot(time, pdwn, 'b', label='Downstream Pressure')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (MPa)')
        plt.title('Click start and end points for processing')
        plt.legend()
        plt.grid()
        plt.show()

        # Get user input
        pts = plt.ginput(2, timeout=-1)
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
        time = time[idx[0]:idx[1] + 1] - time[idx[0]]
        pup = pup[idx[0]:idx[1] + 1]
        pdwn = pdwn[idx[0]:idx[1] + 1]

        # Fit and model
        updata, dwndata, up_err, dwn_err, up_params_bs, dwn_params_bs = sin_fits_bootstrap(pup, pdwn, time, N)
        plot_bootstrap_distributions(up_params_bs, dwn_params_bs, updata, dwndata, 3)

        upmodel = updata[3] + updata[0] * np.sin(time * 2 * np.pi / updata[1] + updata[2])
        dwnmodel = dwndata[3] + dwndata[0] * np.sin(time * 2 * np.pi / dwndata[1] + dwndata[2])

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

        # Permeant properties
        if permeant == 'water':
            C, visc = water(Temp, updata[3])
        elif permeant == 'argon':
            C, visc = argon(Temp, updata[3])
        elif permeant == 'rheolube':
            C, visc = rheolube(Temp, updata[3])
        else:
            C, visc = argon(Temp, updata[3])

        bd = Dv * C
        bd_err = C * Dv_err
        A = np.abs(dwndata[0] / updata[0])
        Aerr = A * np.sqrt((up_err[0] / (2 * updata[0]))**2 + (dwn_err[0] / (2 * dwndata[0]))**2)
        logAerr = np.abs(Aerr / A / np.log(10))
        phi = updata[2] - dwndata[2]
        phierr = np.sqrt((up_err[2] / 2)**2 + (dwn_err[2] / 2)**2)
        T = updata[1]
        if phi < 0:
            phi += 2 * np.pi

        xi, eta, Afit, phifit, A0, phi0 = solve_bern_eq(A, phi, w)
        if first_loop==True:
            plot_nomo(5)
        plt.figure(5)
        plt.errorbar(phi, np.log10(A), xerr=phierr, yerr=logAerr, fmt='o')
        
        plt.plot(phi0, np.log10(A0), '^', color='g')
        plt.plot(phifit, np.log10(Afit), 'x', color='r')

        if xi == 0:
            eta_err = eta * np.sqrt((Aerr / A)**2 + (A * Aerr / (1 - A**2))**2)
            xi_err = np.nan
            k = (eta * np.pi * l * visc * bd) / (area * T)
            kerr = np.abs(k) * np.sqrt((eta_err / eta)**2 + (l_err / l)**2 + (bd_err / bd)**2 + (2 * dia_err / dia)**2 + (up_err[1] / T)**2)
            bc = 0
            bc_err = np.nan
        else:
            Adist = dwn_params_bs[:, 0] / up_params_bs[:, 0]
            phidist = up_params_bs[:, 2] - dwn_params_bs[:, 2]
            indices=np.where(phidist<0)[0]
            phidist[indices]=phidist[indices]+2*np.pi
            eta_dist = np.zeros_like(Adist)
            xi_dist = np.zeros_like(Adist)
            for p, (Ai, phii) in enumerate(zip(Adist, phidist)):
                xi_dist[p], eta_dist[p], *_ = solve_bern_eq(Ai, phii, w)
            ind = np.where((xi_dist < 16) & (Adist > 0))[0]
            _, eta_err = norm.fit(eta_dist[ind])
            _, xi_err = norm.fit(xi_dist[ind])
            k = (eta * np.pi * l * visc * bd) / (area * T)
            kerr = np.abs(k) * np.sqrt((eta_err / eta)**2 + (l_err / l)**2 + (bd_err / bd)**2 + (2 * dia_err / dia)**2 + (up_err[1] / T)**2)
            bc = (xi * bd) / (area * l)
            bc_err = np.abs(bc) * np.sqrt((xi_err / xi)**2 + (bd_err / bd)**2 + (2 * dia_err / dia)**2 + (l_err / l)**2)

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
        print(f"Table {'saved' if first_loop else 'appended'} to {outfile}")
        first_loop = False
        if not ask_to_continue():
            break
        #cont = input("Process another dataset? (y/n): ").strip().lower()
        #if cont != 'y':
            #break


