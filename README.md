# Pore Pressure Oscilation Technique Processing Code PPOTPC
## Julian Mecklenburgh (University of Manchester), Izzy Ashman (Liverpool University) and Dan Faulkner (Liverpool University)
Code to process pore pressure oscilation permeability measurement technique. The code is based on paper Bernabé et al. (2006). The script takes data of upstream and dowstram pressure varies with time and fits the oscilations with sine waves using a least-squares spectral ananlysis technique. Then from the fits it calculates the amplitude gain of the downstran to upstream and the phase shift between the upstream and downstream. From the gain $A$ and phase shift $\phi$ the dimensionless permeability $\eta$ and dimesionless storativity $\xi$ are calculated from eq 1 of Bernabé et al. (2006) given here:

$$A\mathrm{e}^{-\mathrm{i}\phi}=\left(\frac{1+\mathrm{i}}{\sqrt{\xi\eta}}\mathrm{sinh}\left[\left(1+\mathrm{i}\right)\sqrt{\frac{\xi}{\eta}}\right]+\mathrm{cosh}\left[\left(1+\mathrm{i}\right)\sqrt{\frac{\xi}{\eta}}\right]\right)^{-1}$$

A graphical representation of this eqquation is given here:
<img width="607" height="475" alt="image" src="https://github.com/user-attachments/assets/15a93b07-ba4f-4571-96e2-ead6b92663c5" />

This equation cannot be simply rearranged to find $\eta$ and $\xi$ so has to be solved iteratively by seeking to minimize the mismatch function:

$$C\left(\xi_{i},\eta_{i}\right)=w\left[\frac{\log_{10}\left(A_{i}\right)}{\log_{10}\left(A\right)}\right]^{2}+\left(1-w\right)\left[\phi_{i}-\phi\right]^{2}$$

where $A_i$ and $\phi_i$ are the values of gain and phase calculated from equation 1 at each iteration using the current guess for $\eta$ and $\xi$. $w$ is the wieght factor from 0 to 1 if $w=0.5$ $A$ and $\phi$ are equally wieghted. From $\xi$ and $\eta$ the permeability $k$ and the sample storativity $\beta$ are calculated using:

$$\xi=\frac{SL\beta}{\beta_\mathrm{D}}$$
$$\eta=\frac{STk}{\pi L\mu\beta_\mathrm{D}}$$

where $S$ is the cross sectional area of sample in $\mathrm{m^2}$, $L$ is the length in m, $\beta_\mathrm{D}$ is the downstream storage in $\mathrm{m^3/Pa}$, $T$ period (s) and $\mu$ is the fluid viscosity in Pa s. The downstream strorage is the product of the downstream volume and the fluid compressibility and it can either be measured directly by measuring how pressure changes for a given change in volume or the downstream volume can be measured and the compressibility calculated from thermodynamic data.

## Input Data
The code requires times seies data of upstream and downstream pore pressure in our labs this is stored in a tab deliminated text file with time in seconds and the pore pressures in MPa. The other use inputs for example sample dimensions are added to the main program in the User inputs segment:
```
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
HeaderRows=3 # Number of header rows
time_col=0 # time column
Pup_col=1 # upstream Pressure column
Pdwn_col=2 # downstream pressure column
Pc_col=3 # Confining pressure column
# fiting params
N=20 # No of remaples for bootstrapping
w=0.5 # wieght 0.5 equal wiegth of A and phi
```
You can set the column number in the input data file for time, upstream pressure and downstream pressure using the parameters ```time_col```, ```Pup_col``` and ```Pdwn_col```. Then also the number of header rows needs to be set with ```HeaderRows```. The specific pore fluid used in the experiment can be set be setting the parameter ```permeant``` to either 'water', 'argon' or 'rheolube' the three main permeants used in the lab in Manchester. Other permeants could be used but the compressibility and viscosity would need to be defined.  For water the compressibity and viscosity is calculated from pressure and temperature using the International Assocation for the Properties of Water and Steam (IAPWS) python project for calculating the properties of water ```iapws```. For argon there is a python script that calculates the compressibilty and viscosity from pressure and temperature using published formulations.  For Rheolube there is a procedure in the script that calculates the compressibilty and viscosity from pressure and temperature using published formulations.  

## LSSA Method
The script uses a least-squares spectral analysis (LSSA) technique to fit the sine wave oscilations of the upstream and downstream waves. The is preferred to FFT method as you do not need to have a whole number of waveforms, the data do not have to be evenly sampled and the method can reliably fit a single waveform although results are best with ~5-10 waves. First the frequency (or period) of the inputted upstream oscilations is calculated by evaluating the Lomb-Scargle periodogram that outputs the power-spectrum density using the ```LombScargle``` function from the ```Astropy``` project. Then the upstream amplitiude $A_{\mathrm{up}}$, downstream amplitude $A_{\mathrm{dwn}}$, upstream phase $\phi_{\mathrm{up}}$, downstream phase $\phi_{\mathrm{dwn}}$, upstream offset $C_{\mathrm{up}}$ and downstream offset $C_{\mathrm{dwn}}$ are claculated using a linear regression where you solve $\alpha$, $\beta$ and $\gamma$ in the following equation:

$$y=A+B\mathrm{sin}(\omega t)+C\mathrm{cos}(\omega t)$$

by solving the matrix equation:

$$\begin{pmatrix}
y_1\\
y_2\\
\vdots\\
y_n\\
\end{pmatrix}=
\begin{pmatrix}
1&\mathrm{sin}(\omega t_1)&\mathrm{cos}(\omega t_1)\\
1&\mathrm{sin}(\omega t_2)&\mathrm{cos}(\omega t_2)\\
\vdots&\vdots&\vdots\\
1&\mathrm{sin}(\omega t_n)&\mathrm{cos}(\omega t_n)\\
\end{pmatrix}
\begin{pmatrix}
\alpha\\
\beta\\
\gamma\\
\end{pmatrix}
$$
where the amplitude $A$, offest $B$ and phase $\phi$ are calculated using:
$$ A=\sqrt{\beta^{2}+\gamma^{2}}$$
$$ B=\alpha$$ and
$$\phi = \mathrm{atan}\left(\frac{\gamma}{\beta}\right)$$
One issue with this fitting is it cannot tell the difference between 2 sin waves 180° out of phase so a check is made to look at residuals of for the fit parameter and a fit 180° phase shifted the one with lowest residuals is then chosen. Then we refine the fit using a minimization routine where we fit both upstream and down stream together with the same period. This is to avoid using a different period to fit upstream and downstream. We minimize the function $E$:
$$ g_{\mathrm{up}}=C_{\mathrm{up}}+A_{\mathrm{up}}\mathrm{sin}(2\pi f t + \phi_{\mathrm{up}})$$
$$ g_{\mathrm{dwn}}=C_{\mathrm{dwn}}+A_{\mathrm{dwn}}\mathrm{sin}(2\pi f t + \phi_{\mathrm{dwn}})$$
$$ E=\sum{(g_{\mathrm{up}}-P_{\mathrm{up}})^2+(g_{\mathrm{dwn}}-P_{\mathrm{dwn}})^2}$$

Then we can calculate the phase shift and gain between the upstream and downstream.

To estimate the errors in the fitted parameters we use a bootstrapping technique. We do this by randomnly resampling the residuals and adding them to the fit. We resample the residuals $N$ times and get $N$ values of the fitted parameters then we can use the standard deviation of the boot strapped fitted parameters to estimate the errors. The script plots the distributions of the fitted parameters.
<img width="785" height="899" alt="image" src="https://github.com/user-attachments/assets/3f534dba-ada0-4d8e-a84a-97b4310073f4" />


## Solving the Bernabé equation
From the amplitude gain $A$ and the phase shift $\phi$ the dimensionles permeabilty $\eta$ and the dimensionless storativity $\xi$ are calculated by iteratively solving equation above. The inital values $\eta_0$ and $\xi_0$ are estimated by linearly interpolating from values in a lookup table.  We use the ```minimize``` function in the ```scipy.optimize``` functions to iteratively solve for the values of $\eta$ and $\xi$. Errors in $\eta$ and $\xi$ are estimated by using the distributions of $A$ and $\phi$ from the boot strapping.

## Output
The script saves a csv text file with the following columns:
|File|start index|end index|ConfP|PoreP|UpAmp|Gain|delA|Phase|delphi|Period|delT|eta|deleta|xi|delxi|Permeability|delk|Storage Capacity|delbeta|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Implementation
To use this script you need the following packages installed in python:

```tkinter```
```pandas```
```numpy```
```matplotlib```
```seaborn```
```astropy```
```scipy```
```iapws```

Also in the folder you are working in you will need the ```argon.py``` file and the ```lookup.mat``` file. We use Spyder (6.0.7) running Python 3.11.7 to run the script. To be able to use graphical user interaction with figures you will need to set the graphics backend to Tk. To do this go to Tools > Preferences then  select IPython console  and Graphics tab setting graphics backend to Tk.

<img width="906" height="693" alt="image" src="https://github.com/user-attachments/assets/e6eab440-f068-4278-9e2d-ea167b0d40ba" />

Before running the script user inputs are set in the script see section above.

When you run the script you will be asked: 
1) where you want to save the processed data.
2) where the experimental data file is.
3) select the start of data you want to process and the end on the figure.
4) the script will chugg away and calculate permeability and display amplitude and phase shit on the nomogram.
5) asked whether you want to process another data set if yes you goto step 2 above if no end
6) output file is saved.



## References
Bernabé, Y., et al. (2006). "A note on the oscillating flow method for measuring rock permeability." International Journal of Rock Mechanics and Mining Sciences 43(2): 311-316.

https://iapws.org/release.html

https://pypi.org/project/iapws/
