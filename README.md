# Pore Pressure Oscilation Technique Processing Code PPOTPC
## Julian Mecklenburgh (University of Manchester), Izzy Ashman (Liverpool University) and Dan Faulkner (Liverpool University)
Code to process pore pressure oscilation permeability measurement technique. The code is based on paper Bernabé et al. (2006). The script takes data of upstream and dowstram pressure varies with time and fits the oscilations with sine waves using a least-squares spectral ananlysis technique. Then from the fits it calculates the amplitude gain of the downstream to upstream and the phase shift between the upstream and downstream. From the gain $A$ and phase shift $\phi$ the dimensionless permeability $\eta$ and dimesionless storativity $\xi$ are calculated from eq 1 of Bernabé et al. (2006) given here:

$$A\mathrm{e}^{-\mathrm{i}\phi}=\left(\frac{1+\mathrm{i}}{\sqrt{\xi\eta}}\mathrm{sinh}\left[\left(1+\mathrm{i}\right)\sqrt{\frac{\xi}{\eta}}\right]+\mathrm{cosh}\left[\left(1+\mathrm{i}\right)\sqrt{\frac{\xi}{\eta}}\right]\right)^{-1}$$

A graphical representation of this equation is given here:

<img width="607" height="475" alt="image" src="https://github.com/user-attachments/assets/15a93b07-ba4f-4571-96e2-ead6b92663c5" />

This equation cannot be simply rearranged to find $\eta$ and $\xi$ so has to be solved iteratively by seeking to minimize the mismatch function:

$$C\left(\xi_{i},\eta_{i}\right)=w\left[\frac{\log_{10}\left(A_{i}\right)}{\log_{10}\left(A\right)}\right]^{2}+\left(1-w\right)\left[\phi_{i}-\phi\right]^{2}$$

where $A_i$ and $\phi_i$ are the values of gain and phase calculated from equation 1 at each iteration using the current guess for $\eta$ and $\xi$. $w$ is the wieght factor from 0 to 1 if $w=0.5$ $A$ and $\phi$ are equally wieghted. From $\xi$ and $\eta$ the permeability $k$ and the sample storativity $\beta$ are calculated using:

$$\xi=\frac{SL\beta}{\beta_\mathrm{D}}$$
$$\eta=\frac{STk}{\pi L\mu\beta_\mathrm{D}}$$

where $S$ is the cross sectional area of sample in $\mathrm{m^2}$, $L$ is the length in m, $\beta_\mathrm{D}$ is the downstream storage in $\mathrm{m^3/Pa}$, $T$ period (s) and $\mu$ is the fluid viscosity in Pa s. The downstream strorage is the product of the downstream volume and the fluid compressibility and it can either be measured directly by measuring how pressure changes for a given change in volume or the downstream volume can be measured and the compressibility calculated from thermodynamic data.

## Input Data
The code requires time series data of upstream and downstream pore pressure. The preferred/default workflow is to read tab-delimited `.dat` files with `[File] mode = dat`. If the source data are SHIVA/MATLAB `.mat` files, use `[File] mode = mat` to read them directly rather than first converting them to `.dat`. Other user inputs, for example sample dimensions, are added to the `.ini` config file:
```
[Sample]
# sample length in mm
l = 12.68
l_err = 0.01e-3
# sample thickness mode: fixed uses l; mean is optional for continuous processing
thickness_mode = fixed
# sample diameter in mm
dia = 18.83
dia_err = 0.01

[Storage]
# downstream storage mode: Dv or bd
# Dv mode calculates beta_D = Dv x fluid compressibility for each measurement
# bd mode uses a directly measured downstream storage capacity
mode = Dv
# downstream volume in m^3
Dv = 9.6085e-6
# error in downstream volume
Dv_err = 0.01e-6

[Experiment]
Temp = 293
# 'water' ,'argon' or 'rheolube'
permeant = water 

[File]
# data file type: dat or mat
mode = dat
# Number of header rows
HeaderRows = 3
# time column
time_col = 0
# upstream Pressure column
Pup_col = 1
# downstream pressure column
Pdwn_col = 2
# Confining pressure column
Pc_col = 3 

[Processing]
# processing type: sin for one selected oscillation interval, cont for continuous moving-window processing
proc_type = sin

[Fitting]
# No of remaples for bootstrapping
N = 20
# weight 0.5 equal wiegth of A and phi
w = 0.5
#min period s
Tmin = 10
#max period s
Tmax = 10000 
# print Bernabé xi boundary/cutoff diagnostics during main fits
xi_debug = false
# show measured data and best-fit curves for continuous moving windows
debug_bestfit = false
debug_bestfit_every = 1
debug_bestfit_pause = 0.05
```
You can set the column number in the input data file for time, upstream pressure and downstream pressure using the parameters ```time_col```, ```Pup_col``` and ```Pdwn_col```. Then also the number of header rows needs to be set with ```HeaderRows```. The specific pore fluid used in the experiment can be set be setting the parameter ```permeant``` to either 'water', 'argon' or 'rheolube' the three main permeants used in the lab in Manchester. Other permeants could be used but the compressibility and viscosity would need to be defined.  For water the compressibity and viscosity is calculated from pressure and temperature using the International Assocation for the Properties of Water and Steam (IAPWS) python project for calculating the properties of water ```iapws```. For argon there is a python script written by Julian that calculates the compressibilty and viscosity from pressure and temperature using published formulations.  For Rheolube there is a procedure in the script that calculates the compressibilty and viscosity from pressure and temperature using published formulations.  

### Downstream storage mode: `Dv` or `bd`

The script supports two ways of setting the downstream storage capacity, $\beta_\mathrm{D}$.

Use `Dv` mode when you know the downstream volume and want the script to calculate storage from the fluid compressibility at the measurement pressure:

```ini
[Storage]
mode = Dv
Dv = 9.6085e-6
Dv_err = 0.01e-6
```

Use `bd` mode when you have directly measured downstream storage capacity:

```ini
[Storage]
mode = bd
bd = 2.2522378352e-15
bd_err = 5e-17
```

If `mode` is omitted, the script uses `bd` mode when a `bd` value is present; otherwise it falls back to the older `Dv` mode. Legacy configs using `bd_mode` are still accepted. The script first looks for an `.ini` file with the same base name as the selected data file, then falls back to the root `config.ini`.

### Data file mode: `dat` or `mat`

The `[File]` section controls how the input data file is read. Use `mode = dat` for tab-delimited `.dat` files and column indices:

```ini
[File]
mode = dat
HeaderRows = 1
time_col = 16
Pup_col = 18
Pdwn_col = 22
Pc_col = 17
```

Use `mode = mat` to read a MATLAB `.mat` file directly without first converting it to `.dat`. In this mode, provide variable names instead of column numbers:

```ini
[File]
mode = mat
time_var = Time
Pup_var = PumpPressure
Pdwn_var = Pf
Pc_var = Normal

# Optional scale factors. For SHIVA files, Time is commonly stored in ms,
# so time_scale = 0.001 converts it to seconds.
time_scale = 0.001
Pup_scale = 1
Pdwn_scale = 1
Pc_scale = 1
```

For `.mat` files, the configured variables must be numeric vectors of the same length. `scipy.io.loadmat()` can read normal MATLAB `.mat` files up to v7.2. MATLAB v7.3/HDF5 files need an additional reader such as `h5py` or `mat73`.

### Processing type: `sin` or `cont`

The `[Processing]` section controls whether the script processes one selected pore-pressure oscillation interval or a continuous moving-window experiment:

```ini
[Processing]
proc_type = sin
```

Use `proc_type = sin` for the standard workflow where one region of interest is selected and one permeability result is appended to the output CSV.

Use `proc_type = cont` for continuous/cycling pore-pressure experiments where permeability is calculated through time using a moving window. The wave period should be approximately constant over the selected data. The selected region of interest is used to estimate the initial wave parameters, then the script processes only that selected ROI with time-based windows of `periods_2_proc` periods and steps forward by one period. Window start/stop positions are chosen from the actual timestamps, not from a fixed number of sample indices:

```ini
[Processing]
proc_type = cont
periods_2_proc = 5
```

During continuous processing, individual windows can be skipped if the sine fit does not converge or if the fitted gain is outside the physical Bernabé domain (`0 < A < 1`). The run continues and reports how many windows were skipped. The continuous permeability/storage plot uses base-10 logarithmic y-axes and shaded `± error` bands when positive values are available; if a series is entirely non-positive, that axis falls back to linear scale with a note on the plot. Non-positive values are retained in the CSV.

By default, both `sin` and `cont` processing use the fixed sample length `l` from `[Sample]`:

```ini
[Sample]
l = 12.68
thickness_mode = fixed
```

Optionally, the script can use the mean sample thickness from the data file. Set `thickness_mode = mean` and provide either a `.dat` column or a `.mat` variable. Thickness values are converted to mm using `thickness_scale` before the mean is calculated. Before fitting, the selected ROI is checked for valid thickness values; by default values must satisfy `0 < thickness <= 5` mm.

- with `proc_type = sin`, the mean is calculated inside the selected ROI;
- with `proc_type = cont`, the mean is calculated inside each moving window.

```ini
[Sample]
thickness_mode = mean

[File]
# for mode = dat
thickness_col = 4
# raw thickness value x thickness_scale = mm
thickness_scale = 1
thickness_min_mm = 0
thickness_max_mm = 5
```

For direct MATLAB input:

```ini
[Sample]
thickness_mode = mean

[File]
# for mode = mat
thickness_var = Thickness
# raw thickness value x thickness_scale = mm
thickness_scale = 1
thickness_min_mm = 0
thickness_max_mm = 5
```

The output CSV includes `Thickness_mm` and `ThicknessStd_mm`: for `sin` they refer to the selected ROI, and for `cont` they refer to each processed moving window. Existing configs that omit `thickness_mode` keep the old behavior (`fixed`).

When the Bernabé solution lies on the `xi = 0` boundary, `Storage Capacity` is written as `0`, while `delbeta` is written as `NaN` because the uncertainty of a boundary-pinned storage term is not defined.

To diagnose why `xi` is being set to zero, enable optional debug output:

```ini
[Fitting]
xi_debug = true
```

This prints `A`, `phi`, the `xi = 0` boundary phase `phi_xi0`, `phi - phi_xi0`, interpolated starting values, and whether the numerical `xi < 1e-4` floor/cutoff was applied during the main Bernabé solve. Bootstrap solves are not printed.

To inspect the sine fit used inside continuous processing, enable:

```ini
[Fitting]
debug_bestfit = true
debug_bestfit_every = 1
debug_bestfit_pause = 0.05
```

This updates Figure 8 during `proc_type = cont` with two stacked subplots: upstream measured data as black points with the upstream best fit as a solid red line, and downstream measured data as black points with the downstream best fit as a solid blue line. The downstream subplot also shows the fitted offset + linear trend component as a solid cyan line. Each plotted Figure 8 is saved automatically as a PNG in a subfolder with the same stem as the output CSV, for example `data/s2077RED/figure8_window_0001_idx_...png` for `data/s2077RED.csv`. Increase `debug_bestfit_every` to plot/save fewer windows, or increase `debug_bestfit_pause` to keep each debug window visible longer. Use a finite pause value; indefinite GUI waits can freeze with TkAgg when the figure is closed. Closing Figure 8 disables `debug_bestfit` and lets the continuous analysis continue normally.

### Replotting continuous CSV results

Continuous permeability/storage results can be replotted from an existing output CSV without rerunning the full processing workflow:

```bash
python plot_continuous_results.py data/s2077RED.csv
```

If no CSV path is provided, the script opens a file picker. To save a figure without opening a plot window:

```bash
python plot_continuous_results.py data/s2077RED.csv --save s2077RED_perm_storage.png --no-show
```

Measured gain/phase values can also be replotted on a Bernabé nomogram from an existing CSV:

```bash
python plot_nomogram_results.py data/s2077RED.csv
```

For a non-interactive save:

```bash
python plot_nomogram_results.py data/s2077RED.csv --save s2077RED_nomogram.png --no-show
```

For diagnostics of whether points fall below the `xi = 0` nomogram boundary, plot upstream amplitude, gain, phase, and `phi - arccos(Gain)` through time:

```bash
python plot_continuous_diagnostics.py data/s2077RED.csv
```

Negative `phi - arccos(Gain)` values are below the `xi = 0` boundary. To save without opening a plot window:

```bash
python plot_continuous_diagnostics.py data/s2077RED.csv --save s2077RED_diagnostics.png --no-show
```

## LSSA Method
The script uses a least-squares spectral analysis (LSSA) technique to fit the sine wave oscilations of the upstream and downstream waves. The is preferred to FFT method as you do not need to have a whole number of waveforms, the data do not have to be evenly sampled and the method can reliably fit a single waveform although results are best with ~5-10 waves. First the frequency (or period) of the inputted upstream oscilations is calculated by evaluating the Lomb-Scargle periodogram that outputs the power-spectrum density using the ```LombScargle``` function from the ```Astropy``` project. Then the upstream amplitiude $A_{\mathrm{up}}$, downstream amplitude $A_{\mathrm{dwn}}$, upstream phase $\phi_{\mathrm{up}}$, downstream phase $\phi_{\mathrm{dwn}}$, upstream offset $C_{\mathrm{up}}$ and downstream offset $C_{\mathrm{dwn}}$ are claculated using a linear regression where you solve $\alpha$, $\beta$ and $\gamma$ in the following equation:

$$y=\alpha+\beta\mathrm{sin}(\omega t)+\gamma\mathrm{cos}(\omega t)$$

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
$$ C=\alpha$$ and
$$\phi = \mathrm{atan}\left(\frac{\gamma}{\beta}\right)$$
One issue with this fitting is it cannot tell the difference between 2 sin waves 180° out of phase so a check is made to look at residuals of for the fit parameter and a fit 180° phase shifted the one with lowest residuals is then chosen. Then we refine the fit using a minimization routine where we fit both upstream and down stream together with the same period. This is to avoid using a different period to fit upstream and downstream. We minimize the function $E$:
$$ g_{\mathrm{up}}=C_{\mathrm{up}}+A_{\mathrm{up}}\mathrm{sin}(2\pi f t + \phi_{\mathrm{up}})$$
$$ g_{\mathrm{dwn}}=C_{\mathrm{dwn}}+A_{\mathrm{dwn}}\mathrm{sin}(2\pi f t + \phi_{\mathrm{dwn}})+L_{\mathrm{dwn}}t$$
$$ E=\sum{(g_{\mathrm{up}}-P_{\mathrm{up}})^2+(g_{\mathrm{dwn}}-P_{\mathrm{dwn}})^2}$$

Then we can calculate the phase shift and gain between the upstream and downstream.

To estimate the errors in the fitted parameters we use a bootstrapping technique. We do this by randomnly resampling the residuals and adding them to the fit. We resample the residuals $N$ times and get $N$ values of the fitted parameters then we can use the standard deviation of the boot strapped fitted parameters to estimate the errors. The script plots the distributions of the fitted parameters.
<img width="785" height="899" alt="image" src="https://github.com/user-attachments/assets/3f534dba-ada0-4d8e-a84a-97b4310073f4" />


## Solving the Bernabé equation
From the amplitude gain $A$ and the phase shift $\phi$ the dimensionles permeabilty $\eta$ and the dimensionless storativity $\xi$ are calculated by iteratively solving equation above. The inital values $\eta_0$ and $\xi_0$ are estimated by linearly interpolating from values in a lookup table.  We use the ```minimize``` function in the ```scipy.optimize``` functions to iteratively solve for the values of $\eta$ and $\xi$. Errors in $\eta$ and $\xi$ are estimated by using the distributions of $A$ and $\phi$ from the boot strapping.

## Output
The script saves a csv text file with the following columns for single-interval (`sin`) processing:
|File|start index|end index|ConfP|Thickness_mm|ThicknessStd_mm|PoreP|UpAmp|Gain|delA|Phase|delphi|Period|delT|eta|deleta|xi|delxi|Permeability|delk|Storage Capacity|delbeta|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Continuous (`cont`) processing saves the same fitted/calculated quantities as a time series, with `Time`, `Thickness_mm`, and `ThicknessStd_mm` for each moving window.

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
```numdifftools```
```tqdm```

Also in the folder you are working in you will need the ```argon.py``` file and the ```lookup.mat``` file. 

### Spyder
We use Spyder (6.0.7) running Python 3.11.7 to run the script. To be able to use graphical user interaction with figures you will need to set the graphics backend to Tk. To do this go to Tools > Preferences then  select IPython console  and Graphics tab setting graphics backend to Tk.

<img width="906" height="693" alt="image" src="https://github.com/user-attachments/assets/e6eab440-f068-4278-9e2d-ea167b0d40ba" />

### Anaconda setup
```
conda env create --file environment.yml
conda activate permeability
python PPO_perm_process.py
```

### Workflow
Before running the script, set user inputs in the `.ini` config file as described above.

When you run the script you will be asked: 
1) where you want to save the processed data.
2) where the experimental data file is.
3) select the start of data you want to process and the end on the figure.
4) the script will chugg away and calculate permeability and display amplitude and phase shit on the nomogram.
5) asked whether you want to process another data set if yes you goto step 2 above if no end
6) output file is saved.
## Continuous Data
There is functionality in this version to calculate how permeability varies with time. Set `[Processing] proc_type = cont` in the `.ini` file. The wave period should be approximately constant over the whole dataset. First, select a region of interest to calculate the initial wave parameters. The script then calculates permeability for a moving window of `periods_2_proc` waves and steps the window forward by one whole period. The data are plotted as time series and on the nomogram, and the processed data are saved as a CSV file.
## Acknowledgements
This code has been under development for many years with involvement from a number of coleagues in Manchester and Liverpool including: Ernie Rutter, Rosanne McKernan, Kier Groves, Mike Chandler, Rochelle Taylor, Yusuf Bashir, Lining Yang, Pete Armitage, John Bedford... Note need to add more people from Liverpool.

## References
Bernabé, Y., et al. (2006). "A note on the oscillating flow method for measuring rock permeability." International Journal of Rock Mechanics and Mining Sciences 43(2): 311-316.

https://iapws.org/release.html

https://pypi.org/project/iapws/
