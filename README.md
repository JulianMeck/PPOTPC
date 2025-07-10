# Pore Pressure Oscilation Technique Processing Code PPOTPC
Code to process pore pressure oscilation permeability measurement technique. The code is based on paper Bernabé et al. (2006). The script takes data of upstream and dowstram pressure varies with time and fits the oscilations with sine waves using a least-squares spectral ananlysis technique. Then from the fits it calculates the amplitude gain of the downstran to upstream and the phase shift between the upstream and downstream. From the gain *A* and phase shift $\phi$ and gain the dimensionless permeability $\eta$ and dimesionless storativity $\xi$ are calculated from eq 1 of Bernabé et al. (2006) given here:

$$A\mathrm{e}^{-\mathrm{i}\phi}=\left(\frac{1+\mathrm{i}}{\sqrt{\xi\eta}}\mathrm{sinh}\left[\left(1+\mathrm{i}\right)\sqrt{\frac{\xi}{\eta}}\right]+\mathrm{cosh}\left[\left(1+\mathrm{i}\right)\sqrt{\frac{\xi}{\eta}}\right]\right)^{-1}$$

This equation cannot be simply rearranged to find $\eta$ and $\xi$ so has to be solved iteratively by seeking to minimize the mismatch function:

$$C\left(\xi_{i},\eta_{i}\right)=w\left[\frac{\log_{10}\left(A_{i}\right)}{\log_{10}\left(A\right)}\right]^{2}+\left(1-w\right)\left[\phi_{i}-\phi\right]^{2}$$

where $A_i$ and $\phi_i$ are the values of gain and phase calculated from equation 1 at each iteration using the current guess for $\eta$ and $\xi$. $w$ is the wieght form 0 to 1 if $w=0.5$ $\xi$ and $\eta$ are equally wieghted. From $\xi$ and $\eta$ the permeability $k$ and the sample storativity $\beta$ are calculated using:

$$\xi=\frac{SL\beta}{\beta_\mathrm{D}}$$
$$\eta=\frac{STk}{\pi L\mu\beta_\mathrm{D}}$$

## References
Bernabé, Y., et al. (2006). "A note on the oscillating flow method for measuring rock permeability." International Journal of Rock Mechanics and Mining Sciences 43(2): 311-316.
