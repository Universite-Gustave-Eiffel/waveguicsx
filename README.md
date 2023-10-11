# waveguicsx

A class for solving complex waveguide problems based on SLEPc eigensolver

The full documentation is entirely defined in the `waveguide.py' module

The following matrix problem is considered: $(\textbf{K}_0-\omega^2\textbf{M}+\text{i}k(\textbf{K}_1+\textbf{K}_1^\text{T})+k^2\textbf{K}_2)\textbf{U}=\textbf{F}$.
This kind of problem typically stems from the so-called SAFE (Semi-Analytical Finite Element) method. See references below for theoretical details.

The inputs are the matrices $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ (PETSc format).
In the tutorials, these matrices are built from the open finite element (FE) platform FEniCSX, but any other FE code could be used instead.

Waveguides can be two-dimensional (e.g. plates), three-dimensional (arbitrarily shaped cross-section), inhomogeneous in the
transverse directions, anisotropic. Complex-valued problems can be handled including the effects of non-propagating modes (evanescent, inhomogeneous),
viscoelastic loss (complex material properties) or perfectly matched layers (PML) to simulate buried waveguides.

The free response ($\textbf{F}=\textbf{0}$) is an eigenvalue problem, solved iteratively by varying the parameter which can be
the angular frequency $\omega$ or the wavenumber $k$. In the former case, the eigenvalue is $k$, while in the latter case, the eigenvalue is $\omega^2$.

Various modal properties (energy velocity, group velocity, excitability...) can be post-processed as a function of the frequency and plotted as dispersion curves.

The forced reponse ($\textbf{F}\neq\textbf{0}$) is solved in the frequency domain by expanding the solution as a sum of eigenmodes using biorthogonality
relationship, leading to very fast computations of excited wavefields.

Another class, the class Signal, is also provided to easily handle the transforms of signals from frequency to time and inversely, as well as the generation of
excitation pulses.

The loop over the parameter (angular frequency or wavenumber) can be parallelized, as shown in tutorials.


### Citation

Please cite the project (https://github.com/treyssede/waveguicsx) if used for your own projects or academic publications


### Example

```python
from waveguicsx.waveguide import Waveguide
parameter = np.arange(0.1, 5, 0.1)
# Initialization
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(omega=parameter) #or: wg.set_parameters(wavenumber=parameter)
# Solution of eigenvalue problem and post-processing of modal properties (iteration over the parameter)
wg.solve(nev=50, target=0) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.compute_energy_velocity()
# Plot dispersion curves
wg.plot()
wg.plot_energy_velocity()
# Forced response in the frequency domain, at degree of freedom dof and axial coordinate z
wg.compute_response_coefficient(F=F, dof=dof)
wg.plot_coefficient()
wg.plot_excitability()
frequency, response = wg.compute_response(dof=dof, z=[1, 10, 50], spectrum=excitation.spectrum)
# Transient response
response = Signal(frequency=frequency, spectrum=response)
response.plot_spectrum()
response.ifft()
response.plot()
plt.show()
```

### Installation
 
Install preferably inside a Docker Container (dolfinx/dolfinx:latest)
```bash
# move to installation location with cd

# get the repo from github
git clone https://github.com/treyssede/waveguicsx.git

# move into to repository, and install using pip
cd ./waveguicsx
python3 -m pip install -e .

# test the installation from any location in the path:
python3 -c "from waveguicsx.waveguide import Waveguide; print('ok')"
```

### A few references by the author about the SAFE modeling of waveguides

F. Treyssède, L. Laguerre, Numerical and analytical calculation of modal excitability for elastic wave generation in lossy waveguides, Journal of the Acoustical Society of America 133 (2013), 3827–3837

K. L. Nguyen, F. Treyssède, C. Hazard, Numerical modeling of three-dimensional open elastic waveguides combining semi-analytical finite element and perfectly matched layer methods, Journal of Sound and Vibration 344 (2015), 158-178

F. Treyssède, Spectral element computation of high-frequency leaky modes in three-dimensional solid waveguides, Journal of Computational Physics 314 (2016), 341-354

M. Gallezot, F. Treyssède, L. Laguerre, A modal approach based on perfectly matched layers for the forced response of elastic open waveguides, Journal of Computational Physics 356 (2018), 391-409
