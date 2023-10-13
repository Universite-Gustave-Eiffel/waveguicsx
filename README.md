# waveguicsx

waveguicsx, a python library for solving complex waveguide problems
Copyright (C) 2023  Fabien Treyssede

This file is part of waveguicsx.

waveguicsx is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

waveguicsx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with waveguicsx. If not, see <https://www.gnu.org/licenses/>.

Contact: fabien.treyssede@univ-eiffel.fr


### 0. Introduction

waveguicsx is a python library for solving complex waveguide problems based on SLEPc eigensolver.

The full documentation is entirely defined in the `waveguide.py' module.

The following matrix problem is considered: $(\textbf{K}_0-\omega^2\textbf{M}+\text{i}k(\textbf{K}_1+\textbf{K}_1^\text{T})+k^2\textbf{K}_2)\textbf{U}=\textbf{F}$. This kind of problem typically stems from the so-called SAFE (Semi-Analytical Finite Element) method. See references below for theoretical details.

The library contains two classes. The main class, the class Waveguide, enables to solve the waveguide problem defined by the following inputs: $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ and $\textbf{F}$. The other class, the class Signal, is also provided to easily handle the transforms of signals from frequency to time and inversely, as well as the generation of excitation pulses.

The code enables to deal with complex waveguides, two-dimensional (e.g. plates) or three-dimensional (arbitrarily shaped cross-section), inhomogeneous in the transverse directions, anisotropic. Complex-valued problems can be handled including the effects of non-propagating modes (evanescent, inhomogeneous), viscoelastic loss (complex material properties) or perfectly matched layers (PML) to simulate buried waveguides.

The free response ($\textbf{F}=\textbf{0}$) is an eigenvalue problem, solved iteratively by varying the parameter which can be
the angular frequency $\omega$ or the wavenumber $k$. In the former case, the eigenvalue is $k$, while in the latter case, the eigenvalue is $\omega^2$. The loops over the parameter (angular frequency or wavenumber) can be parallelized, as shown in some tutorials (using mpi4py).

Various modal properties (energy velocity, group velocity, excitability...) can be post-processed as a function of the frequency and plotted as dispersion curves.

The forced reponse ($\textbf{F}\neq\textbf{0}$) is solved in the frequency domain by expanding the solution as a sum of eigenmodes using biorthogonality relationship, leading to very fast computations of excited wavefields.

Example:

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


### 1. Prerequisites

waveguicsx requires SLEPc and PETSc (slepc4py, petsc4py).

In the tutorials (examples subfolder), the matrices $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ (and vector $\textbf{F}$ if any) are built from the open finite element (FE) platform FEniCSX. However, any other FE code can be used instead. The only necessary inputs to waveguicsx are the matrices $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ (PETSc matrix format), as well as the vector $\textbf{F}$ for a forced response.

Tutorials are py files, formatted such that they can be opened in a text editor or in a jupyter notebook.


### 2. Installation
 
Simply clone the waveguicsx public repository

```bash
# move to installation location with cd

# get the repo from github
git clone https://github.com/treyssede/waveguicsx.git

# move into repository, and install using pip
cd ./waveguicsx
python3 -m pip install -e .

# test the installation from any location in the path:
python3 -c "from waveguicsx.waveguide import Waveguide; print('ok')"
```

To run FEniCSX for the tutorials, we recommend using a docker image with the latest stable release of DOLFINx executed in complex mode before running scripts:
```bash
docker run -ti dolfinx/dolfinx:stable
source /usr/local/bin/dolfinx-complex-mode
```
or , instead, a Jupyter Lab environment with the latest stable release of DOLFINx:
```bash
docker run --init -ti -p 8888:8888 dolfinx/lab:stable
```
See https://fenicsproject.org/ for details.


### 3. Tutorials

Several tutorials are provided in the examples subfolder.


### 4. Authors and contributors

waveguicsx is currently developed and maintained at Université Gustave Eiffel by Dr. Fabien Treyssède, with some help from Dr. Maximilien Lehujeur (github software management, python formatting, beta testing) and Dr. Pierric Mora (parallelization of loops, beta testing). Please see the AUTHORS file for a list of contributors.

Feel free to contact me by email for further information or questions about waveguicsx.

contact: fabien.treyssede@univ-eiffel.fr


### 5. How to cite

Please cite the software project as follows if used for your own projects or academic publications:

F. Treyssède, waveguicsx (a python library for solving complex waveguides problems), 2023; software available at https://github.com/treyssede/waveguicsx.

For theoretical details about finite element modeling of waveguide problems, here are also a few references by the author about the SAFE modeling of waveguides:

F. Treyssède, L. Laguerre, Numerical and analytical calculation of modal excitability for elastic wave generation in lossy waveguides, Journal of the Acoustical Society of America 133 (2013), 3827–3837

K. L. Nguyen, F. Treyssède, C. Hazard, Numerical modeling of three-dimensional open elastic waveguides combining semi-analytical finite element and perfectly matched layer methods, Journal of Sound and Vibration 344 (2015), 158-178

F. Treyssède, Spectral element computation of high-frequency leaky modes in three-dimensional solid waveguides, Journal of Computational Physics 314 (2016), 341-354

M. Gallezot, F. Treyssède, L. Laguerre, A modal approach based on perfectly matched layers for the forced response of elastic open waveguides, Journal of Computational Physics 356 (2018), 391-409


### 6. License

waveguicsx is freely available under the GNU GPL, version 3.

