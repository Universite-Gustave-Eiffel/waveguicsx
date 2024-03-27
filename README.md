[![Generic badge](https://github.com/Universite-Gustave-Eiffel/waveguicsx/actions/workflows/pages/pages-build-deployment/badge.svg)](https://universite-gustave-eiffel.github.io/waveguicsx/)	

**Waveguicsx, a python library for solving complex waveguide problems**

**Copyright (C) 2023-2024  Fabien Treyssede**

This file is part of waveguicsx.

waveguicsx is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

waveguicsx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with waveguicsx. If not, see <https://www.gnu.org/licenses/>.

Contact: fabien.treyssede@univ-eiffel.fr


## 0. Introduction

Waveguicsx is a python library for solving complex waveguide problems based on SLEPc eigensolver.

Waveguicsx is freely available under the GNU GPL, version 3, thus providing an accessible solution to the scientific community. Waveguicsx stems from expertise gained over the past 15 years through the guided wave research activity of the GeoEND team at Université Gustave Eiffel in the fields of non-destructive evaluation and geophysics.

The full documentation is entirely defined in the `waveguide.py' module. See also https://universite-gustave-eiffel.github.io/waveguicsx.

Waveguicsx can deal with complex elastic waveguides, two-dimensional (e.g. plates) or three-dimensional (arbitrarily shaped cross-section), inhomogeneous in the transverse directions, anisotropic. Complex-valued problems can be handled including the effects of non-propagating modes (evanescent, inhomogeneous), viscoelastic loss (complex material properties) or perfectly matched layers (PML) to simulate buried waveguides.

More precisely, waveguicsx solves the following matrix problem: $(\textbf{K}_0-\omega^2\textbf{M}+\text{i}k(\textbf{K}_1+\textbf{K}_1^\text{T})+k^2\textbf{K}_2)\textbf{U}=\textbf{F}$. This kind of problem typically stems from the so-called semi-analytical finite element (FE) method. See references below for theoretical details.

The inputs of waveguicsx are: the matrices $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ (PETSc matrix format) to compute the free response of waveguide (dispersion curves), as well as the excitation vector $\textbf{F}$ if computing forced response is required. These matrices can be built from your own favorite code. In this case, you just need to import these matrices to Python and converted them to PETSc format (see basic examples below). In case you do not have any code to generate these matrices, you can use the open finite element software FEniCSX (installation required) as shown in the tutorials.

The free response ($\textbf{F}=\textbf{0}$) corresponds an eigenvalue problem, solved iteratively by varying the angular frequency $\omega$, or alternatively, the wavenumber $k$, leading to dispersion curve results. In the former case, the eigenvalue is $k$, while in the latter case, the eigenvalue is $\omega^2$. The loops over the parameter (angular frequency or wavenumber) can be parallelized, as shown in some tutorials (using mpi4py). Various modal properties (energy velocity, group velocity, excitability...) can be post-processed as a function of the frequency and plotted as dispersion curves.

The forced reponse ($\textbf{F}\neq\textbf{0}$) is solved in the frequency domain by expanding the solution as a sum of eigenmodes using biorthogonality relationship, leading to very fast computations of the excited wavefields. The transient response can finally be processed in the time domain by inverse FFT.

The library contains two classes. The main class, the class Waveguide, enables to solve the waveguide problem defined by the following inputs: $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ and $\textbf{F}$. The other class, the class Signal, is provided to easily handle the transforms of signals from frequency to time and inversely, as well as the generation of excitation pulses.

Waveguicsx can also solve scattering problems by local inhomogeneities based on transparent boundary conditions (so-called hybrid FE-SAFE method), please see the dedicated section below.


## 1. Basic examples

**Basic example 1: dispersion curves of a homogeneous plate**

```python
###########################################
# Basic example 1: dispersion curves of a homogeneous plate
#
# Important note:\
# This basic example uses previously built PETSc matrices stored into a binary file.\
# If you want to use your own matrices, you have to convert them to PETSc format. Examples of conversion are given below.\
# ** Conversion of a 2d numpy array M to PETSc (dense matrix): **\
# M = PETSc.Mat().createDense(M.shape, array=M)\
# ** Importing sparse matrix M from Matlab to scipy (sparse matrix): **\
# matrices = scipy.io.loadmat('matlab_file.mat') #here, the Matlab file 'matlab_file.mat' is supposed to contain the variable M (Matlab sparse matrix)\
# M = matrices['M'] #'M' is the name of the Matlab variable\
# ** Conversion of a scipy sparse matrix M to PETSc: **\
# M = M.tocsr() #convert to csr format first\
# M = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
from waveguicsx.waveguide import Waveguide

###########################################
# Load PETSc matrices, K0, K1, K2 and M saved into the binary file 'BasicExample.dat'.\
# This file contains matrices for a homogeneous plate of thickness 1 and Poisson ratio 0.3.\
# It can be found in the subfolder 'examples'\
# (file generated from the tutorial 'Elastic_Waveguide_Plate2D_TransientResponse.py')
viewer = PETSc.Viewer().createBinary('BasicExample.dat', 'r') #note: calls below must be in order that objects have been stored
K0 = PETSc.Mat().load(viewer)
K1 = PETSc.Mat().load(viewer)
K2 = PETSc.Mat().load(viewer)
M = PETSc.Mat().load(viewer)

###########################################
# Initialization of waveguide
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(omega=np.arange(0.1, 10.1, 0.1)) #set the parameter range (here, normalized angular frequency)
#wg.set_parameters(wavenumber=np.arange(0.1, 10.1, 0.1)) #uncomment this line if the parameter is the wavenumber instead of the angular frequency (reduce also nev)

###########################################
# Solution of eigenvalue problem (iteration over parameter)
wg.solve(nev=20, target=0) #access to eigensolutions with: wg.eigenvalues[iomega][imode], wg.eigenvectors[iomega][idof,imode]
wg.compute_energy_velocity() #post-process energy velocity

###########################################
# Plot dispersion curves (by default, normalized)
wg.plot() #normalized angular frequency vs. normalized wavenumber
wg.plot_energy_velocity() #normalized energy velocity vs. normalized angular frequency
plt.show()

###########################################
# Example of dimensional plots
h, cs, rho = 0.01, 3260, 7800 #plate thickness (m), shear wave celerity (m/s), density (kg/m**3)
wg.set_plot_scaler(length=h, time=h/cs, mass=rho*h**3, dim=2) #set characteristic length, time and mass
wg.plot() #frequency (Hz) vs. wavenumber (1/m)
#wg.plot_energy_velocity() #energy velocity (m/s) vs. frequency (Hz)
# Energy velocity plot with user-defined units (here, m/ms vs. MHz-mm)
wg.plot_scaler["energy_velocity"] = cs/1000 #units in m/ms
wg.plot_scaler["frequency"] = cs/1000 #frequency units in MHz-mm
sc = wg.plot_energy_velocity(direction=+1) #plot positive-going modes
sc.axes.set_xlabel('Frequency-thickness (MHz-mm)')
sc.axes.set_ylabel('Energy velocity (m/ms)')
plt.show()
```

**Basic example 2: forced response of a homogeneous plate**

```python
###########################################
# Basic example 2: forced response of a homogeneous plate
#
# Important note:\
# This basic example uses previously built PETSc matrices stored into a binary file.\
# If you want to use your own matrices, you have to convert them to PETSc format. Examples of conversion are given below.\
# ** Conversion of a 2d numpy array M to PETSc (dense matrix): **\
# M = PETSc.Mat().createDense(M.shape, array=M)\
# ** Importing sparse matrix M from Matlab to scipy (sparse matrix): **\
# matrices = scipy.io.loadmat('matlab_file.mat') #here, the Matlab file 'matlab_file.mat' is supposed to contain the variable M (Matlab sparse matrix)\
# M = matrices['M'] #'M' is the name of the Matlab variable\
# ** Conversion of a scipy sparse matrix M to PETSc: **\
# M = M.tocsr() #convert to csr format first\
# M = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
from waveguicsx.waveguide import Waveguide, Signal

###########################################
# Load PETSc matrices, K0, K1, K2 and M, as well as PETSc vector F, saved into the binary file 'BasicExample.dat'.\
# This file contains matrices for a homogeneous plate of thickness 1 and Poisson ratio 0.3.\
# It can be found in the subfolder 'examples'\
# (file generated from the tutorial 'Elastic_Waveguide_Plate2D_TransientResponse.py')
viewer = PETSc.Viewer().createBinary('BasicExample.dat', 'r') #note: calls below must be in order that objects have been stored
K0 = PETSc.Mat().load(viewer)
K1 = PETSc.Mat().load(viewer)
K2 = PETSc.Mat().load(viewer)
M = PETSc.Mat().load(viewer)
F = PETSc.Vec().load(viewer)

###########################################
# Input parameters
h, rho, cs = 0.01, 7800, 3218 #plate thickness (m), density (kg/m3), shear wave celerity (m/s)
nev = 20 #number of eigenvalues requested at each frequency

###########################################
# Excitation spectrum (toneburst)
excitation = Signal()
excitation.toneburst(fs=400e3, T=2e-3, fc=100e3, n=8) #central frequency 100 kHz, 8 cycles, duration 2 ms
excitation.plot()
excitation.plot_spectrum()
plt.show()
omega = 2*np.pi*excitation.frequency #angular frequency range (rad/s)
omega = omega*h/cs #normalize angular frequency, because PETSC matrices have been generated for a plate of thickness 1

###########################################
# Initialization of waveguide
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(omega=omega) #set the parameter range (here, normalized angular frequency)
wg.set_plot_scaler(length=h, time=h/cs, mass=rho*h**3, dim=2) #uncomment this line if you want to plot dimensional results, otherwise comment for normalized results

###########################################
# Free response (dispersion curves)
wg.solve(nev=nev, target=0) #solution of eigenvalue problem (iteration over parameter)
wg.compute_group_velocity() #post-process group velocity

###########################################
# Computation of modal coefficients due to excitation vector F
# and modal excitabilities at degree of freedom dof
# F is a unit point force applied normally to the bottom surface of the plate
wg.compute_response_coefficient(F=F, dof=38) #here, dof index 38 is x-component (i.e. normal to the plate) at x=1 (i.e. top of the plate)
sc = wg.plot_energy_velocity(c=['excitability',np.abs], norm='log') #plot the energy velocity colored by the excitability modulus
sc.colorbar.set_label('excitability')
#sc.axes.set_ylim([0, 2*cs]) #set y limits if necessary
#sc.set_clim([1e-14,1e-11]) #set colorbar limits if necessary
plt.show()

###########################################
# Forced response at degree of freedom dof and axial coordinates z (z is normalized by h)
frequency, response = wg.compute_response(dof=38, z=[50], spectrum=excitation.spectrum, plot=False) #response in the frequency domain at z/h=50
response = Signal(frequency=frequency, spectrum=response)
response.plot_spectrum()
response.ifft()
response.plot()
plt.show()
```


## 2. Prerequisites

Waveguicsx requires the complex version of SLEPc and PETSc (slepc4py, petsc4py).

The necessary inputs to waveguicsx are the matrices $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ (PETSc matrix format) to compute the free response of waveguide (dispersion curves), as well as the vector $\textbf{F}$ to compute the forced response. These matrices can be built from any code, then imported to Python and converted to PETSc format.

**Tutorials:**

In the tutorials (see subfolder 'examples'), these matrices are built from the open finite element (FE) platform FEniCSX. To run these tutorials, you will therefore need to install FEniCSX first. Note that tutorials are py files formatted such that they can be conveniently opened in a text editor or in a jupyter notebook.

*Tutorial files are currently written for FEniCSX v0.6.0, minor modifications may be required for use with latest versions (work under progress).*


## 3. Installation

### 3.1 Clone the Waveguicsx public repository

```bash
git clone https://github.com/Universite-Gustave-Eiffel/waveguicsx.git
cd ./waveguicsx
```

### 3.2 Generate the docker image and run the container

FEniCSX is not a dependency of waveguicsx. Nevertheless, it is required to run the tutorials.
We recommend using the docker image of DOLFINX/v0.6.0 sourced in complex mode before running the tutorials :

Install [Docker](https://docs.docker.com/engine/install) and authorize [non-root users](https://docs.docker.com/engine/install/linux-postinstall/).

Then run the following shell script file of waveguicsx repository (launch docker container):
```bash
./launch_fenicsx.sh
```
The first run will install FEniCSX inside the container, which may take time.

*Only the first time* : once the container is launched, install the waveguicsx package using: 
```bash
[complex]waveguicsxuser@hostname:~$ python3 -m pip install -e .
```
*Warning* : the waveguicsx folder (i.e. `.`) is mounted inside the container in `/home/waveguicsxuser` :
so that all changes are persistent and modify the repository of the host system as well.  
The python package files will be installed in the `.local` folder (ignored by `git`), 
so that it is not necessary to reinstall the package with pip each time the container is launched.

Once the container is launched, here are examples of usage inside the container :
```bash
[complex]waveguicsxuser@hostname:~$ python3 ./examples/BasicExample1.py  # run a basic example (check that everything works)
```
```bash
[complex]waveguicsxuser@hostname:~$ python3 ./examples/Elastic_Waveguide_SquareBar3D.py  # run a tutorial in cli
```
```bash
[complex]waveguicsxuser@hostname:~$ jupyter notebook  # launch the jupyter notebook from inside the container
```
```bash
[complex]waveguicsxuser@hostname:~$ exit  # leave the container
```


## 4. Documentation

The documentation is entirely defined in the `waveguide.py' module.

You can also see the full documentation at: https://universite-gustave-eiffel.github.io/waveguicsx.

You can also build the documentation, using `python setup.py doc` and opening the front page in `./doc/Waveguicsx_documentation.html`.


## 5. Tutorials

Various tutorials are provided in the subfolder 'examples'. These tutorials fully depict simple as well as more complex problems, two-dimensional (plates) or three-dimensional (bars, rail...), including viscoelastic loss or perfectly matched layers (used for buried waveguides). In particular, these tutorials show how to build the finite element matrices $\textbf{K}_0$, $\textbf{K}_1$, $\textbf{K}_2$, $\textbf{M}$ with FEniCSX. Installing FEniCSX is therefore required to run the tutorials.

In case you have your own code to generate these matrices, you can readily forget FEniCSX parts in each tutorial, and only consider the part dedicated to waveguicsx.


## 6. Scattering by local inhomogeneities

Waveguicsx can also solve scattering problems by local inhomogeneities based on transparent boundary conditions.

This has been implemented in the scattering.py module, defining a third class named Scattering. The full documentation is entirely defined in the scattering.py module.

The following matrix problem is considered: $(\textbf{K}-\omega^2\textbf{M}-\text{i}\omega\textbf{C})\textbf{U}=\textbf{F}$. This kind of problem typically typically stems from a finite element (FE) model of a small portion of waveguide including a local inhomogeneity (e.g. defects). The cross-section extremities of the truncated FE model are then handled by transparent boundary conditions (BCs) to reproduce semi-infinite waveguides. Such an approach is sometimes called as hybrid FE-SAFE method (see references below for theoretical details). The so-obtained scattering problem is solved repeatedly for each frequency (using PETSc). The loops over the angular frequency can be parallelized, as shown in some tutorials (using mpi4py).

The user must supply the following inputs: the global FE matrices $\textbf{K}$, $\textbf{M}$, $\textbf{C}$ (stiffness, mass and viscous damping), the global FE vector $\textbf{F}$ (excitation inside the FE model) and the transparent BCs (localized by their degrees of freedom in the global vector U). Transparent BCs are Waveguide objects, which must have been solved prior to the scattering problem solution (see waveguide.py module).

For each angular frequency $\omega$, the solution to the scattering problem yields the displacement $\textbf{U}$ of the FE model, as well as the outgoing modal coefficients of every transparent BC.

Note that the FE matrices can be built from any code (then imported to Python and converted to PETSc format). In the tutorials, these matrices are built from the open finite element (FE) platform FEniCSX (see subfolder 'examples').

**Basic scattering example: reflection of Lamb modes by the free edge of a plate**

```python
###########################################
# Basic scattering example: reflection of Lamb modes by the free edge of a plate\
# This simple example involves only one transparent boundary condition (the "inlet"),\
# supposed to be at the left-hand side of the FE box (negative outward normal)\
#
# Important note:\
# This basic example uses previously built PETSc matrices stored into a binary file.\
# If you want to use your own matrices, you have to convert them to PETSc format. Examples of conversion are given below.\
# ** Conversion of a 2d numpy array M to PETSc (dense matrix): **\
# M = PETSc.Mat().createDense(M.shape, array=M)\
# ** Importing sparse matrix M from Matlab to scipy (sparse matrix): **\
# matrices = scipy.io.loadmat('matlab_file.mat') #here, the Matlab file 'matlab_file.mat' is supposed to contain the variable M (Matlab sparse matrix)\
# M = matrices['M'] #'M' is the name of the Matlab variable\
# ** Conversion of a scipy sparse matrix M to PETSc: **\
# M = M.tocsr() #convert to csr format first\
# M = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr, M.indices, M.data))

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
from waveguicsx.waveguide import Waveguide
from waveguicsx.scattering import Scattering

###########################################
# Load PETSc matrices, M, K, Ms, K0, K1, K2 saved into the binary file 'BasicScatteringExample.dat'.\
# This file contains matrices for the reflection of Lamb modes by the free edge of a homogeneous plate of thickness 1 and Poisson ratio 0.25.\
# It can be found in the subfolder 'examples'\
# (file generated from the tutorial 'Scattering_Elastic_Waveguide_Plate2D_Gmsh.py')
viewer = PETSc.Viewer().createBinary('BasicScatteringExample.dat', 'r') #note: calls below must be in order that objects have been stored
for string in ['M', 'K', 'Ms', 'K0', 'K1', 'K2']:
    globals()[string] = PETSc.Mat().load(viewer)
tbc_dofs = PETSc.Vec().load(viewer) #loading dofs of the tbc
tbc_dofs = tbc_dofs[:].real.astype('int32')

###########################################
# Input parameters
omega = 2*np.sqrt(3)*np.linspace(1.48, 1.60, num=100) #normalized angular frequency range
nev = 30 #tbc number of eigenvalues requested at each frequency

###########################################
# Scattering initialization
ws = Scattering(MPI.COMM_WORLD, M, K, 0*M, [('waveguide0', -tbc_dofs)]) #M and K are the mass and stiffness matrices of the FE box
#reminder: tbc_dofs are the global degrees of freedom, set negative by convention when the normal is negative (here, we suppose n=-ey)

###########################################
# Solve waveguide problem associated with the tbc
ws.waveguide0 = Waveguide(MPI.COMM_WORLD, Ms, K0, K1, K2) #Ms, K0, K1 and K2 are SAFE matrices associated with the tbc (here, named 'waveguide0')
ws.waveguide0.set_parameters(omega=omega)
ws.waveguide0.solve(nev)
ws.waveguide0.compute_traveling_direction()
ws.waveguide0.compute_poynting_normalization()

###########################################
# Solving scattering problem
index = np.argmin(np.abs(ws.waveguide0.eigenvalues[0]-4.36)) #4.36 is the S1 wavenumber value at angular frequency 5.13 roughly (normalized values)
mode = ws.waveguide0.track_mode(0, index, threshold=0.98, plot=True) #track a mode, specified by its index at a given frequency, over the whole frequency range
ws.set_ingoing_mode('waveguide0', mode) #set mode as a single ingoing mode, coeff is 1 (here, power is also 1 thanks to poynting normalization)
ws.set_parameters()
ws.solve()

###########################################
# Plot reflected power coefficients vs. angular frequency
ws.waveguide0.compute_complex_power()
sc = ws.waveguide0.plot(y=('complex_power', lambda x:np.abs(np.real(x))), direction=-1)
sc.axes.set_ylim(0, 1.2)
sc.axes.set_ylabel('Reflected power')
plt.show()
```


## 7. Authors and contributors

Waveguicsx is currently developed and maintained at Université Gustave Eiffel by Dr. Fabien Treyssède, with some contributions from Dr. Maximilien Lehujeur (github software management, python formatting, beta testing) and Dr. Pierric Mora (parallelization of loops in tutorials, beta testing). Please see the AUTHORS file for a list of contributors.

Feel free to contact me by email for further information or questions about waveguicsx.

contact: fabien.treyssede@univ-eiffel.fr


## 8. How to cite

Please cite the software project as follows if used for your own projects or academic publications:

F. Treyssède, waveguicsx (a python library for solving complex waveguides problems), 2023; software available at https://github.com/Universite-Gustave-Eiffel/waveguicsx.

For theoretical details about finite element modeling of waveguide problems, here are also a few references by the author about the SAFE modeling of elastic waveguides:

F. Treyssède, L. Laguerre, Numerical and analytical calculation of modal excitability for elastic wave generation in lossy waveguides, Journal of the Acoustical Society of America 133 (2013), 3827–3837

K. L. Nguyen, F. Treyssède, C. Hazard, Numerical modeling of three-dimensional open elastic waveguides combining semi-analytical finite element and perfectly matched layer methods, Journal of Sound and Vibration 344 (2015), 158-178

F. Treyssède, Spectral element computation of high-frequency leaky modes in three-dimensional solid waveguides, Journal of Computational Physics 314 (2016), 341-354

M. Gallezot, F. Treyssède, L. Laguerre, A modal approach based on perfectly matched layers for the forced response of elastic open waveguides, Journal of Computational Physics 356 (2018), 391-409

And here are a few references by the author about the hybrid FE-SAFE method for modeling scattering in elastic waveguides:

F. Benmeddour, F. Treyssède, L. Laguerre, Numerical modeling of guided wave interaction with non-axisymmetric cracks in elastic cylinders, International Journal of Solids and Structures 48 (2011), 764-774

M. Gallezot, F. Treyssède, L. Laguerre, Numerical modelling of wave scattering by local inhomogeneities in elastic waveguides embedded into infinite media, Journal of Sound and Vibration 443 (2019), 310-327


## 9. License

Waveguicsx is freely available under the GNU GPL, version 3.

