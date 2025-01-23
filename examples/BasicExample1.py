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
import os
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
from waveguicsx.waveguide import Waveguide

if PETSc.ScalarType != np.dtype('complex128'):
    raise TypeError(f'Wrong petsc scalar type {PETSc.ScalarType=} '
                    f'please source the complex libraries '
                    f'of dolfinx to run this example ')

###########################################
# Load PETSc matrices, K0, K1, K2 and M saved into the binary file 'BasicExample.dat'.\
# This file contains matrices for a homogeneous plate of thickness 1 and Poisson ratio 0.3.\
# It can be found in the subfolder 'examples'\
# (file generated from the tutorial 'Elastic_Waveguide_Plate2D_TransientResponse.py')

example_folder = os.path.dirname(os.path.abspath(__file__))
example_data_file = os.path.join(example_folder, 'BasicExample.dat')
viewer = PETSc.Viewer().createBinary(example_data_file, 'r') #note: calls below must be in order that objects have been stored
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
