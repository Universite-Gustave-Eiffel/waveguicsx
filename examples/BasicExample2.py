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
# Load PETSc matrices, K0, K1, K2 and M, as well as PETSc vector F, saved into the binary file 'BasicExample_K0K1K2MF.dat'.\
# This file contains matrices for a homogeneous plate of thickness 1 and Poisson ratio 0.3.\
# It can be found in the subfolder 'examples'\
# (file generated from the tutorial 'Elastic_Waveguide_Plate2D_TransientResponse.py')
viewer = PETSc.Viewer().createBinary('BasicExample_K0K1K2MF.dat', 'r') #note: calls below must be in order that objects have been stored
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
