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
