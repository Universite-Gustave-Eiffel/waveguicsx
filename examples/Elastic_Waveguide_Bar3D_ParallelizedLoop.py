##################################
# 3D (visco-)elastic waveguide example\
# The cross-section is a 2D unit square with free boundary conditions on its 1D boundaries, material: viscoelastic steel\
# The waveguide FE formulation (SAFE) leads to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$\
# Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
# In this example:
# - the parameter loop (here, the frequency loop) is distributed on all processes
# - FE mesh and matrices are (therefore) built on each local process
# Reminder for an execution in parallel mode (e.g. 4 processes):
#  mpiexec -n 4 python3 Elastic_Waveguide_Bar3D_ParallelizedLoop.py

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
#import pyvista
#pyvista.set_jupyter_backend("none"); pyvista.start_xvfb() #uncomment with jupyter notebook (try also: "static", "pythreejs", "ipyvtklink")

##################################
# Scaled input parameters
rho, cs, cl, kappas, kappal = 1.0, 1.0, 1.8282, 0.008, 0.003 #density, shear and longitudinal wave celerities, shear and longitudinal bulk wave attenuations
N = 25 #number of elements along one side of the square
nev = 10 #number of eigenvalues
omega = np.arange(0.2, 8, 0.2) #frequency range (eigenvalues are wavenumber)
cs, cl = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi) #complex celerities

##################################
# Create mesh and finite elements (six-node triangles with three dofs per node for the three components of displacement)
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, N, N) #MPI.COMM_SELF = FE mesh is built on each local process
element = ufl.VectorElement("CG", "triangle", 2, 3) #Lagrange element, triangle, quadratic "P2", 3D vector
V = dolfinx.fem.FunctionSpace(mesh, element)

##################################
# Create Material properties (isotropic)
def isotropic_law(rho, cs, cl):
    E, nu = rho*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2), 0.5*(cl**2-2*cs**2)/(cl**2-cs**2)
    C11 = C22 = C33 = E/(1+nu)/(1-2*nu)*(1-nu)
    C12 = C13 = C23 = E/(1+nu)/(1-2*nu)*nu
    C44 = C55 = C66 = E/(1+nu)/2
    return ((C11,C12,C13,0,0,0), 
            (C12,C22,C23,0,0,0), 
            (C13,C23,C33,0,0,0), 
            (0,0,0,C44,0,0), 
            (0,0,0,0,C55,0), 
            (0,0,0,0,0,C66))
C = isotropic_law(rho, cs, cl)
C = dolfinx.fem.Constant(mesh, PETSc.ScalarType(C))

##################################
# Create free boundary conditions (or uncomment lines below for Dirichlet)
bcs = []

##################################
# Define variational problem (SAFE method)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Lxy = lambda u: ufl.as_vector([u[0].dx(0), u[1].dx(1), 0, u[0].dx(1)+u[1].dx(0), u[2].dx(0), u[2].dx(1)])
Lz  = lambda u: ufl.as_vector([0, 0, u[2], 0, u[0], u[1]])
k1 = ufl.inner(C*Lxy(u), Lxy(v)) * ufl.dx
k1_form = dolfinx.fem.form(k1)
k2 = ufl.inner(C*Lz(u), Lxy(v)) * ufl.dx
k2_form = dolfinx.fem.form(k2)
k3 = ufl.inner(C*Lz(u), Lz(v)) * ufl.dx
k3_form = dolfinx.fem.form(k3)
m = rho*ufl.inner(u, v) * ufl.dx
mass_form = dolfinx.fem.form(m)

##################################
# Build PETSc matrices
M = dolfinx.fem.petsc.assemble_matrix(mass_form, bcs=bcs, diagonal=0.0)
M.assemble()
K1 = dolfinx.fem.petsc.assemble_matrix(k1_form, bcs=bcs)
K1.assemble()
K2 = dolfinx.fem.petsc.assemble_matrix(k2_form, bcs=bcs, diagonal=0.0)
K2.assemble()
K3 = dolfinx.fem.petsc.assemble_matrix(k3_form, bcs=bcs, diagonal=0.0)
K3.assemble()

##################################
# Solve the eigenproblem with SLEPc\
# The parameter is k, the eigenvalue is omega**2
# The parameter loop is parallelized
from waveguicsx.waveguide import Waveguide
# Parallelization
comm = MPI.COMM_WORLD #use all processes for the loop
size = comm.Get_size()  #number of processors
rank = comm.Get_rank()  #returns the rank of the process that called it within comm_world
# Split the parameter range and scatter to all
if rank == 0: #define on rank 0 only
    param_split = np.array_split(omega, size) #split param in blocks of length size roughly
else:
    param_split = None
param_local = comm.scatter(param_split, root=0) #scatter 1 block per process
# Solve
wg = Waveguide(MPI.COMM_SELF, M, K1, K2, K3) #MPI.COMM_SELF = SLEPc will used FE matrices on each local process
wg.set_parameters(omega=param_local)
wg.solve(nev)
# Gather
wg.omega = comm.reduce([wg.omega], op=MPI.SUM, root=0) #reduce works for lists: brackets are necessary (wg.omega is not a list but a numpy array)
wg.eigenvalues = comm.reduce(wg.eigenvalues, op=MPI.SUM, root=0)
#wg.eigenvectors = comm.reduce(wg.eigenvectors, op=MPI.SUM, root=0) #don't do this line: reduce cannot pickle 'petsc4py.PETSc.Vec' objects (keep the mode shapes distributed on each processor rather than gather them)
# Plot results
if rank == 0:
    wg.omega = np.concatenate(wg.omega) #wg.omega is transformed to a numpy array for a proper use of wg.plot()
    wg.plot()
    #plt.savefig("Elastic_Waveguide_Bar3D_ParallelizedLoop.svg")
    plt.show()

