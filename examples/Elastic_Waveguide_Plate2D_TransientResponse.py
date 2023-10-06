##################################
# 2D (visco-)elastic waveguide example (Lamb modes in a plate excited near 1st ZGV resonance)\
# The cross-section is a 1D line with free boundary conditions on its boundaries\
# material: viscoelastic steel\
# The waveguide FE formulation (SAFE) leads to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$\
# This eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers)\
# Viscoelastic loss can be included by introducing imaginary parts (negative) to wave celerities\
# Results are to be compared with Figs. 5b, 7a and 8a of paper: Treyssede and Laguerre, JASA 133 (2013), 3827-3837\
# Note: the depth direction is x, the axis of propagation is z

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
import pyvista

#from waveguicsx.waveguide import Waveguide
from waveguide import Waveguide
pyvista.set_jupyter_backend("none"); pyvista.start_xvfb() #uncomment with jupyter notebook (try also: "static", "pythreejs", "ipyvtklink")

##################################
# Input parameters
h = 0.01 #core half-length (m)
N = 10 #number of finite elements along one half-side
rho, cs, cl = 7800, 3218, 6020 #core density (kg/m3), shear and longitudinal wave celerities (m/s)
kappas, kappal = 0*0.008, 0*0.003 #core shear and longitudinal bulk wave attenuations (Np/wavelength)
nev = 20 #number of eigenvalues

##################################
# Excitation spectrum
from wavesignal import Signal
excitation = Signal(alpha=0*np.log(50)/5e-3)
excitation.toneburst(fs=1000e3, T=5e-3, fc=250e3, n=5)
excitation.plot()
excitation.plot_spectrum()
#excitation.ifft(coeff=1); excitation.plot() #uncomment for check only
plt.show()
omega = 2*np.pi*excitation.frequency #angular frequency range (rad/s)

##################################
# Re-scaling
L0 = h #characteristic length
T0 = h/cs #characteristic time
M0 = rho*h**3#characteristic mass
h = h/L0
rho, cs, cl = rho/M0*L0**3, cs/L0*T0, cl/L0*T0
omega = omega*T0
cs, cl = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi) #complex celerities (core)

##################################
# Create mesh and finite elements (three-node lines with two dofs per node for the two components of displacement)
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, N, np.array([0, h]))
element = ufl.VectorElement("CG", "interval", 2, 2) #Lagrange element, line element, quadratic "P2", 2D vector
V = dolfinx.fem.FunctionSpace(mesh, element)

##################################
# Create Material properties (isotropic)
def isotropic_law(rho, cs, cl):
    E, nu = rho*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2), 0.5*(cl**2-2*cs**2)/(cl**2-cs**2)
    C11 = C22 = E/(1+nu)/(1-2*nu)*(1-nu)
    C12 = E/(1+nu)/(1-2*nu)*nu
    C33 = E/(1+nu)/2
    return ((C11,C12,0), 
            (C12,C22,0), 
            (0,0,C33))
C = isotropic_law(rho, cs, cl)
C = dolfinx.fem.Constant(mesh, PETSc.ScalarType(C))

##################################
# Create free boundary conditions
bcs = []

##################################
# Define variational problem (SAFE method)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Lxy = lambda u: ufl.as_vector([u[0].dx(0), 0, u[1].dx(0)])
Lz = lambda u: ufl.as_vector([0, u[1], u[0]])
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
K0 = dolfinx.fem.petsc.assemble_matrix(k1_form, bcs=bcs)
K0.assemble()
K1 = dolfinx.fem.petsc.assemble_matrix(k2_form, bcs=bcs, diagonal=0.0)
K1.assemble()
K2 = dolfinx.fem.petsc.assemble_matrix(k3_form, bcs=bcs, diagonal=0.0)
K2.assemble()

##################################
# Solve the eigenproblem with SLEPc\
# The parameter is omega, the eigenvalue is k
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(omega=omega)
wg.solve(nev) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.plot()
plt.show() #blocking

###############################################################################
# Excitation force: point force at x=0 oriented in the x-direction
dof_coords = V.tabulate_dof_coordinates()
x0 = np.array([0, 0, 0]) #desired coordinate of point force
dof = int(np.argmin(np.linalg.norm(dof_coords - x0, axis=1))) #find nearest dof
print(f'Point force coordinates (nearest dof):  {(dof_coords[dof,:])}') #check
F = M.createVecRight()
dof = dof*2 + 0 #x-direction
F[dof] = 1

###############################################################################
# Computation of excitabilities and forced response\
# Results are to be compared with Figs. 5b and 7a of Treyssede and Laguerre, JASA 133 (2013), 3827-3837
wg.compute_response_coefficient(F=F, dof=dof)
ax = wg.plot_excitability()
ax.set_yscale('log')
ax.set_ylim(1e-3,1e2)
frequency, response, axs = wg.compute_response(dof=dof, z=h/2, spectrum=excitation.spectrum, plot=True)
axs[0].get_lines()[0].set_color("black")
plt.close()

###############################################################################
# Time response\
# Results are to be compared with Fig. 8a of Treyssede and Laguerre, JASA 133 (2013), 3827-3837
response = Signal(frequency=frequency, spectrum=response, alpha=0*np.log(50)/5e-3*T0)
response.plot_spectrum()
response.ifft(coeff=1)
response.plot()
plt.show()

##################################
# Mesh visualization
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim))
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.add_mesh(grid, style='points', render_points_as_spheres=True, point_size=10)
plotter.view_xy()
plotter.show()

##################################
# Mode shape visualization
ik, imode = 100, 5 #parameter index, mode index to visualize
vec = wg.eigenvectors[ik].getColumnVector(imode)
u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
u_grid["u"] = np.array(vec).real.reshape(int(np.array(vec).size/V.element.value_shape), int(V.element.value_shape)) #V.element.value_shape is equal to 2
u_grid["u"] = np.insert(u_grid["u"], 1, 0, axis=1) #insert a zero column to the second component (the y component)
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh
u_plotter.add_mesh(u_grid.warp_by_vector("u", factor=1e-2), opacity=0.8, show_scalar_bar=True, show_edges=False) #do not show edges of higher order elements with pyvista
u_plotter.view_zx()
u_plotter.show_axes()
u_plotter.show()

""

