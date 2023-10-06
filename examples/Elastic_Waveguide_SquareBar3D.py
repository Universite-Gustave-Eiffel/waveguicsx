##################################
# 3D (visco-)elastic waveguide example\
# The cross-section is a 2D square with free boundary conditions on its boundaries\
# material: viscoelastic steel\
# The waveguide FE formulation (SAFE) leads to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$\
# This eigenproblem is solved with the varying parameter as the wavenumber (eigenvalues are then frequencies) or as the frequency (eigenvalues are then wavenumbers)\
# Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities\
# Results are compared with Euler-Bernoulli analytical solutions for the lowest wavenumber or frequency parameter

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
import pyvista

from waveguicsx.waveguide import Waveguide
#pyvista.set_jupyter_backend("none"); pyvista.start_xvfb() #uncomment with jupyter notebook (try also: "static", "pythreejs", "ipyvtklink")

##################################
# Input parameters
a = 2.7e-3 #core half-length (m)
N = 10 #number of finite elements along one half-side
rho, cs, cl = 7932, 3260, 5960 #core density (kg/m3), shear and longitudinal wave celerities (m/s)
kappas, kappal = 0.008, 0.003 #core shear and longitudinal bulk wave attenuations (Np/wavelength)
omega = 2*np.pi*np.linspace(500e3/1000, 500e3, num=50) #angular frequency range (rad/s)
wavenumber = omega/cs #wavenumber range (1/m, eigenvalues are frequency)
nev = 20 #number of eigenvalues

##################################
# Re-scaling
L0 = a #characteristic length
T0 = a/cs #characteristic time
M0 = rho*a**3#characteristic mass
a = a/L0
rho, cs, cl = rho/M0*L0**3, cs/L0*T0, cl/L0*T0
omega = omega*T0
wavenumber = wavenumber*L0
cs, cl = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi) #complex celerities (core)

##################################
# Create mesh and finite elements (six-node triangles with three dofs per node for the three components of displacement)
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-a, -a]), np.array([a, a])], 
                               [2*N, 2*N], dolfinx.mesh.CellType.triangle)
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
# Dirichlet test case:
#fdim = mesh.topology.dim - 1
#boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim, marker=lambda x: np.logical_or(np.isclose(x[0], -a),
#                                                                                                       np.isclose(x[0], +a)+
#                                                                                                       np.isclose(x[1], -a)+
#                                                                                                       np.isclose(x[1], +a)))
#boundary_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets)
#bcs = [dolfinx.fem.dirichletbc(value=np.zeros(3, dtype=PETSc.ScalarType), dofs=boundary_dofs, V=V)]

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
K0 = dolfinx.fem.petsc.assemble_matrix(k1_form, bcs=bcs)
K0.assemble()
K1 = dolfinx.fem.petsc.assemble_matrix(k2_form, bcs=bcs, diagonal=0.0)
K1.assemble()
K2 = dolfinx.fem.petsc.assemble_matrix(k3_form, bcs=bcs, diagonal=0.0)
K2.assemble()

##################################
# Solve the eigenproblem with SLEPc\
# The parameter is k, the eigenvalue is omega**2
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(wavenumber=wavenumber)
wg.solve(nev) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.plot()
plt.show()

##################################
# Comparison with Euler-Bernoulli analytical solutions
k = wg.wavenumber[0]
print(f'Computed eigenvalues for the first wavenumber (k={k}):\n {np.around(wg.eigenvalues[0],decimals=6)}')
E, mu, I, Ip, S = rho*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2), rho*cs**2, (2*a)**4/12, 0.1406*(2*a)**4, (2*a)**2 #0.1406 is for square section
print(f'Euler-Bernoulli beam solution (only accurate for low frequency):\n \
        Bending wave: {np.around(np.sqrt(E*I/rho/S*k**4),decimals=6)}\n \
        Torsional wave: {np.around(np.sqrt(mu*Ip/rho/(2*I)*k**2),decimals=6)}\n \
        Longitudinal wave: {np.around(np.sqrt(E/rho*k**2),decimals=6)}')

##################################
# Solve the eigenproblem with SLEPc\
# The parameter is omega, the eigenvalue is k
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(omega=omega)
wg.solve(nev) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.plot()
plt.show() #blocking

##################################
# Comparison with Euler-Bernoulli analytical solutions
w = wg.omega[0]
print(f'Computed eigenvalues for the first frequency (omega={w}):\n {np.around(wg.eigenvalues[0],decimals=6)}')
print(f'Euler-Bernoulli beam solution (only accurate for low frequency):\n \
        Bending wave: {np.around(np.sqrt(np.sqrt(rho*S/E/I*w**2)),decimals=6)}\n \
        Torsional wave: {np.around(np.sqrt(rho*2*I/mu/Ip*w**2),decimals=6)}\n \
        Longitudinal wave: {np.around(np.sqrt(rho/E*w**2),decimals=6)}')

##################################
# Mesh visualization
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim))
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()

##################################
# Mode shape visualization
ik, imode = 0, 5 #parameter index, mode index to visualize
vec = wg.eigenvectors[ik].getColumnVector(imode)
u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
u_grid["u"] = np.array(vec).real.reshape(int(np.array(vec).size/V.element.value_shape), int(V.element.value_shape)) #V.element.value_shape is equal to 3
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh
u_plotter.add_mesh(u_grid.warp_by_vector("u", factor=2.0), opacity=0.8, show_scalar_bar=True, show_edges=False) #do not show edges of higher order elements with pyvista
u_plotter.show_axes()
u_plotter.show()


""
