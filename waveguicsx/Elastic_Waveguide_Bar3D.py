##################################
# 3D elastic waveguide\
# The cross-section is a 2D unit square with free boundary conditions on its 1D boundaries\
# Waveguide FE formulation (SAFE) leading to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
import pyvista

##################################
# Scaled input parameters
rho, cs, cl = 1.0, 1.0, 1.8 #density, shear and longitudinal wave celerities
N = 25 #number of elements along one side of the square
nev = 10 #number of eigenvalues

##################################
# Create mesh and finite elements (six-node triangles with three dofs per node for the three components of displacement)
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
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
#boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim, marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
#                                                                                                       np.isclose(x[0], 1.0)+
#                                                                                                       np.isclose(x[1], 0.0)+
#                                                                                                       np.isclose(x[1], 1.0)))
#boundary_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets)
#bcs = [dolfinx.fem.dirichletbc(value=np.zeros(3, dtype=PETSc.ScalarType), dofs=boundary_dofs, V=V)]

##################################
# Define variational problem (SAFE method)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Lxy = lambda u: ufl.as_vector([u[0].dx(0), u[1].dx(1), 0, u[0].dx(1)+u[1].dx(0), u[2].dx(0), u[2].dx(1)])
Ls  = lambda u: ufl.as_vector([0, 0, u[2], 0, u[0], u[1]])
k1 = ufl.inner(C*Lxy(u), Lxy(v)) * ufl.dx
k1_form = dolfinx.fem.form(k1)
k2 = (ufl.inner(C*Ls(u), Lxy(v)) - ufl.inner(Lxy(u),C*Ls(v))) * ufl.dx
k2_form = dolfinx.fem.form(k2)
k3 = ufl.inner(C*Ls(u), Ls(v)) * ufl.dx
k3_form = dolfinx.fem.form(k3)
m = rho*ufl.inner(u, v) * ufl.dx
mass_form = dolfinx.fem.form(m)

##################################
# Build PETSc matrices
M = dolfinx.fem.petsc.assemble_matrix(mass_form, bcs=bcs)
M.assemble()
K1 = dolfinx.fem.petsc.assemble_matrix(k1_form, bcs=bcs)
K1.assemble()
K2 = dolfinx.fem.petsc.assemble_matrix(k2_form, bcs=bcs)
K2.assemble()
K3 = dolfinx.fem.petsc.assemble_matrix(k3_form, bcs=bcs)
K3.assemble()

##################################
# Solve the eigenproblem with SLEPc\
# The parameter is k, the eigenvalue is omega**2
from Waveguide import Waveguide
wg = Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
wg.set_parameters(wavenumber=np.arange(0.1, 2, 0.1))
wg.solve(nev) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.plot_dispersion()
plt.show()

##################################
# Comparison with Euler-Bernoulli analytical solutions (for check)
k = wg.wavenumber[0]
print(f'Computed eigenvalues for the first wavenumber (k={k}):\n {np.around(wg.eigenvalues[0],decimals=10)}')
print(f'Euler-Bernoulli beam solution (only accurate for low frequency):\n \
        Bending wave: {np.around(np.sqrt(k**4*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2)/12*(1**2)),decimals=10)}\n \
        Torsional wave: {np.around(np.sqrt(k**2*cs**2*(6*0.1406)),decimals=10)}\n \
        Longitudinal wave: {np.around(np.sqrt(k**2*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2)),decimals=10)}')

##################################
# Mesh visualization
pyvista.set_jupyter_backend("none") #try also: "static", "pythreejs" ?
pyvista.start_xvfb()
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
u_plotter.add_mesh(u_grid, style="wireframe", color="k")
u_plotter.add_mesh(u_grid.warp_by_vector("u", factor=0.05), show_edges=True)
u_plotter.show_axes()
u_plotter.show()

##################################
# Solve the eigenproblem with SLEPc\
# The parameter is omega, the eigenvalue is k
wg = Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
wg.set_parameters(omega=np.arange(0.2, 8, 0.2))
wg.solve(nev) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.plot_dispersion()
plt.show() #blocking

##################################
# Comparison with Euler-Bernoulli analytical solutions (for check)
w = wg.omega[0]
print(f'Computed eigenvalues for the first frequency (omega={w}):\n {np.around(wg.eigenvalues[0],decimals=10)}')
print(f'Euler-Bernoulli beam solution (only accurate for low frequency):\n \
        Bending wave: {np.around(np.sqrt(w/np.sqrt(cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2)/12*(1**2))),decimals=10)}\n \
        Torsional wave: {np.around(w/np.sqrt(cs*6*0.1406),decimals=10)}\n \
        Longitudinal wave: {np.around(w/np.sqrt(cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2)),decimals=10)}')
