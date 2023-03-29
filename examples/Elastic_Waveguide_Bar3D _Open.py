##################################
# 3D (visco-)elastic waveguide example\
# The cross-section is a 2D unit square buried into a PML external elastic medium of thickness 0.25,\
# material: viscoelastic steel into cement grout\
# The waveguide FE formulation (SAFE) leads to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$\
# Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
# The PML has a parabolic profile

import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import matplotlib.pyplot as plt
import pyvista

from waveguicsx.waveguide import Waveguide

#TODO:
#- Dirichlet BC instead of Neumann
#- subdomains: any other method, simpler ?
#- subplots: is it possible to use several data (here, gammax and gammay) without redefining the 'grid' ?


##################################
# Scaled input parameters (materials: steel into cement)
rho, cs, cl, kappas, kappal = 1.0, 1.0, 1.8282, 0.008, 0.003 #density, shear and longitudinal wave celerities, shear and longitudinal bulk wave attenuations (core)
rho_ext, cs_ext, cl_ext, kappas_ext, kappal_ext = 0.2017, 0.5215, 0.8620, 0.100, 0.043 #idem for the external medium
alpha = 2+4j #average value of the absorbing function inside the PML
N = 25 #number of elements along one side of the domain
nev = 10 #number of eigenvalues
omega = np.arange(0.2, 8, 0.2) #frequency range (eigenvalues are wavenumber)
cs, cl, cs_ext, cl_ext = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi), cs_ext/(1+1j*kappas_ext/2/np.pi), cl_ext/(1+1j*kappal_ext/2/np.pi) #complex celerities

##################################
# Create mesh and finite elements (six-node triangles with three dofs per node for the three components of displacement)
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-0.75, -0.75]), np.array([0.75, 0.75])], 
                               [N, N], dolfinx.mesh.CellType.triangle)
element = ufl.VectorElement("CG", "triangle", 2, 3) #Lagrange element, triangle, quadratic "P2", 3D vector
V = dolfinx.fem.FunctionSpace(mesh, element)

##################################
# Create Material properties, discontinuous between core and exterior
def isotropic_law(rho, cs, cl):
    E, nu = rho*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2), 0.5*(cl**2-2*cs**2)/(cl**2-cs**2)
    C11 = C22 = C33 = E/(1+nu)/(1-2*nu)*(1-nu)
    C12 = C13 = C23 = E/(1+nu)/(1-2*nu)*nu
    C44 = C55 = C66 = E/(1+nu)/2
    return np.array([[C11,C12,C13,0,0,0], 
                     [C12,C22,C23,0,0,0], 
                     [C13,C23,C33,0,0,0], 
                     [0,0,0,C44,0,0], 
                     [0,0,0,0,C55,0], 
                     [0,0,0,0,0,C66]]).reshape(36,1)
C0 = isotropic_law(rho, cs, cl)
C_ext = isotropic_law(rho_ext, cs_ext, cl_ext)
def eval_C(x):
    values = C0 * np.logical_and(abs(x[0])<=0.5, abs(x[1])<=0.5) \
           + C_ext * np.logical_or(abs(x[0])>=0.5, abs(x[1])>=0.5)
    return values
Q = dolfinx.fem.FunctionSpace(mesh, ufl.TensorElement("DG", "triangle", 0, (6,6)))
C = dolfinx.fem.Function(Q)
C.interpolate(eval_C) #C.vector.getSize() should be equal to number of cells x 36
# Vizualization
#pyvista.set_jupyter_backend("none"); pyvista.start_xvfb() #uncomment with jupyter notebook (try also: "static", "pythreejs", "ipyvtklink")
plotter = pyvista.Plotter(window_size=[600, 200])
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32)))
grid.cell_data["C"] = C.x.array[0::36].real #index from 0 to 35, 0 being for C11
grid.set_active_scalars("C")
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()

##################################
# Create PML functions, continuous
def eval_gamma(x): #comment on pml to do !!!!!!!!!!!!!!!!!!!!!!!!!!
    values = [1+3*(alpha-1)*((abs(x[0])-0.5)/0.25)**2*(abs(x[0])>=0.5), #gammax
              1+3*(alpha-1)*((abs(x[1])-0.5)/0.25)**2*(abs(x[1])>=0.5)] #gammay
    return values
Q = dolfinx.fem.FunctionSpace(mesh, ufl.VectorElement("CG", "triangle", 1, 2)) #linear "P1"????????????, 2 components (gammay and gammay)
gamma = dolfinx.fem.Function(Q)
gamma.interpolate(eval_gamma)
# Vizualization
plotter = pyvista.Plotter(window_size=[600, 200], shape=(1,2))
gridx = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Q))
#gridx = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)) #????this line is used for plotting mesh only???????????????
gammax = gamma.x.array[0::2]
gridx.point_data["gammax"] = gammax.imag
gridx.set_active_scalars("gammax")
plotter.subplot(0,0)
plotter.add_mesh(gridx, show_edges=True)
plotter.view_xy()
gridy = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Q))
gammay = gamma.x.array[1::2]
gridy.point_data["gammay"] = gammay.imag
gridy.set_active_scalars("gammay")
plotter.subplot(0,1)
plotter.add_mesh(gridy, show_edges=True)
plotter.view_xy()
plotter.show()

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
#TODO: PMLization by including terms gammax=gamma[0] and gammay=gamma[1]!!!!!!!!!!!!!!!!!!!!!!!!!!!!
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Lxy = lambda u: ufl.as_vector([u[0].dx(0), u[1].dx(1), 0, u[0].dx(1)+u[1].dx(0), u[2].dx(0), u[2].dx(1)])
Lz  = lambda u: ufl.as_vector([0, 0, u[2], 0, u[0], u[1]])
k1 = ufl.inner(C*Lxy(u), Lxy(v)) * ufl.dx
k1_form = dolfinx.fem.form(k1)
k2 = (ufl.inner(C*Lz(u), Lxy(v)) - ufl.inner(C*Lxy(u),Lz(v))) * ufl.dx
k2_form = dolfinx.fem.form(k2)
k3 = ufl.inner(C*Lz(u), Lz(v)) * ufl.dx
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
# The parameter is omega, the eigenvalue is k
wg = waveguide.Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
wg.set_parameters(omega=omega)
wg.solve(nev) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]
wg.plot_dispersion()
plt.show()

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

