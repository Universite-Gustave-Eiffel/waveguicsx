##################################
# 3D (visco-)elastic waveguide example\
# The cross-section is a 2D square buried into a PML external elastic medium\
# material: viscoelastic steel into cement grout\
# The waveguide FE formulation (SAFE) leads to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$\
# Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities\
# The PML has a parabolic profile\
# The FE mesh is built from Gmsh\
# Results are to be compared with Fig. 8 of paper: Treyssede, Journal of Computational Physics 314 (2016), 341–354

import gmsh #full documentation entirely defined in the `gmsh.py' module
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
b, le = 1.5*a, a/16 #half-length including PML (m), finite element characteristic length (m)
rho_core, cs_core, cl_core = 7932, 3260, 5960 #core density (kg/m3), shear and longitudinal wave celerities (m/s)
kappas_core, kappal_core = 0.008, 0.003 #core shear and longitudinal bulk wave attenuations (Np/wavelength)
rho_ext, cs_ext, cl_ext = 1600, 1700, 2810 #for the external medium
kappas_ext, kappal_ext = 0.100, 0.043 #for the external medium
alpha = 2+4j #average value of the absorbing function inside the PML
omega = 2*np.pi*np.linspace(0, 26e6/(a*1e3), num=200) #angular frequencies (rad/s)
nev = 10 #number of eigenvalues
target = lambda omega: omega/cl_core.real #target set at the longitudinal wavenumber of core to find L(0,n) modes

##################################
# Re-scaling
L0 = a #characteristic length
T0 = a/cs_core #characteristic time
M0 = rho_core*a**3#characteristic mass
a, b, le = a/L0, b/L0, le/L0
rho_core, cs_core, cl_core = rho_core/M0*L0**3, cs_core/L0*T0, cl_core/L0*T0
rho_ext, cs_ext, cl_ext = rho_ext/M0*L0**3, cs_ext/L0*T0, cl_ext/L0*T0
omega = omega*T0
cs_core, cl_core = cs_core/(1+1j*kappas_core/2/np.pi), cl_core/(1+1j*kappal_core/2/np.pi) #complex celerities (core)
cs_ext, cl_ext = cs_ext/(1+1j*kappas_ext/2/np.pi), cl_ext/(1+1j*kappal_ext/2/np.pi) #complex celerities (exterior)

##################################
# Create mesh from Gmsh and finite elements (six-node triangles with three dofs per node for the three components of displacement)
gmsh.initialize()
# Core
gmsh.model.geo.addPoint(+a, -a, 0, le, 1)
gmsh.model.geo.addPoint(+a, +a, 0, le, 2)
gmsh.model.geo.addPoint(-a, +a, 0, le, 3)
gmsh.model.geo.addPoint(-a, -a, 0, le, 4)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
core = gmsh.model.geo.addPlaneSurface([1])
# External medium
gmsh.model.geo.addPoint(+b, -b, 0, le, 5)
gmsh.model.geo.addPoint(+b, +b, 0, le, 6)
gmsh.model.geo.addPoint(-b, +b, 0, le, 7)
gmsh.model.geo.addPoint(-b, -b, 0, le, 8)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 5, 8)
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
exterior = gmsh.model.geo.addPlaneSurface([2, 1])
# Physical groups
gmsh.model.geo.synchronize() #the CAD entities must be synchronized with the Gmsh model
gmsh.model.addPhysicalGroup(2, [core], tag=1) #2: for 2D (surface)
gmsh.model.addPhysicalGroup(2, [exterior], tag=2) #2: for 2D (surface)
gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], tag=21) #1: for 1D (line)
# Generate mesh
gmsh.model.mesh.generate(2) #generate a 2D mesh
gmsh.model.mesh.setOrder(2) #interpolation order for the geometry, here 2nd order
# From gmsh to fenicsx
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
# # Reminder for save & read
# gmsh.write("Elastic_Waveguide_Bar3D_Open.msh") #save to disk
# mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("Elastic_Waveguide_Bar3D_Open.msh", MPI.COMM_WORLD, rank=0, gdim=2)
gmsh.finalize() #called when done using the Gmsh Python API
# Visualize FE mesh with pyvista
Vmesh = dolfinx.fem.FunctionSpace(mesh, ufl.FiniteElement("CG", "triangle", 1)) #order 1 is properly handled with pyvista
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Vmesh))
grid.cell_data["Marker"] = cell_tags.values
grid.set_active_scalars("Marker")
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()
# Finite element space
element = ufl.VectorElement("CG", "triangle", 2, 3) #Lagrange element, triangle, quadratic "P2", 3D vector
V = dolfinx.fem.FunctionSpace(mesh, element)

##################################
# Create Material properties, discontinuous between core and exterior
def isotropic_law(rho, cs, cl):
    E, nu = rho*cs**2*(3*cl**2-4*cs**2)/(cl**2-cs**2), 0.5*(cl**2-2*cs**2)/(cl**2-cs**2)
    C11 = C22 = C33 = E/(1+nu)/(1-2*nu)*(1-nu)
    C12 = C13 = C23 = E/(1+nu)/(1-2*nu)*nu
    C44 = C55 = C66 = E/(1+nu)/2
    C = np.array([[C11,C12,C13,0,0,0], 
                     [C12,C22,C23,0,0,0], 
                     [C13,C23,C33,0,0,0], 
                     [0,0,0,C44,0,0], 
                     [0,0,0,0,C55,0], 
                     [0,0,0,0,0,C66]])
    C = C[np.triu_indices(6)] #the upper-triangle part of C (21 elements only)
    return PETSc.ScalarType(C)
C_core = isotropic_law(rho_core, cs_core, cl_core)
C_ext = isotropic_law(rho_ext, cs_ext, cl_ext)
Q = dolfinx.fem.FunctionSpace(mesh, ufl.TensorElement("DG", "triangle", 0, (6,6), symmetry=True)) #symmetry enables to store 21 elements instead of 36
C = dolfinx.fem.Function(Q)
cells = cell_tags.find(1) #core (tag=1)
C.x.array[[range(21*c,21*c+21) for c in cells]] = np.tile(C_core, (len(cells),1))
cells = cell_tags.find(2) #exterior (tag=2)
C.x.array[[range(21*c,21*c+21) for c in cells]] = np.tile(C_ext, (len(cells),1))
# Same approach for density
Q = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
rho = dolfinx.fem.Function(Q)
cells = cell_tags.find(1) #core (tag=1)
rho.x.array[cells] = np.tile(rho_core, len(cells))
cells = cell_tags.find(2) #exterior (tag=2)
rho.x.array[cells] = np.tile(rho_ext, len(cells))
# Vizualization
plotter = pyvista.Plotter(window_size=[600, 400])
gridC = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)) #or *dolfinx.plot.create_vtk_mesh(Vmesh)
gridC.cell_data["Cij"] = C.x.array[0::21].real #index from 0 to 35, 0 being for C11...
#gridC.cell_data["Cij"] = rho.x.array.real #for checking density
gridC.set_active_scalars("Cij")
plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh (with vertices of order 1 elements only owing to pyvista) 
plotter.add_mesh(gridC, opacity=0.8, show_scalar_bar=True, show_edges=False)
plotter.add_text('Re(Cij)', 'upper_edge', color='black', font_size=8)
plotter.view_xy()
plotter.show()

##################################
# Create Cartesian PML functions, continuous
def eval_gamma(x): #pml profile: continuous and parabolic
    values = [1+3*(alpha-1)*((abs(x[0])-a)/(b-a))**2*(abs(x[0])>=a), #gammax
              1+3*(alpha-1)*((abs(x[1])-a)/(b-a))**2*(abs(x[1])>=a)] #gammay
    return values
Q = dolfinx.fem.FunctionSpace(mesh, ufl.VectorElement("CG", "triangle", 2, 2))
gamma = dolfinx.fem.Function(Q)
gamma.interpolate(eval_gamma)
# Vizualization
plotter = pyvista.Plotter(window_size=[800, 400], shape=(1,2))
gridx = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Q))
gridx.point_data["gammax"] = gamma.x.array[0::2].imag
gridx.set_active_scalars("gammax")
plotter.subplot(0,0)
plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh
plotter.add_mesh(gridx, opacity=0.8, show_scalar_bar=True, show_edges=False)
plotter.add_text('Im(gammax)', 'upper_edge', color='black', font_size=8)
plotter.view_xy()
gridy = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Q))
gridy.point_data["gammay"] = gamma.x.array[1::2].imag
gridy.set_active_scalars("gammay")
plotter.subplot(0,1)
plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh
plotter.add_mesh(gridy, opacity=0.8, show_scalar_bar=True, show_edges=False)
plotter.add_text('Im(gammay)', 'upper_edge', color='black', font_size=8)
plotter.view_xy()
plotter.show()

##################################
# Create free boundary conditions (or uncomment lines below for Dirichlet)
# bcs = []
# Dirichlet test case:
boundary_facets = facet_tags.find(21) #outer boundary (tag=21)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=mesh.topology.dim-1, entities=boundary_facets)
bcs = [dolfinx.fem.dirichletbc(value=np.zeros(3, dtype=PETSc.ScalarType), dofs=boundary_dofs, V=V)]

##################################
# Define variational problem (SAFE method)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
Lx = lambda u: ufl.as_vector([u[0].dx(0), 0, 0, u[1].dx(0), u[2].dx(0), 0])
Ly = lambda u: ufl.as_vector([0, u[1].dx(1), 0, u[0].dx(1), 0, u[2].dx(1)])
Lz  = lambda u: ufl.as_vector([0, 0, u[2], 0, u[0], u[1]])
gammax = gamma[0]
gammay = gamma[1]
k1 = (ufl.inner(C*(gammay/gammax*Lx(u)+Ly(u)), Lx(v)) + ufl.inner(C*(Lx(u)+gammax/gammay*Ly(u)), Ly(v))) * ufl.dx
k1_form = dolfinx.fem.form(k1)
k2 = (gammay*ufl.inner(C*Lz(u), Lx(v))+gammax*ufl.inner(C*Lz(u), Ly(v))) * ufl.dx
k2_form = dolfinx.fem.form(k2)
k3 = ufl.inner(C*Lz(u), Lz(v)) * gammax * gammay * ufl.dx
k3_form = dolfinx.fem.form(k3)
m = rho * ufl.inner(u, v) * gammax * gammay * ufl.dx
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
wg.evp.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE) #here, preferred to TARGET_IMAGINARY
wg.solve(nev, target=target)

##################################
# Plot dispersion curves\
# Results are to be compared with Fig. 8 of Treyssede, Journal of Computational Physics 314 (2016), 341–354
ax = wg.plot_energy_velocity()
ax.set_xlim([0, 2*np.pi*26e3/3260])
ax.set_ylim([0, 6000/3260])
ax = wg.plot_attenuation()
ax.set_xlim([0, 2*np.pi*26e3/3260])
ax.set_ylim([0, 2000/(8.686*1000)])
plt.show()
#wg.plot_spectrum(index=0)
#plt.show()

##################################
# Mode shape visualization
ik, imode = 50, 1 #parameter index, mode index to visualize
vec = wg.eigenvectors[ik].getColumnVector(imode)
u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
u_grid["u"] = np.array(vec).real.reshape(int(np.array(vec).size/V.element.value_shape), int(V.element.value_shape)) #V.element.value_shape is equal to 3
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh
u_plotter.add_mesh(u_grid.warp_by_vector("u", factor=2.0), opacity=0.8, show_scalar_bar=True, show_edges=False) #do not show edges of higher order elements with pyvista
u_plotter.show_axes()
u_plotter.show()
