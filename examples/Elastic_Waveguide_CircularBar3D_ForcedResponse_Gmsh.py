##################################
# 3D elastic waveguide example\
# The cross-section is a 2D circle with free boundary conditions on its 1D boundaries, material: elastic steel\
# The waveguide FE formulation (SAFE) leads to the following eigenvalue problem:\
# $(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$\
# This eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).\
# The forced response is computed for a point force at the center node ot the cross-section.\
# Results are to be compared with Figs. 5, 6 and 7 of paper: Treyssede, Wave Motion 87 (2019), 75-91.

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
#For proper use with a jupyter notebook, uncomment the following line:
#pyvista.set_jupyter_backend("none"); pyvista.start_xvfb() #try also: "static", "pythreejs", "ipyvtklink"...

##################################
# Input parameters
a = 2.7e-3 #cross-section radius (m)
le = a/8 #finite element characteristic length (m)
rho, cs, cl = 7800, 3296, 5963 #density (kg/m3), shear and longitudinal wave celerities (m/s)
kappas, kappal = 0*0.008, 0*0.003 #shear and longitudinal bulk wave attenuations (Np/wavelength)
omega = np.arange(0.1, 10.1, 0.1)*cs/a #angular frequencies (rad/s)
nev = 60 #number of eigenvalues

##################################
# Re-scaling
L0 = a #characteristic length
T0 = a/cs #characteristic time
M0 = rho*a**3#characteristic mass
a, le = a/L0, le/L0
rho, cs, cl = rho/M0*L0**3, cs/L0*T0, cl/L0*T0
omega = omega*T0
cs, cl = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi) #complex celerities

##################################
# Create mesh from Gmsh and finite elements (six-node triangles with three dofs per node for the three components of displacement)
gmsh.initialize()
# Core
origin = gmsh.model.geo.addPoint(+0, 0, 0, le, 1)
gmsh.model.geo.addPoint(+a, 0, 0, le, 2)
gmsh.model.geo.addPoint(0, +a, 0, le, 3)
gmsh.model.geo.addPoint(-a, 0, 0, le, 4)
gmsh.model.geo.addPoint(0, -a, 0, le, 5)
gmsh.model.geo.addCircleArc(2, 1, 3, 1)
gmsh.model.geo.addCircleArc(3, 1, 4, 2)
gmsh.model.geo.addCircleArc(4, 1, 5, 3)
gmsh.model.geo.addCircleArc(5, 1, 2, 4)
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
disk = gmsh.model.geo.addPlaneSurface([1])
# Physical groups
gmsh.model.geo.synchronize() #the CAD entities must be synchronized with the Gmsh model
gmsh.model.addPhysicalGroup(2, [disk], tag=1) #2: for 2D (surface)
#gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], tag=11) #1: for 1D (line)
# Generate mesh
gmsh.model.mesh.embed(0, [origin], 2, disk) #ensure node points at the origin
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
# Create Material properties (isotropic)
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
                     [0,0,0,0,0,C66]])
C = isotropic_law(rho, cs, cl)
C = dolfinx.fem.Constant(mesh, PETSc.ScalarType(C))

##################################
# Create free boundary conditions
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
K0 = dolfinx.fem.petsc.assemble_matrix(k1_form, bcs=bcs)
K0.assemble()
K1 = dolfinx.fem.petsc.assemble_matrix(k2_form, bcs=bcs, diagonal=0.0)
K1.assemble()
K2 = dolfinx.fem.petsc.assemble_matrix(k3_form, bcs=bcs, diagonal=0.0)
K2.assemble()

##################################
# Solve the eigenproblem with SLEPc (the parameter is omega, the eigenvalue is k)
wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
wg.set_parameters(omega=omega, two_sided=False)
wg.evp.setTolerances(tol=1e-10, max_it=20)
wg.solve(nev=nev, target=0) #access to components with: wg.eigenvalues[ik][imode], wg.eigenvectors[ik][idof,imode]

##################################
# Plot dispersion curves\
# Results are to be compared with Fig. 5 of Treyssede, Wave Motion 87 (2019), 75-91
wg.plot_energy_velocity(direction=+1)
plt.show()

##################################
# Excitation force definition (point force)
dof_coords = V.tabulate_dof_coordinates()
x0 = np.array([0, 0, 0]) #desired coordinate of point force
dof = int(np.argmin(np.linalg.norm(dof_coords - x0, axis=1))) #find nearest dof
print(f'Point force coordinates (nearest dof):  {(dof_coords[dof,:])}') #check
F0 = M.createVecRight()
dof = dof*3 + 2 #orientation along z
#dof = dof - 2 #uncomment this line for an orientation along x instead
F0[dof] = 1
##Uncomment lines below for distributed excitation
#body_force = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 1))) #body force of unit amplitude along z
#traction = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 0))) #zero traction
#ds = ufl.Measure("ds", domain=mesh)
#f = ufl.inner(body_force, v) * ufl.dx + ufl.inner(traction, v) * ds
#f_form = dolfinx.fem.form(f)
#F = dolfinx.fem.petsc.assemble_vector(f_form)
#F.assemble()

##################################
# Computation of excitability and forced response\
# Results are to be compared with Figs. 6 and 7 of Treyssede, Wave Motion 87 (2019), 75-91
wg.compute_response_coefficient(F=F0, dof=dof)
wg.plot_coefficient()
ax = wg.plot_excitability()
ax.set_yscale('log')
ax.set_ylim(1e-3,0.5e+1)
frequency, response, axs = wg.compute_response(dof=dof, z=[5], spectrum=None, plot=True) #spectrum=excitation.spectrum
axs[0].set_yscale('log')
axs[0].set_ylim(1e-2,1e+1)
axs[0].get_lines()[0].set_color("black")
plt.close()

##################################
# Mode shape visualization
ik, imode = 50, 1 #parameter index, mode index to visualize
vec = wg.eigenvectors[ik].getColumnVector(imode)
u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
u_grid["u"] = np.array(vec).real.reshape(int(np.array(vec).size/V.element.value_shape), int(V.element.value_shape)) #V.element.value_shape is equal to 3
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid, style="wireframe", color="k") #FE mesh
u_plotter.add_mesh(u_grid.warp_by_vector("u", factor=0.5), opacity=0.8, show_scalar_bar=True, show_edges=False) #do not show edges of higher order elements with pyvista
u_plotter.show_axes()
u_plotter.show()


""

