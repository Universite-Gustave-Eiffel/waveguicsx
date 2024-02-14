##################################
# Scattering by free end in 2D (visco-)elastic waveguide example (reflection of Lamb modes by free end)\
# The cross-section is a 1D line with free boundary conditions on its boundaries\
# The inhomogeneous part, including the free end, is a 2D rectangle\
# material: viscoelastic steel\
# The problem is solved using a hybrid FE-SAFE method, with transparent boundary condition in the inlet cross-section\
# The inlet eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers)\
# Viscoelastic loss can be included by introducing imaginary parts (negative) to wave celerities\
# Results are to be compared with Figs. ??? of paper: ????

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
pyvista.set_jupyter_backend("static"); pyvista.start_xvfb() #try also: "static", "pythreejs", "ipyvtklink"...

##################################
# Input parameters
h = 1 #plate thickness (m)
L = 1 #plate length (m) for the inhomogeneous part
le = h/4 #finite element characteristic length (m)
E, nu, rho = 2.0e+11, 0.25, 7800 #Young's modulus (Pa), Poisson ratio, density (kg/m3)
kappas, kappal = 0.008*0, 0.003*0 #shear and longitudinal bulk wave attenuations (Np/wavelength)
omega = 2*np.pi*np.linspace(10, 1000, num=10) #2*np.sqrt(3)*np.linspace(1.48, 1.60, num=100) #angular frequencies (rad/s)
nev = 30 #number of eigenvalues
cs, cl = np.sqrt(E/rho/(2*(1+nu))), np.sqrt(E/rho*(1-nu)/((1+nu)*(1-2*nu)))
free_end = False #True: only one tbc (at one end), False: two tbcs (at both ends)

##################################
# Re-scaling
L0 = h #characteristic length
T0 = h/cs #characteristic time
M0 = rho*h**3 #characteristic mass
h, L, le = h/L0, L/L0, le/L0
rho, cs, cl = rho/M0*L0**3, cs/L0*T0, cl/L0*T0
omega = omega*T0
cs, cl = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi) #complex celerities (core)

##################################
# Create FE mesh (2D)
gmsh.initialize()
gmsh.model.geo.addPoint(0, 0, 0, le, 1)
gmsh.model.geo.addPoint(h, 0, 0, le, 2)
gmsh.model.geo.addPoint(h, L, 0, le, 3)
gmsh.model.geo.addPoint(0, L, 0, le, 4)
tbc1 = gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
tbc2 = gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
domain = gmsh.model.geo.addPlaneSurface([1])
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, [tbc1], tag=0) #tag=0: 1st transparent boundary condition, 'inlet'
gmsh.model.addPhysicalGroup(1, [tbc2], tag=1) #tag=1: 2nd transprent boundary condition, 'outlet'
gmsh.model.addPhysicalGroup(2, [domain], tag=2) #tag=2: the whole FE domain (2D mesh)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
#gmsh.model.mesh.recombine() #recombine into quadrilaterals
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
tbc_tags = [0] if free_end else [0, 1]

##################################
# Finite element space (2D)
element = ufl.VectorElement("CG", "triangle", 2, 2) #Lagrange element, triangle, quadratic "P2", 2D vector
V = dolfinx.fem.FunctionSpace(mesh, element)

##################################
# Visualize FE mesh with pyvista
Vmesh = dolfinx.fem.FunctionSpace(mesh, ufl.FiniteElement("CG", "triangle", 1)) #cell of order 1 is properly handled with pyvista
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Vmesh))
grid.cell_data["Marker"] = cell_tags.values
grid.set_active_scalars("Marker")
plotter.add_mesh(grid, show_edges=True)
grid_nodes = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V)) #add higher-order nodes
plotter.add_mesh(grid_nodes, style='points', render_points_as_spheres=True, point_size=2)
plotter.view_xy()
plotter.show()

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

###############################################################################
# Define the 2D variational formulation
dx = ufl.Measure("dx", domain=mesh)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
L = lambda u: ufl.as_vector([u[0].dx(0), u[1].dx(1), u[1].dx(0)+u[0].dx(1)])
k = ufl.inner(C*L(u), L(v)) * dx
k_form = dolfinx.fem.form(k)
m = rho*ufl.inner(u, v) * dx
mass_form = dolfinx.fem.form(m)
M = dolfinx.fem.petsc.assemble_matrix(mass_form, bcs=bcs, diagonal=0.0)
M.assemble()
K = dolfinx.fem.petsc.assemble_matrix(k_form, bcs=bcs)
K.assemble()
dofs = dolfinx.fem.Function(V)
dofs.vector.setValues(range(M.size[0]), range(M.size[0])) #global dof vector

###############################################################################
# Determination of sign of outward normal of transparent boundaries (for check)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
n = ufl.FacetNormal(mesh)
ey = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 1))) #ey: unit vector along the waveguide axis
tbc_normal = []
for tag in tbc_tags:
    normal = ufl.inner(n, ey) * ds(tag) #integration of the y-component of normal along the boundary
    normal = dolfinx.fem.form(normal)
    normal = np.sign(dolfinx.fem.assemble_scalar(normal)).real
    tbc_normal.append(normal.astype('int8'))
print(tbc_normal)

##################################
# For each transparent boundary condition:
# - extract the FE mesh (1D)
# - define the variational form (SAFE formulation)
# - assemble matrices
# - locate tbc dofs in the global matrix
Ms, K0, K1, K2, tbc_dofs = [], [], [], [], []
for tag in tbc_tags:  
    #Extract the 1D mesh
    safe_mesh = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim-1, facet_tags.indices[facet_tags.values==tag])[0]
    #Define the variational form
    safe_element = ufl.VectorElement("CG", safe_mesh.ufl_cell(), 2, 2)
    safe_V = dolfinx.fem.FunctionSpace(safe_mesh, safe_element)
    dx = ufl.Measure("dx", domain=safe_mesh)
    u = ufl.TrialFunction(safe_V)
    v = ufl.TestFunction(safe_V)
    Lxy = lambda u: ufl.as_vector([u[0].dx(0), 0, u[1].dx(0)])
    Lz = lambda u: ufl.as_vector([0, u[1], u[0]])
    k0 = ufl.inner(C*Lxy(u), Lxy(v)) * dx
    k0_form = dolfinx.fem.form(k0)
    k1 = ufl.inner(C*Lz(u), Lxy(v)) * dx
    k1_form = dolfinx.fem.form(k1)
    k2 = ufl.inner(C*Lz(u), Lz(v)) * dx
    k2_form = dolfinx.fem.form(k2)
    m = rho*ufl.inner(u, v) * dx
    mass_form = dolfinx.fem.form(m)
    Ms.append(dolfinx.fem.petsc.assemble_matrix(mass_form, bcs=bcs, diagonal=0.0))
    Ms[-1].assemble()
    K0.append(dolfinx.fem.petsc.assemble_matrix(k0_form, bcs=bcs))
    K0[-1].assemble()
    K1.append(dolfinx.fem.petsc.assemble_matrix(k1_form, bcs=bcs, diagonal=0.0))
    K1[-1].assemble()
    K2.append(dolfinx.fem.petsc.assemble_matrix(k2_form, bcs=bcs, diagonal=0.0))
    K2[-1].assemble()
    #Locate tbc dofs in the global matrix (trick using interpolate)
    tbc_dofs.append(dolfinx.fem.Function(safe_V))
    tbc_dofs[-1].interpolate(dofs)
    tbc_dofs[-1] = tbc_dofs[-1].vector.array.real.round().astype('int32')    

##################################
# Scattering problem
from scattering import Scattering
#Initialization
tbcs =  [('waveguide0', -tbc_dofs[0])] if free_end else [('waveguide0', -tbc_dofs[0]), ('waveguide1', +tbc_dofs[1])] #-1 because tbc_normal[0]=-1
ws = Scattering(MPI.COMM_WORLD, M, K, 0*M, tbcs)
#Solve waveguide problem of 1st tbc
print(f'\nTransparent boundary condition 0\n')
ws.waveguide0 = Waveguide(MPI.COMM_WORLD, Ms[0], K0[0], K1[0], K2[0])
ws.waveguide0.set_parameters(omega=omega)
ws.waveguide0.solve(nev)
ws.waveguide0.compute_traveling_direction()
#Solve waveguide problem of 2nd tbc
if not free_end:
    print(f'\nTransparent boundary condition 1\n')
    ws.waveguide1 = Waveguide(MPI.COMM_WORLD, Ms[1], K0[1], K1[1], K2[1])
    ws.waveguide1.set_parameters(omega=omega)
    ws.waveguide1.solve(nev)
    ws.waveguide1.compute_traveling_direction()
print(f'\n')
#Set a single ingoing mode
mode = ws.waveguide0.track_mode(0, 0, threshold=0.98, plot=True)
ws.set_ingoing_mode('waveguide0', mode)
#Solve the scattering problem
ws.set_parameters()
print(f'KSP solver type: {ws.ksp.getType()}')
ws.solve()
#Plot energy balance
ws.plot_energy_balance()

##################################
# TEMP (checks)...
# print(ws.waveguide0.eigenvalues[0])
if True:#not free_end:
    length = 1
    print('SAFE1')
    imode = np.nonzero(ws.waveguide0.traveling_direction[-1]==+1)[0].astype('int32')
    [print(np.sum(ws.waveguide0.coefficient[-1][imode] * ws.waveguide0.eigenvectors[-1][dof,imode] * np.exp(1j*ws.waveguide0.eigenvalues[-1][imode]*length))) for dof in range(tbc_dofs[0].size)]
    print('SAFE2')
    imode = np.nonzero(ws.waveguide1.traveling_direction[-1]==+1)[0].astype('int32')
    [print(np.sum(ws.waveguide1.coefficient[-1][imode] * ws.waveguide1.eigenvectors[-1][dof,imode])) for dof in range(tbc_dofs[1].size)]

##################################
# Plotting a displacement solution
iomega = 0
vec = ws.displacement[iomega]
grid_interp = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Vmesh))
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
#grid["u"] = np.array(vec).real.reshape(int(np.array(vec).size/V.element.value_shape), int(V.element.value_shape)) #V.element.value_shape is equal to 2
grid["uy"] = np.array(vec).real[1::2]
grid_interp = grid_interp.interpolate(grid) #interpolation onto a finite element mesh of order 1 to plot both the FE mesh and the results with pyvista
plotter = pyvista.Plotter()
plotter.add_mesh(grid_interp)
#plotter.add_mesh(grid_interp.warp_by_vector("u", factor=0.1), show_scalar_bar=False, show_edges=True)
#plotter.show_axes()
plotter.view_xy()
plotter.show()

""

