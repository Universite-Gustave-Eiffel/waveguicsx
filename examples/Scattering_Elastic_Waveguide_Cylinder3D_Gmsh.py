# This file is a tutorial for waveguicsx (*), whose inputs are:
# - the matrices K, M, C and the internal excitation vector F
# - the matrices K0, K1, K2 and Ms for each transparent BC.
# In the tutorial, these matrices are finite element matrices generated by FEnicSX (**).
#  (*) waveguicsx is a python library for solving complex waveguide problems
#      Copyright (C) 2023-2024  Fabien Treyssede
#      waveguicsx is free software distributed under the GNU General Public License
#      (https://github.com/treyssede/waveguicsx)
# (**) FEniCSx is an open-source computing platform for solving partial differential equations
#      distributed under the GNU Lesser General Public License (https://fenicsproject.org/)

##################################
# Scattering in 3D elastic waveguide example
# Reflection of Pochhammer-Chree modes by the free edge of a cylinder or by notch\
# The cross-section is a 2D disk with free boundary conditions on its boundaries\
# The inhomogeneous part, including free edge or notch, is a 3D cylinder\
# material: elastic steel\
# The problem is solved using FEM with transparent boundary condition in the inlet cross-section\
# The inlet eigenproblem is solved using SAFE as a function of frequency (eigenvalues are wavenumbers)\
# Results can be compared with the following papers, for free edge and notch respectively:
# - Gregory and Gladwell, Q. J. Mech. Appl. Math.  42 (1989), 327–337
# - Benmeddour et al., IJSS 48 (2011), 764-774.

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
from waveguicsx.scattering import Scattering
#For proper use with a jupyter notebook, uncomment the following line:
pyvista.set_jupyter_backend("static"); pyvista.start_xvfb() #try also: "static", "pythreejs", "ipyvtklink"...

##################################
# Input parameters
a = 1 #cylinder radius (m)
L = 2*a #cylinder length (m) for the inhomogeneous part
le = a/2 #finite element characteristic length (m)
E, nu, rho = 2.0e+11, 0.25, 7800 #Young's modulus (Pa), Poisson ratio, density (kg/m3)
kappas, kappal = 0.008*0, 0.003*0 #shear and longitudinal bulk wave attenuations (Np/wavelength)
cs, cl = np.sqrt(E/rho/(2*(1+nu))), np.sqrt(E/rho*(1-nu)/((1+nu)*(1-2*nu)))
#omega = np.linspace(1.8, 2.4, num=50)*cl/a #angular frequencies (rad/s), for comparison with Gregory and Gladwell
omega = np.linspace(0.1, 3.0, num=30)*cl/a #for comparison with Benmeddour et al.
nev = 40 #number of eigenvalues
free_end = False #True: only one tbc (at one end), False: two tbcs (at both ends, note: check that an incident mode is outgoing properly)
notch_depth, notch_thickness = 1*a, 0.05*a #case without notch: set notch_depth to 0

##################################
# Re-scaling
L0 = a #characteristic length
T0 = a/cs #characteristic time
M0 = rho*a**3 #characteristic mass
a, L, le = a/L0, L/L0, le/L0
rho, cs, cl = rho/M0*L0**3, cs/L0*T0, cl/L0*T0
omega = omega*T0
cs, cl = cs/(1+1j*kappas/2/np.pi), cl/(1+1j*kappal/2/np.pi) #complex celerities (core)

##################################
# Create FE mesh (2D)
gmsh.initialize()
gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, a, 0)
if notch_depth!=0: #notch case
    gmsh.model.occ.addBox(a, a, L/2, -2*a, -notch_depth, notch_thickness, 1)
    gmsh.model.occ.cut([(3, 0)], [(3, 1)], 2)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
volumes = gmsh.model.getEntities(dim=3)
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], 0) #volume tag: 0
surfaces = gmsh.model.occ.getEntities(dim=2)
for s in surfaces:
    center_of_mass = gmsh.model.occ.getCenterOfMass(s[0], s[1]) #useful to identify surface numbers of tbcs
    if np.isclose(center_of_mass[2], 0):
        gmsh.model.addPhysicalGroup(s[0], [s[1]], 0) #inlet tbc tag: 0
    if np.isclose(center_of_mass[2], L):
        gmsh.model.addPhysicalGroup(s[0], [s[1]], 1) #outlet tbc tag: 1
gmsh.model.mesh.setOrder(2)
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
gmsh.finalize()
tbc_tags = [0] if free_end else [0, 1]

##################################
# Finite element space (3D)
element = ufl.VectorElement("CG", "tetrahedron", 2, 3) #Lagrange element, tetrahedron, quadratic, 3D vector
V = dolfinx.fem.FunctionSpace(mesh, element)

##################################
# Visualize FE mesh with pyvista
Vmesh = dolfinx.fem.FunctionSpace(mesh, ufl.FiniteElement("CG", "tetrahedron", 1)) #cell of order 1 is properly handled with pyvista
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(Vmesh))
grid.cell_data["Marker"] = cell_tags.values
grid.set_active_scalars("Marker")
plotter.add_mesh(grid, show_edges=True)
grid_nodes = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V)) #add higher-order nodes
plotter.add_mesh(grid_nodes, style='points', render_points_as_spheres=True, point_size=2)
plotter.show()

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

###############################################################################
# Define the 2D variational formulation
dx = ufl.Measure("dx", domain=mesh)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
L = lambda u: ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2),
                            u[0].dx(1)+u[1].dx(0), u[0].dx(2)+u[2].dx(0), u[1].dx(2)+u[2].dx(1)])
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
ez = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0, 1))) #ez: unit vector along the waveguide axis
tbc_normal = []
for tag in tbc_tags:
    normal = ufl.inner(n, ez) * ds(tag) #integration of the z-component of normal along the boundary
    normal = dolfinx.fem.form(normal)
    normal = np.sign(dolfinx.fem.assemble_scalar(normal)).real
    tbc_normal.append(normal.astype('int8'))
print(tbc_normal)

##################################
# For each transparent boundary condition:
# - extract the FE mesh (2D)
# - define the variational form (SAFE formulation)
# - assemble matrices
# - locate tbc dofs in the global matrix
Ms, K0, K1, K2, tbc_dofs = [], [], [], [], []
for tag in tbc_tags:  
    #Extract the 2D mesh
    safe_mesh = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim-1, facet_tags.indices[facet_tags.values==tag])[0]
    #Define the variational form
    safe_element = ufl.VectorElement("CG", safe_mesh.ufl_cell(), 2, 3)
    safe_V = dolfinx.fem.FunctionSpace(safe_mesh, safe_element)
    dx = ufl.Measure("dx", domain=safe_mesh)
    u = ufl.TrialFunction(safe_V)
    v = ufl.TestFunction(safe_V)
    Lxy = lambda u: ufl.as_vector([u[0].dx(0), u[1].dx(1), 0, u[0].dx(1)+u[1].dx(0), u[2].dx(0), u[2].dx(1)])
    Lz  = lambda u: ufl.as_vector([0, 0, u[2], 0, u[0], u[1]])
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
# Scattering initialization
tbcs =  [('waveguide0', -tbc_dofs[0])] if free_end else [('waveguide0', -tbc_dofs[0]), ('waveguide1', +tbc_dofs[1])] #-1 because tbc_normal[0]=-1
ws = Scattering(MPI.COMM_WORLD, M, K, 0*M, tbcs)
#Solve waveguide problem of 1st tbc
print(f'\nTransparent boundary condition 0\n')
ws.waveguide0 = Waveguide(MPI.COMM_WORLD, Ms[0], K0[0], K1[0], K2[0])
ws.waveguide0.set_parameters(omega=omega)
ws.waveguide0.solve(nev)
ws.waveguide0.compute_traveling_direction()
ws.waveguide0.compute_poynting_normalization()
#Solve waveguide problem of 2nd tbc
if not free_end:
    print(f'\nTransparent boundary condition 1\n')
    ws.waveguide1 = Waveguide(MPI.COMM_WORLD, Ms[1], K0[1], K1[1], K2[1])
    ws.waveguide1.set_parameters(omega=omega)
    ws.waveguide1.solve(nev)
    ws.waveguide1.compute_traveling_direction()
    ws.waveguide1.compute_poynting_normalization()
print(f'\n')

##################################
# Solving scattering problem
# index = np.argmin(np.abs(ws.waveguide0.eigenvalues[0]-2.59)) #2.59 is the L(0,1) wavenumber value at angular frequency 3.12 (roughly), for comparison with Gregory and Gladwell
index = np.argmin(np.abs(ws.waveguide0.eigenvalues[0]-0.11)) #0.11 is the L(0,1) wavenumber value at angular frequency 0.17 (roughly), for comparison with Benmeddour et al.
mode = ws.waveguide0.track_mode(0, index, threshold=0.98, plot=True) #track mode over the whole frequency range
ws.set_ingoing_mode('waveguide0', mode) #set mode as a single ingoing mode, coeff is 1 (its power is also 1 thansk to poynting normalization)
ws.set_parameters(solver='direct')
ws.solve()
ws.plot_energy_balance() #checking tbc modal truncature
plt.show()
#ws.ksp.view()

###############################################################################
# Plot modal coefficients w.r.t angular frequency\
# Results can be compared with paper: Gregory and Gladwell, Q. J. Mech. Appl. Math.  42 (1989), 327–337
ws.waveguide0.compute_complex_power()
sc = ws.waveguide0.plot(y=('complex_power', lambda x:np.abs(np.real(x))), direction=-1)
sc.axes.set_ylim(0, 1.2)
sc.axes.set_ylabel('Reflected power')
#ws.waveguide0.plot_complex_power() #plot both real and imaginary part (signed)
#ws.waveguide0.plot_energy_velocity(c=('coefficient', np.abs)) #plot ve curves colored by |q|
if not free_end:
    ws.waveguide1.compute_complex_power()
    sc = ws.waveguide1.plot(y=('complex_power', lambda x:np.abs(np.real(x))), direction=+1)
    sc.axes.set_ylim(0, 1.2)
    sc.axes.set_ylabel('Transmitted power')

###############################################################################
# Example of plot for a single mode
# index = np.argmin(np.abs(ws.waveguide0.eigenvalues[0]+2.59)) #-2.59 is the opposite L(0,1) wavenumber value at angular frequency 3.12 (roughly)
index = np.argmin(np.abs(ws.waveguide0.eigenvalues[0]+0.11)) #-0.11 is the opposite L(0,1) wavenumber value at angular frequency 0.17 (roughly)
mode2 = ws.waveguide0.track_mode(0, index, threshold=0.98, plot=False) #track mode (0, 0))
sc = ws.waveguide0.plot_complex_power(mode=mode2)
sc[0].axes.legend()
plt.show()

##################################
# Example of plot for displacement field at a given angular frequency
i = 20 #angular frequency index
vec = ws.displacement[i]
grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
#grid["ux"] = np.array(vec).real[0::3] #x-component of displacement
#grid["uy"] = np.array(vec).real[1::3] #y-component of displacement
grid["uz"] = np.array(vec).real[2::3] #z-component of displacement
plotter = pyvista.Plotter()
plotter.add_mesh(grid)
plotter.show_axes()
plotter.show()