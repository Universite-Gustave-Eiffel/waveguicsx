Tutorials
=========


0. Three-dimensional elastic bar of square cross-section
--------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square with free boundary conditions on its boundaries.
Material: viscoelastic steel.
The SAFE eigenproblem is solved with the varying parameter as the wavenumber (eigenvalues are then frequencies) or as the frequency (eigenvalues are then wavenumbers).
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
Results are compared with Euler-Bernoulli analytical solutions for the lowest wavenumber or frequency parameter.

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D.py


1. Three-dimensional elastic bar of square cross-section with parallelization
-----------------------------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square with free boundary conditions on its 1D boundaries.
Material: viscoelastic steel.
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
In this example:

* the parameter loop (here, the frequency loop) is distributed on all processes
* FE mesh and matrices are built on each local process

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D_ParallelizedLoop.py


2. Three-dimensional elastic bar of square cross-section buried into a PML external medium
------------------------------------------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square buried into a PML external elastic medium.
Material: viscoelastic steel into cement grout.
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
The PML has a parabolic profile.
Results are to be compared with Fig. 8 of paper: Treyssede, Journal of Computational Physics 314 (2016), 341–354.

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D_PML.py


3. Three-dimensional elastic bar of square cross-section buried into a PML external medium using gmsh
-----------------------------------------------------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square buried into a PML external elastic medium.
Material: viscoelastic steel into cement grout.
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
The PML has a parabolic profile.
The FE mesh is built from Gmsh.
Results are to be compared with Fig. 8 of paper: Treyssede, Journal of Computational Physics 314 (2016), 341–354.

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D_PML_Gmsh.py


4. Excitation of a three-dimensional elastic bar of circular cross-section
--------------------------------------------------------------------------

3D elastic waveguide example.
The cross-section is a 2D circle with free boundary conditions on its 1D boundaries, material: elastic steel.
The eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).
The forced response is computed for a point force at the center node ot the cross-section.
Results are to be compared with Figs. 5, 6 and 7 of paper: Treyssede, Wave Motion 87 (2019), 75-91.

.. literalinclude:: ../examples/Elastic_Waveguide_CircularBar3D_ForcedResponse_Gmsh.py


5. Excitation of a three-dimensional elastic bar of circular cross-section with parallelization
-----------------------------------------------------------------------------------------------

3D elastic waveguide example.
The cross-section is a 2D circle with free boundary conditions on its 1D boundaries, material: elastic steel.
The eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).
The forced response is computed for a point force at the center node ot the cross-section.
Results are to be compared with Figs. 5, 6 and 7 of paper: Treyssede, Wave Motion 87 (2019), 75-91.
In this example:

* the parameter loop (here, the frequency loop) is distributed on all processes
* FE mesh and matrices are built on each local process

.. literalinclude:: ../examples/Elastic_Waveguide_CircularBar3D_ForcedResponse_Gmsh_ParallelizedLoop.py


6. Time response of a two-dimensional plate excited near its first ZGV resonance
--------------------------------------------------------------------------------

2D (visco-)elastic waveguide example (Lamb modes in a plate excited near 1st ZGV resonance).
The cross-section is a 1D line with free boundary conditions on its boundaries.
Material: viscoelastic steel.
The eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).
Viscoelastic loss can be included by introducing imaginary parts (negative) to wave celerities.
Results are to be compared with Figs. 5b, 7a and 8a of paper: Treyssede and Laguerre, JASA 133 (2013), 3827-3837.
Note: the depth direction is x, the axis of propagation is z.

.. literalinclude:: ../examples/Elastic_Waveguide_Plate2D_TransientResponse.py


7. Dispersion curves of a rail 
------------------------------

3D elastic waveguide example.
The cross-section is a 2D rail profile (60E1, 60.21kg/m) with free boundary conditions on its 1D boundaries, material: elastic steel.
This eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).
The FE mesh is built from gmsh with a .geo file.

.. literalinclude:: ../examples/Rail60E1/Elastic_Waveguide_Rail_Gmsh.py


8. Reflection of Lamb modes by the free edge of a plate
-------------------------------------------------------

Scattering in 2D elastic waveguide example (reflection of Lamb modes by the free edge of a plate).
The cross-section is a 1D line with free boundary conditions on its boundaries.
The inhomogeneous part, including the free edge, is a 2D rectangle.
Material: elastic steel.
The problem is solved using FEM with transparent boundary condition (tbc) in the inlet cross-section.
The inlet eigenproblem is solved using SAFE as a function of frequency (eigenvalues are wavenumbers).
Results can be compared with Fig. 4 of paper: Karunasena et al., CMAME 125 (1995), 221-233 (see also
Gregory and Gladwell, J. of Elast. 13 (1983), 185-206).

.. literalinclude:: ../examples/Scattering_Elastic_Waveguide_Plate2D_Gmsh.py


9. Reflection and transmission of Pochhammer-Chree modes inside a cylinder
--------------------------------------------------------------------------

Scattering in 3D elastic waveguide example.
Reflection of Pochhammer-Chree modes by the free edge of a cylinder or by notch.
The cross-section is a 2D disk with free boundary conditions on its boundaries.
The inhomogeneous part, including free edge or notch, is a 3D cylinder.
Material: elastic steel.
The problem is solved using FEM with transparent boundary condition in the inlet and outlet cross-sections.
The tbc eigenproblem is solved using SAFE as a function of frequency (eigenvalues are wavenumbers)\
Results can be compared with the following papers, for free edge and notch respectively:

* Gregory and Gladwell, Q. J. Mech. Appl. Math.  42 (1989), 327–337
* Benmeddour et al., IJSS 48 (2011), 764-774.

.. literalinclude:: ../examples/Scattering_Elastic_Waveguide_Cylinder3D_Gmsh.py

