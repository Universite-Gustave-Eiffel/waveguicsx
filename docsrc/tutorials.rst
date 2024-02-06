Tutorials
=========


0. 3D elastic bar of square cross-section
-----------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square with free boundary conditions on its boundaries.
Material: viscoelastic steel.
The SAFE eigenproblem is solved with the varying parameter as the wavenumber (eigenvalues are then frequencies) or as the frequency (eigenvalues are then wavenumbers).
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
Results are compared with Euler-Bernoulli analytical solutions for the lowest wavenumber or frequency parameter.

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D.py


1. 3D elastic bar of square cross-section with parallelization
--------------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square with free boundary conditions on its 1D boundaries.
Material: viscoelastic steel.
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
In this example:

* the parameter loop (here, the frequency loop) is distributed on all processes
* FE mesh and matrices are built on each local process

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D_ParallelizedLoop.py


2. 3D elastic bar of square cross-section buried into a PML external medium
---------------------------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square buried into a PML external elastic medium.
Material: viscoelastic steel into cement grout.
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
The PML has a parabolic profile.
Results are to be compared with Fig. 8 of paper: Treyssede, Journal of Computational Physics 314 (2016), 341–354.

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D_PML.py


3. 3D elastic bar of square cross-section buried into a PML external medium using gmsh
--------------------------------------------------------------------------------------

3D (visco-)elastic waveguide example.
The cross-section is a 2D square buried into a PML external elastic medium.
Material: viscoelastic steel into cement grout.
Viscoelastic loss is included by introducing imaginary parts (negative) to wave celerities.
The PML has a parabolic profile.
The FE mesh is built from Gmsh.
Results are to be compared with Fig. 8 of paper: Treyssede, Journal of Computational Physics 314 (2016), 341–354.

.. literalinclude:: ../examples/Elastic_Waveguide_SquareBar3D_PML_Gmsh.py


4. Excitation of a 3D elastic bar of circular cross-section
-----------------------------------------------------------

3D elastic waveguide example.
The cross-section is a 2D circle with free boundary conditions on its 1D boundaries, material: elastic steel.
The eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).
The forced response is computed for a point force at the center node ot the cross-section.
Results are to be compared with Figs. 5, 6 and 7 of paper: Treyssede, Wave Motion 87 (2019), 75-91.

.. literalinclude:: ../examples/Elastic_Waveguide_CircularBar3D_ForcedResponse_Gmsh.py


5. Excitation of a 3D elastic bar of circular cross-section with parallelization
--------------------------------------------------------------------------------

3D elastic waveguide example.
The cross-section is a 2D circle with free boundary conditions on its 1D boundaries, material: elastic steel.
The eigenproblem is solved with the varying parameter as the frequency (eigenvalues are then wavenumbers).
The forced response is computed for a point force at the center node ot the cross-section.
Results are to be compared with Figs. 5, 6 and 7 of paper: Treyssede, Wave Motion 87 (2019), 75-91.
In this example:

* the parameter loop (here, the frequency loop) is distributed on all processes
* FE mesh and matrices are built on each local process

.. literalinclude:: ../examples/Elastic_Waveguide_CircularBar3D_ForcedResponse_Gmsh_ParallelizedLoop.py


6. Time response of a plate excited near its first ZGV resonance
----------------------------------------------------------------

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

