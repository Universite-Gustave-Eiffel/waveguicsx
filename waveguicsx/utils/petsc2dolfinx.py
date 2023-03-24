def petsc2dolfinx(V,vector):
    """
    Return a PETSc vector under the form of dolfinx solution object.
    The size of vector must be compatible with V.
    
    Example:
    V = dolfinx.fem.FunctionSpace(mesh, element)
    wg = waveguicsx.Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
    wg.set_parameters(wavenumber=np.arange(0.1, 2, 0.1))
    wg.solve(nev)
    vector = wg.eigenvectors[ik].getColumnVector(imode)
    uh = petsc2dolfinx(V,vector)
    
    Parameters
    ----------
    V: the dolfinx function space
    vector: a PETSc vector
    
    Returns
    -------
    uh: the dolfinx object with vector stored in the attribute uh.vector
    """
    
    uh = dolfinx.fem.Function(V) #initialization
    assert (vector.getSize()==uh.vector.getSize()), "Vector size not compatible with V!"
    uh.vector.setValues(range(uh.vector.getSize()), vector)
    return uh
