#####################################################################
# waveguicsx, a python library for solving complex waveguide problems
# 
# Copyright (C) 2023-2024  Fabien Treyssede
# 
# This file is part of waveguicsx.
# 
# waveguicsx is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# waveguicsx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with waveguicsx. If not, see <https://www.gnu.org/licenses/>.
# 
# Contact: fabien.treyssede@univ-eiffel.fr
#####################################################################


from petsc4py import PETSc
from waveguicsx.waveguide import Waveguide

import numpy as np
import time
import matplotlib.pyplot as plt


class Scattering:
    """
    A class for solving scattering problems by local inhomogeneities in complex waveguides based on PETSc.
    
    The full documentation is entirely defined in the `scattering.py' module.

    The following matrix problem is considered: (K-omega**2*M-1j*omega*C)*U=F.
    This kind of problem typically typically stems from a finite element (FE) model of a small portion of waveguide
    including a local inhomogeneity (e.g. defects). The cross-section extremities of the truncated FE model are then
    handled as transparent boundary conditions (BCs) to reproduce semi-infinite waveguides.
    The so-obtained scattering problem is solved repeatedly for each frequency.
    
    This class enables to deal with scattering in complex waveguides, two-dimensional (e.g. plates) or three-dimensional
    (arbitrarily shaped cross-section), inhomogeneous in the transverse directions, anisotropic. Complex-valued problems
    can be handled including the effects of non-propagating modes (evanescent, inhomogeneous), viscoelastic loss (complex
    material properties) or perfectly matched layers (PML) to simulate buried waveguides.
    
    !!!The loops over the angular frequency can be parallelized, as shown in some tutorials (using mpi4py)!!!!
    
    Transparent BCs are Waveguide objects, which must have been solved prior to the scattering
    problem solution, yielding the following object attributes: omega, eigenvalues, eigenvectors, eigenforces and traveling
    direction (see waveguide.py module for details). Transparent BCs are localized by their degrees of
    freedom in the global vector U. This means that the local degree of freedom i of a given eigenvector/eigenforce
    stored in a Waveguide object is located at the global degree of freedom dofs[i] of the FE model.
    
    The user must supply the following inputs:
    - K, M, C, the global FE matrices (stiffness, mass and viscous damping)
    - F and F_spectrum, the global FE vector of internal excitation  (i.e. sources inside the FE model), if any,
      and its spectrum
    - tbcs, a list of pairs (name, dofs) which characterize the transparent BCs, where name is a string specifying the
      attribute name of a given transparent BC (this attribute will be a Waveguide object) and dofs is a numpy array
      of the global degrees of freedom for this transparent BC.
    - the ingoing mode coefficients, specified by the attribute coefficient in each transparent BC
    Important convention: if the waveguide axis of a transparent BC is oriented outward (i.e. outside the FE box), dofs are
    positive, but if oriented inward (i.e. inside the FE box), a negative sign has to be assigned to dofs by the user.
    
    The solution to the scattering problem yields:
    - the displacement U of the FE model for each angular frequency omega
    - the outgoing modal coefficients of every transparent BC and for each omega
    - the energy balance post-processed for each angular frequency
      which enables to check the error due to the modal truncature introduced in the transparent BCs.
    See Attributes below for more details.
    
    
    Example::
    
        ..................TO DO...................   
    
    
    Attributes
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI communicator (parallel processing)
    M, K, C : petsc4py.PETSc.Mat
        FE matrices
    F : petsc4py.PETSc.Vec
        FE force vector (internal excitation)
    F_spectrum : numpy.ndarray
        the modulation spectrum of F (size must be the same as omega)
    tbc : list of pairs (name, dofs)
        name is a string corresponding to the desired attribute name of a tbc (Waveguide object),
        dofs is a numpy array of the degrees of freedom of the tbc (positive if outward
        negative if inward)
    omega : numpy.ndarray
        the angular frequency range, specified by the user in tbcs (Waveguide objects)
    ksp: KSP object (PETSc object)
        solver parameters
    displacement : list of PETSc vectors
        for each angular frequency, the FE displacement vector U (solution of the scattering problem)
    energy_balance : list of 1d numpy array
        for each angular frequency, energy_balance[i] gives the following three-component array,
        [Pin-i*omega/2*U^H*F, Ptot, -i*omega*U^H*D*U], where P=-i*omega/2*U^H*T:
        - the term Pin-i*omega/2*U^H*F represents the input power, supplied by ingoing modes and internal
          forces (this term should have a negative real part)
        - the term Ptot is the complex power flow of the sum of ingoing and ougoing modes
        - the term -i*omega*U^H*D*U is related to the kinetic, potential and dissipated energies in the volume
          (the dissipated energy, defined as positive, is equal to the real part of this term divided by -2*omega).
        - a perfect energy balance is achived if Ptot = -i*omega*U^H*D*U
    name.coefficient : list of numpy.ndarray, where name is a Waveguide object associated with a given transparent BC
        this transparent BC attribute stores modal coefficients at each frequency:
        - the coefficients of ingoing modes are considered as inputs, specified by the user (excitation in the scattering problem)
        - the coefficients of outgoing modes are considered as initially unknown, solution of the scattering problem
        Any non-zero outgoing amplitudes specified in name.coefficient prior to solving the scattering problem
        will hence be discarded and replaced with the scattering solution.
        If the attribute coefficient is empty prior to scattering (no specified ingoing modes), zero ingoing amplitudes
        will be considered by default.
    
    
    Methods
    -------
    __init__(comm:'_MPI.Comm', M:PETSc.Mat, K:PETSc.Mat, C:PETSc.Mat, tbcs:list):
        Constructor, initialization of scattering problem
    set_ingoing_mode(tbc_name, mode, spectrum=None):
        For a given tbc, specified by the string tbc_name, set the coefficient of a single mode to 1.
        mode is a list of indices identifying the mode position at each frequency.
    set_internal_excitation(F, F_spectrum=None):
        Set the internal excitation vector F and its spectrum F_spectrum
    set_parameters():
        Set default parameters of KSP solver (stored in attribute ksp)
    solve():
        Solve the scattering problem repeatedly for the angular frequency range, solutions are stored as attributes
        (names: displacement, energy_balance)
    plot_energy_balance():
        Plot the three terms of energy_balance (complex modulus) at each frequency index for checking modal tbc truncature        
    """
    def __init__(self, comm:'_MPI.Comm', M:PETSc.Mat, K:PETSc.Mat, C:PETSc.Mat, tbcs:list):
        """
        Constructor
        
        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator (parallel processing)
        M, K, C : petsc4py.PETSc.Mat
            FE matrices
        tbcs : list of pairs (name, dofs)
            name is a string corresponding to the desired attribute name of a tbc (Waveguide object),
            dofs is a numpy array of the degrees of freedom of the tbc (positive if outward
            negative if inward)
        """
        self.comm = comm
        self.M = M
        self.K = K
        self.C = C
        self.tbcs = tbcs
        for tbc in tbcs:
            setattr(self, tbc[0], None)
        
        # Set the default values for the internal attributes used in this class
        self.omega: None
        self.F = None
        self.F_spectrum = None
        self.ksp = None
        self.energy_balance = []
        self.displacement = []
        
        # Print the number of degrees of freedom
        print(f'Total number of degrees of freedom: {self.M.size[0]}')
    
    def set_ingoing_mode(self, tbc_name, mode, spectrum=None):
        """
        For a given tbc, specified by the string tbc_name, set the coefficient of a single mode to 1. The coefficients of
        all other modes are set to 0. mode is a list of indices identifying the mode position at each frequency
        (see also method track_mode() of class Waveguide).
        Please ensure that this mode is an ingoing mode (non-zero outgoing amplitudes will be discarded).
        When specified, spectrum is a vector of length omega (numpy.array) used to modulate the coefficient in terms of
        frequency (default: 1 for all frequencies).
        """
        size = getattr(self, tbc_name).omega.size
        if spectrum is None:
            spectrum = np.ones(size)
        if len(spectrum) != size:
            raise NotImplementedError('The length of spectrum must be equal to the length of omega')
        #Initialization of modal coefficients to zero
        getattr(self, tbc_name).coefficient = [np.zeros(getattr(self, tbc_name).eigenvalues[i].size).astype('complex')
                                               for i in range(size)]
        #Set mode index to 1
        for i in range(size):
            if mode[i]>=0:
                getattr(self, tbc_name).coefficient[i][mode[i]] = spectrum[i]
    
    def set_internal_excitation(self, F, F_spectrum=None):
        """
        Set the internal excitation vector F (petsc4py.PETSc.Vec).
        When specified, F_spectrum is a vector of length omega (numpy.array) used to modulate F in terms of
        frequency (default: 1 for all frequencies).
        F and F_spectrum are stored as attributes.
        """
        self.F = F
        self.F_spectrum = spectrum
    
    def set_parameters(self):
        """
        Set default parameters of the KSP solver () stored into the attribute ksp.
        After calling this method, various PETSc parameters can be set by changing the attribute ksp manually.
        """       
        #Default setup
        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setFromOptions()
        #self.ksp.setType(PETSc.KSP.Type.GMRES) #GMRES is default, try also: BCGS, CGS...
    
    def solve(self):
        """
        Solve the scattering problem, i.e. the linear system repeatedly for the angular frequency range.
        The solutions are stored in the attributes displacement and energy_balance.
        The FE problem D*U=F is transformed into:
        Bout^T*(D*Bout-Tout)*Uout = Bout^T*(Tin-D*Bin)*Uin
        where:
        - D is the dynamic stiffness matrix (D=K-omega**2*M-1j*omega*C)
        - Bin and Bout (resp. Tin and Tout) are bases containing ingoing and outgoing modal
          displacements (resp. forces) so that: U=Bin*Uin+Bout*Uout, F=Tin*Uin+Tout*Uout
        - Uin contains internal forces and known ingoing amplitudes (zero by default, if not specified)
        - Uout is the solution containing outgoing modal amplitudes and internal dofs
        """
        
        #Checks        
        for index, tbc in enumerate(self.tbcs):
            if getattr(self, tbc[0]).__class__.__name__ == Waveguide:
                raise NotImplementedError(f'The attribute {tbc[0]} is not an instance of class Waveguide!')
            if len(getattr(self, tbc[0]).eigenvalues)==0:
                raise NotImplementedError(f'The eigenvalue problem of {tbc[0]} has not been solved!')
            if len(getattr(self, tbc[0]).eigenforces)==0:
                raise NotImplementedError(f'The eigenforces of {tbc[0]} have not been computed!')
            if len(getattr(self, tbc[0]).traveling_direction)==0:
                raise NotImplementedError(f'The traveling direction of {tbc[0]} has not been computed!')
            if index==0:
                self.omega = getattr(self, tbc[0]).omega
            else:
                if not np.allclose(getattr(self, tbc[0]).omega, self.omega):
                    raise NotImplementedError(f'The angular frequencies of {tbc[0]} are different from those of {self.tbcs[0][0]}!')
            if len(getattr(self, tbc[0]).coefficient)==0: #create zero arrays if no coefficient has been computed
                getattr(self, tbc[0]).coefficient = [np.zeros(eigenvalues.size).astype('complex') for eigenvalues in getattr(self, tbc[0]).eigenvalues]
        if self.F is None: #create zero vector F by default
            self.F = self.M.createVecRight()
        if self.F_spectrum is None: #create vector of 1 by default
            self.F_spectrum = np.ones(self.omega.size)
        if len(self.F_spectrum) != self.omega.size:
            raise NotImplementedError('The length of spectrum of F must be equal to the length of omega')
        
        print(f'Scattering problem ({len(self.omega)} iterations)')
        
        #Loop on frequency
        for i, omega in enumerate(self.omega):
            
            start = time.perf_counter()
            
            #Pointers
            tbc_dofs = np.concatenate([np.abs(tbc[1]) for tbc in self.tbcs])
            internal_dofs = np.setdiff1d(range(self.M.getSize()[0]), tbc_dofs).astype('int32')
            ingoing_col_pointer = [internal_dofs.size]
            outgoing_col_pointer = [internal_dofs.size]
            for tbc in self.tbcs:
                normal_sign = -1 if any(tbc[1]<0) else +1 #outward normal sign along the waveguide axis
                traveling_direction = getattr(self, tbc[0]).traveling_direction[i]    
                ingoing_col_pointer.extend([ingoing_col_pointer[-1] + np.nonzero(traveling_direction==-normal_sign)[0].size])
                outgoing_col_pointer.extend([outgoing_col_pointer[-1] + np.nonzero(traveling_direction==+normal_sign)[0].size])
            ingoing_ncol = ingoing_col_pointer[-1]
            outgoing_ncol = outgoing_col_pointer[-1]
            
            #Initialization of global projection matrices, filled with ones or zeroes at internal dofs
            Bu_in = self._build_global_internal(internal_dofs=[], ncol=ingoing_ncol)
            Bf_in = self._build_global_internal(internal_dofs=internal_dofs, ncol=ingoing_ncol)
            Bu_out = self._build_global_internal(internal_dofs=internal_dofs, ncol=outgoing_ncol)
            Bf_out = self._build_global_internal(internal_dofs=[], ncol=outgoing_ncol)
            
            #Assemble projection matrices
            for j, tbc in enumerate(self.tbcs):
                #Outgoing matrices
                Bu_temp, Bf_temp = self._build_global_modal(i=i, tbc=tbc, direction_str='outgoing', ncol=outgoing_ncol,
                                                       col_pointer=outgoing_col_pointer[j])
                Bu_out = Bu_out + Bu_temp
                Bf_out = Bf_out + Bf_temp
                #Ingoing matrices
                Bu_temp, Bf_temp = self._build_global_modal(i=i, tbc=tbc, direction_str='ingoing', ncol=ingoing_ncol,
                                                       col_pointer=ingoing_col_pointer[j])
                Bu_in = Bu_in + Bu_temp
                Bf_in = Bf_in + Bf_temp
            
            #Assemble global excitation vector (internal force and ingoing modal amplitudes)
            U_in = PETSc.Vec().createSeq(ingoing_ncol, comm=self.comm)
            U_in.setValues(range(internal_dofs.size), self.F.getValues(internal_dofs)*self.F_spectrum[i]) #fill with the external force vector at internal dofs
            for j, tbc in enumerate(self.tbcs):
                normal_sign = -1 if any(tbc[1]<0) else +1 #outward normal sign along the waveguide axis
                traveling_direction = getattr(self, tbc[0]).traveling_direction[i]
                imodes = np.nonzero(traveling_direction==-normal_sign)[0]
                tbc_coeff = getattr(self, tbc[0]).coefficient[i][imodes]
                U_in.setValues(range(ingoing_col_pointer[j], ingoing_col_pointer[j+1]), tbc_coeff) #fill with ingoing modal coefficients
            
            #Solve scattering system
            D = self.K - omega**2*self.M - 1j*omega*self.C
            Bu_out_transpose = Bu_out.copy().transpose()
            U_out = PETSc.Vec().createSeq(outgoing_ncol, comm=self.comm)
            self.ksp.reset() #reset is necessary because the size of A and b can change at every iteration
            self.ksp.setOperators(Bu_out_transpose*(D*Bu_out-Bf_out)) #the operator A of system Ax=b
            self.ksp.solve(Bu_out_transpose*(Bf_in-D*Bu_in)*U_in, U_out)
            
            #Back to initial dofs
            self.displacement.append(Bu_out*U_out + Bu_in*U_in)
            
            #Energy balance
            self.energy_balance.append(-1j*omega/2*np.array([(Bf_in*U_in).dot(Bu_in*U_in)+(self.F*self.F_spectrum[i]).dot(self.displacement[-1]),
                                                           (Bf_in*U_in+Bf_out*U_out).dot(Bu_in*U_in+Bu_out*U_out),
                                                           (D*self.displacement[-1]).dot(self.displacement[-1])]))
            
            #Store outgoing modal amplitudes
            for j, tbc in enumerate(self.tbcs):
                normal_sign = -1 if any(tbc[1]<0) else +1 #outward normal sign along the waveguide axis
                traveling_direction = getattr(self, tbc[0]).traveling_direction[i]
                imodes = np.nonzero(traveling_direction==+normal_sign)[0]
                getattr(self, tbc[0]).coefficient[i][imodes] = U_out.getValues(range(outgoing_col_pointer[j], outgoing_col_pointer[j+1]))

            print(f'Iteration {i}, elapsed time :{(time.perf_counter() - start):.2f}s')
        
        #Memory saving
        Bu_out_transpose.destroy()
        D.destroy()
        
    def plot_energy_balance(self):
        """
        Plot energy balance at each frequency index for checking modal tbc truncature.
        The energy balance is defined as the difference between tbc net flux and volume power
        divided by the input power (values close to zero indicate that enough modes have
        been included in the transparent BCs).
        """
        energy_balance = np.concatenate(self.energy_balance)
        fig, ax = plt.subplots(1, 1)
        energy_balance = (np.abs(energy_balance[1::3])-np.abs(energy_balance[2::3]))/np.abs(energy_balance[::3])
        ax.plot(np.abs(energy_balance[::3]), linewidth=1, color='black', label='|input power|')
        #ax.plot(np.abs(energy_balance[::3]), linewidth=1, color='black', label='|input power|')
        #ax.plot(np.abs(energy_balance[1::3]), linewidth=1, color='red', label='|tbc net flux|')
        #ax.plot(np.abs(energy_balance[2::3]), linewidth=1, color='blue', label='|volume power|')
        #ax.legend()
        ax.set_xlabel('frequency index')
        ax.set_ylabel('energy balance')
        fig.tight_layout()
        return ax
        
    def _build_global_modal(self, i, tbc, direction_str, ncol, col_pointer):
        """
        Return Bu and Bf, the global modal bases for the displacement eigenvectors and eigenforces
        of the transparent BC given by tbc=(name, dofs), at omega index i, in the direction
        given by direction_str ('ingoing' or outgoing'). For internal use.
        ncol is the total number of columns of the global projection matrix (i.e. number of internal dofs plus
        total number of ingoing/outgoing modes), col_pointer is the column index of the global projection matrix
        where to start the storage of the tbc modal basis.
        Note: this method assumes that eigenvectors and eigenforces are stored as dense PETSc matrices.
        """
        #Initialization
        name, dofs = tbc
        normal_sign = -1 if any(dofs<0) else +1 #outward normal sign along the waveguide axis
        direction = -normal_sign if direction_str=='ingoing' else +normal_sign
        dofs = np.abs(dofs)
        traveling_direction = getattr(self, name).traveling_direction[i]
        imodes = np.nonzero(traveling_direction==direction)[0]
        tbc_size, nmodes = getattr(self, name).eigenvectors[i].getSize() #transparent boundary dofs size, number of modes
        tbc_row = range(0, (tbc_size+1)*len(imodes), len(imodes)) #this line assumes dense PETSC matrix
        size = self.M.getSize()[0] #K, M, C size
        row = np.zeros(size, dtype='int32')
        row[dofs] = np.diff(tbc_row)
        #Build Bu and Bf (modal basis for displacement and force)
        Bu = PETSc.Mat().createAIJ((size, ncol), comm=self.comm)
        Bu.setPreallocationNNZ(nnz=row)
        Bf = PETSc.Mat().createAIJ((size, ncol), comm=self.comm)
        Bf.setPreallocationNNZ(nnz=row)
        col = col_pointer
        for mode in imodes:
            Bu.setValues(dofs, col, getattr(self, name).eigenvectors[i].getColumnVector(mode))
            Bf.setValues(dofs, col, normal_sign * getattr(self, name).eigenforces[i].getColumnVector(mode))
            col = col + 1
        Bu.setUp()
        Bu.assemble()
        Bf.setUp()
        Bf.assemble()
        return Bu, Bf

    def _build_global_internal(self, internal_dofs, ncol):
        """
        Build a global projection matrix filled with ones for internal dofs. If internal_dofs is empty, return
        a zero matrix. internal_dofs are dofs of the FE model excluding transparent boundary dofs.
        ncol is the total number of columns of the global projection matrix (i.e. number of internal dofs plus
        total number of ingoing/outgoing modes).
        """
        size = self.M.getSize()[0] #K, M, C size
        if len(internal_dofs)!=0:
            row = np.zeros(size)
            row[internal_dofs] = 1
            row = np.cumsum(row, dtype='int32')
            row = np.insert(row, 0, 0)
            col = range(internal_dofs.size) #internal dofs are set at the beginning of global matrix
            val = np.ones(internal_dofs.size, dtype='int32')
            B = PETSc.Mat().createAIJWithArrays((size, ncol), [row, col, val], comm=self.comm)
        else: #if internal_dofs is empty, return a zero matrix
            B = PETSc.Mat().createAIJ((size, ncol), nnz=0, comm=self.comm)
            B.assemble()
        return B
