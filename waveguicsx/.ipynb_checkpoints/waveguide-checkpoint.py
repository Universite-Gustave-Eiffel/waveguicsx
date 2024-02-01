#####################################################################
# waveguicsx, a python library for solving complex waveguide problems
# 
# Copyright (C) 2023  Fabien Treyssede
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


from typing import Union, List
from petsc4py import PETSc
from slepc4py import SLEPc

import matplotlib.pyplot as plt
import numpy as np
import time


class Waveguide:
    """
    A class for solving complex waveguide problems based on SLEPc eigensolver.
    
    The full documentation is entirely defined in the `waveguide.py' module.

    The following matrix problem is considered: (K0-omega**2*M + 1j*k*(K1-K1^T) + k**2*K2)*U=F.
    This kind of problem typically stems from the so-called SAFE (Semi-Analytical Finite Element) method.
    
    The class enables to deal with complex waveguides, two-dimensional (e.g. plates) or three-dimensional (arbitrarily
    shaped cross-section), inhomogeneous in the transverse directions, anisotropic. Complex-valued problems can be handled
    including the effects of non-propagating modes (evanescent, inhomogeneous), viscoelastic loss (complex material
    properties) or perfectly matched layers (PML) to simulate buried waveguides.
    
    The free response (F=0) is an eigenvalue problem, solved iteratively by varying the parameter
    which can be the angular frequency omega or the wavenumber k. In the former case, the eigenvalue is k,
    while in the latter case, the eigenvalue is omega^2. The loops over the parameter (angular frequency or wavenumber)
    can be parallelized, as shown in some tutorials (using mpi4py).
    
    Various modal properties (energy velocity, group velocity, excitability...) can be post-processed as a function of the
    frequency and plotted as dispersion curves.
    
    The forced reponse (F is not 0) is solved in the frequency domain by expanding the solution as a sum of
    eigenmodes using biorthogonality relationship, leading to very fast computations of excited wavefields.
    
    
    Example::
    
        from waveguicsx.waveguide import Waveguide
        
        # Definition of the excitation signal (here, a toneburst)
        excitation = Signal()
        excitation.toneburst(fs=8/(2*np.pi), T=49.75*(2*np.pi), fc=2/(2*np.pi), n=5) 
        excitation.plot()
        excitation.plot_spectrum()
        omega = 2*np.pi*excitation.frequency  #omega = np.linspace(0.02, 4, 200)
        
        # Initialization of waveguide
        wg = Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
        wg.set_parameters(omega=omega)
        
        # Solution of eigenvalue problem (iteration over the parameter omega)
        wg.solve(nev=50, target=0) #access to components with: wg.eigenvalues[iomega][imode], wg.eigenvectors[iomega][idof,imode]
        
        # Plot dispersion curves
        wg.plot()
        wg.compute_energy_velocity()
        wg.plot_energy_velocity()
        
        # Computation of modal coefficients and excitabilities
        wg.compute_response_coefficient(F=F, dof=dof)
        wg.plot_coefficient()
        wg.plot_excitability()
        
        # Forced response in the frequency domain, due to a toneburst excitation, at degree of freedom dof and axial coordinates z
        frequency, response = wg.compute_response(dof=dof, z=[50, 100, 150, 200], spectrum=excitation.spectrum)
        
        # Transient response
        response = Signal(frequency=frequency, spectrum=response)
        response.plot_spectrum()
        response.ifft()
        response.plot()
        plt.show()
    
    
    Attributes
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI communicator (parallel processing)
    M, K0, K1, K2 : petsc4py.PETSc.Mat
        SAFE matrices
    F : petsc4py.PETSc.Vec
        SAFE excitation vector
    problem_type : str
        problem_type is "omega" if the varying parameter is omega, "wavenumber" if this is k
    two_sided : bool
        if True, left eigenvectors will be also computed (otherwise, only right eigenvectors are computed)
    target: complex number or user-defined function of the parameter
        target around which eigenpairs are looked for (see method solve)
    omega or wavenumber : numpy.ndarray
        the parameter range specified by the user (see method set_parameters)
    evp : PEP or EPS instance (SLEPc object)
        eigensolver parameters
    eigenvalues : list of numpy arrays
        list of wavenumbers or angular frequencies,
        access to components with eigenvalues[ip][imode] (ip: parameter index, imode: mode index)
    eigenvectors : list of PETSc matrices
        list of mode shapes,
        access to components with eigenvectors[ik][idof,imode] (ip: parameter index, imode: mode index, idof: dof index)
        or eigenvectors[ik].getColumnVector(imode)
    eigenforces : list of PETSc matrices
        list of eigenforces (acces to components: see eigenvectors)
    opposite_going : list of numpy arrays
        list of opposite-going mode (acces to components: see eigenvectors)
    energy_velocity : list of numpy arrays
        list of energy velocity (access to component: see eigenvalues)
    group_velocity : list of numpy arrays
        list of group velocity (access to component: see eigenvalues)
    traveling_direction : list of numpy arrays
        list of traveling_direction (access to component: see eigenvalues)
    pml_ratio : list of numpy arrays
        list of pml ratio, used for filtering out PML modes (access to component: see eigenvalues)
    coefficient : list of numpy arrays
        list of response coefficient to excitation vector F (access to component: see eigenvalues)
    excitability : list of numpy arrays
        list of excitability to excitation vector F (access to component: see eigenvalues)
    plot_scaler : dictionnary
        dictionnary containing the scaling factors of various modal properties, useful to plot results in a dimensional form
    
    Methods
    -------
    __init__(comm:'_MPI.Comm', M:PETSc.Mat, K0:PETSc.Mat, K1:PETSc.Mat, K2:PETSc.Mat):
        Constructor, initialization of waveguide
    set_parameters(omega=None, wavenumber=None, two_sided=False):
        Set problem type (problem_type), the parameter range (omega or wavenumber) as well as default parameters of SLEPc eigensolver (evp);
        set two_sided to True to compute left eigenvectors also (left eigenvectors are the opposite-going modes)
    solve(nev=1, target=0):
        Solve the eigenvalue problem repeatedly for the parameter range, solutions are stored as attributes (names: eigenvalues,
        eigenvectors)
    compute_eigenforce():
        Compute the eigenforces for the whole parameter range and store them as an attribute (name: eigenforces)
    compute_poynting_normalization():
        Normalization of eigenvectors and eigenforces, so that U'=U/sqrt(|P|), where P is the normal component of complex Poynting vector
    compute_opposite_going(plot=False):
        Compute opposite-going mode pairs based on on wavenumber and biorthogonality for the whole parameter range and store
        them as attributes (name: opposite_going), set plot to True to visualize the biorthogonality values of detected pairs        
    compute_energy_velocity():
        Compute the energy velocities for the whole parameter range and store them as an attribute (name: energy_velocity)
    compute_group_velocity():
        Compute the group velocities for the whole parameter range and store them as an attribute (name: energy_velocity)
    compute_traveling_direction():
        Compute the traveling directions for the whole parameter range and store them as an attribute (name: traveling_direction)
    compute_pml_ratio():
        Compute the pml ratios for the whole parameter range and store them as an attribute (name: pml_ratio)
    compute_response_coefficient(F, spectrum=None, wavenumber_function=None, dof=None):
        Compute the response coefficients due to excitation vector F for the whole parameter range and store them as
        an attribute (name: coefficient)
    compute_response(dof, z, spectrum=None, wavenumber_function=None, plot=False):
        Compute the response at the degree of freedom dof and the axial coordinate z for the whole frequency range
    plot(direction=None, pml_threshold=None, ax=None, color="k",  marker="o", markersize=2, linestyle="", **kwargs):
        Plot dispersion curves Re(omega) vs. Re(wavenumber) using matplotlib
    plot_phase_velocity(direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot phase velocity dispersion curves, vp=Re(omega)/Re(wavenumber) vs. Re(omega)
    plot_attenuation(direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot attenuation dispersion curves, Im(wavenumber) vs. Re(omega) if omega is the parameter,
        or Im(omega) vs. Re(omega) if wavenumber is the parameter
    plot_energy_velocity(direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot energy velocity dispersion curves, ve vs. Re(omega)
    plot_group_velocity(direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot group velocity dispersion curves, vg vs. Re(omega)
    plot_coefficient(direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot response coefficients as a function of frequency, |q| vs. Re(omega)
    plot_excitability(direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot excitability as a function of frequency, |e| vs. Re(omega)
    plot_spectrum(index=0, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot the spectrum, Im(eigenvalues) vs. Re(eigenvalues), for the parameter index specified by the user
    set_plot_scaler(length=1, time=1, mass=1, dim=3):
        Define the characteristic length, time, mass, as well as dim, and calculate the scaling factors of modal
        properties, which are stored in the attribute name plot_scaler (useful to visualize plots in a dimensional form)
    """
    def __init__(self, comm:'_MPI.Comm', M:PETSc.Mat, K0:PETSc.Mat, K1:PETSc.Mat, K2:PETSc.Mat):
        """
        Constructor
        
        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator (parallel processing)
        M, K0, K1, K2 : petsc4py.PETSc.Mat
            SAFE matrices
        """
        self.comm = comm
        self.M = M
        self.K0 = K0
        self.K1 = K1
        self.K2 = K2
        self.F = None

        # Set the default values for the internal attributes used in this class
        self.problem_type: str = ""  # "wavenumber" or "omega"
        self.omega: Union[np.ndarray, None] = None
        self.wavenumber: Union[np.ndarray, None] = None
        self.two_sided = None
        self.target = None
        self.evp: Union[SLEPc.PEP, SLEPc.EPS, None] = None
        self.eigenvalues: list = []
        self.eigenvectors: list = []
        self.eigenforces: list = []
        self.opposite_going: list = []
        self.energy_velocity: list = []
        self.group_velocity: list = []
        self.traveling_direction: list = []
        self.pml_ratio: list = []
        self.coefficient: list = []
        self.excitability: list = []
        self.plot_scaler = dict.fromkeys(['omega','wavenumber','energy_velocity','group_velocity','pml_ratio',
                                          'eigenvalues','excitability',
                                          'eigenvectors','eigenforces','coefficient',
                                          'frequency','attenuation','phase_velocity'], 1)
        self._poynting_normalization = None
        self._biorthogonality_factor: list = []
        
        # Print the number of degrees of freedom
        print(f'Total number of degrees of freedom: {self.M.size[0]}')

    def set_parameters(self, omega: Union[np.ndarray, None]=None, wavenumber:Union[np.ndarray, None]=None, two_sided=False):
        """
        Set the parameter range (omega or wavenumber) as well as default parameters of the SLEPc eigensolver (evp).
        The user must specify the parameter omega or wavenumber, but not both.
        This method generates the attributes omega (or wavenumber) and evp.
        After calling this method, various SLEPc parameters can be set by changing the attribute evp manually.
        Set two_sided=True for solving left eigenvectors also.
        
        Parameters
        ----------
        omega or wavenumber : numpy.ndarray
            the parameter range specified by the user
        two_sided : bool
            False if left eigenvectiors are not needed, True if they must be solved also
        """
        if len(self.eigenvalues)!=0:
            print('Eigenvalue problem already solved (re-initialize the Waveguide object to solve a new eigenproblem)')
            return
        if not (wavenumber is None) ^ (omega is None):
            raise NotImplementedError('Please specify omega or wavenumber (and not both)')
        
        # The parameter is the frequency omega, the eigenvalue is the wavenumber k
        if wavenumber is None:
            self.problem_type = "omega"
            self.omega = np.array(omega)
            if not two_sided: #left eigenvectors not required
                # Setup the SLEPc solver for the quadratic eigenvalue problem
                self.evp = SLEPc.PEP()
                self.evp.create(comm=self.comm)
                self.evp.setProblemType(SLEPc.PEP.ProblemType.GENERAL) #note: for the undamped case, HERMITIAN is possible with QARNOLDI and TOAR but surprisingly not faster
                self.evp.setType(SLEPc.PEP.Type.LINEAR) #note: the computational speed of LINEAR, QARNOLDI and TOAR seems to be almost identical
                self.evp.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_IMAGINARY)
            else: #left eigenvectors required by user
                # Setup the SLEPc solver for the quadratic eigenvalue problem linearized externally! (SLEPc.EPS used, setTwoSided is not available in SLEPc.PEP)
                self.evp = SLEPc.EPS()
                self.evp.create(comm=self.comm)
                self.evp.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #note: GHEP (generalized Hermitian) is surprinsingly a little bit slower...
                self.evp.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #note: ARNOLDI also works although slightly slower
                self.evp.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY)
                self.evp.setTwoSided(two_sided)

        # The parameter is the frequency omega, the eigenvalue is the wavenumber k
        elif omega is None:
            if two_sided:
                raise NotImplementedError('two_sided has been set to True: not implemented in case wavenumber is the parameter')
            self.problem_type = "wavenumber"
            self.wavenumber = np.array(wavenumber)
            # Setup the SLEPc solver for the generalized eigenvalue problem
            self.evp = SLEPc.EPS()
            self.evp.create(comm=self.comm)
            self.evp.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #note: GHEP (generalized Hermitian) is surprinsingly a little bit slower...
            self.evp.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #note: ARNOLDI also works although slightly slower
            self.evp.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            #self.evp.setTwoSided(two_sided) #two_sided not implemented: left for future work if necessary
        
        # Common setup
        self.two_sided = two_sided
        self.evp.setTolerances(tol=1e-8, max_it=20)
        ST = self.evp.getST()
        ST.setType(SLEPc.ST.Type.SINVERT)
        #ST.setShift(1e-6) #do not set shift here: it will be set automatically to target later (see solve method)
        self.evp.setST(ST)
        #self.evp.st.ksp.setType('preonly') #'preonly' is the default, other choice could be 'gmres', 'bcgs'...
        #self.evp.st.ksp.pc.setType('lu') #'lu' is the default, other choice could be 'bjacobi'...
        self.evp.setFromOptions()

    def solve(self, nev=1, target=0):
        """
        Solve the dispersion problem, i.e. the eigenvalue problem repeatedly for the parameter range (omega or wavenumber).
        The solutions are stored in the attributes eigenvalues and eigenvectors.
        If two_sided is True, left eigensolutions are also solved.
        
        Note: left eigensolutions correspond to opposite-going modes and are hence added to the right eigensolutions
        (i.e. in eigenvalues and eigenvectors) after removing any possible duplicates.
        
        Parameters
        ----------
        nev : int
            number of eigenpairs requested
        target : complex number or user-defined function of the parameter, optional (default: 0)
            target around which eigenpairs are looked for
            a small shift might sometimes prevent errors (e.g. zero pivot with dirichlet bc)
        """
        if len(self.eigenvalues)!=0:
            print('Eigenvalue problem already solved (re-initialize the Waveguide object to solve a new eigenproblem)')
            return
        self.target = target
        if self.target==0 and self.two_sided:
            raise NotImplementedError('Setting two_sided to True is useless here, please set two_sided to False (target has been set to zero, so that both positive and negative-going modes will be computed) ')
        
        # Eigensolver setup
        self.evp.setDimensions(nev=nev)
        if isinstance(target, (float, int, complex)): #redefine target as a constant function if target is given as a number
            target_constant = target
            target = lambda parameter_value: target_constant
        if self.problem_type == "omega" and self.two_sided: #build Zero and Id matrices
            Zero = PETSc.Mat().createAIJ(self.M.getSize(), comm=self.comm)
            Zero.setUp()
            Zero.assemble()
            Id = PETSc.Mat().createAIJ(self.M.getSize(), comm=self.comm)
            Id.setUp()
            Id.setDiagonal(self.M.createVecRight()+1)
            Id.assemble()
        
        # Loop over the parameter
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        parameters = {"omega": self.omega, "wavenumber": self.wavenumber}[self.problem_type]
        print(f'Waveguide parameter: {self.problem_type} ({len(parameters)} iterations)')
        for i, parameter_value in enumerate(parameters):
            start = time.perf_counter()
            self.evp.setTarget(target(parameter_value))
            if self.problem_type=="wavenumber":
                 self.evp.setOperators(self.K0 + 1j*parameter_value*(self.K1-K1T) + parameter_value**2*self.K2, self.M)
            elif self.problem_type == "omega":
                if not self.two_sided: #left eigenvectors not required -> PEP class is used
                    self.evp.setOperators([self.K0-parameter_value**2*self.M, 1j*(self.K1-K1T), self.K2])
                else: #left eigenvectors are required -> linearize the quadratic evp and use EPS class (PEP class is not possible)
                    coeff = 1 #self.K2.norm(norm_type=PETSc.NormType.FROBENIUS) #NORM_1, FROBENIUS (same as NORM_2 for vectors), INFINITY
                    self.evp.setOperators(self._build_block_matrix(-(self.K0-parameter_value**2*self.M), -1j*(self.K1-K1T), Zero, coeff*Id),
                                          self._build_block_matrix(Zero, self.K2, coeff*Id, Zero))
                    #Note: the operators below enable to get the eigenforces but increase computation time -> discarded...
                    #self.evp.setOperators(self._build_block_matrix(self.K0-parameter_value**2*self.M, Zero, -K1T, Id),
                    #                      self._build_block_matrix(-1j*self.K1, 1j*Id, 1j*self.K2, Zero))
            self.evp.solve()
            #self.evp.errorView()
            #self.evp.valuesView()
            eigenvalues, eigenvectors = self._get_eigenpairs(two_sided=self.two_sided)
            self.eigenvalues.append(eigenvalues)
            self.eigenvectors.append(eigenvectors)
            print(f'Iteration {i}, elapsed time :{(time.perf_counter() - start):.2f}s')
            #self.evp.setInitialSpace(self.eigenvectors[-1]) #self.evp.setLeftInitialSpace(....) #try to use current modal basis to compute next, but may be only the first eigenvector...
        self._poynting_normalization = False
        #print('\n---- SLEPc setup (based on last iteration) ----\n')
        #self.evp.view()
        print('')
        
        # Memory saving
        K1T.destroy()
        self.evp.destroy()

    def compute_eigenforces(self):
        """ Post-process the eigenforces F=(K1^T+1j*k*K2)*U for every mode in the whole parameter range"""
        if len(self.eigenforces)==len(self.eigenvalues):
            print('Eigenforces already computed')
            return
        start = time.perf_counter()
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        for i, eigenvectors in enumerate(self.eigenvectors):
            wavenumber = self._concatenate('wavenumber', i=i) #repeat parameter as many times as the number of eigenvalues
            self.eigenforces.append(K1T*eigenvectors+1j*self.K2*eigenvectors*self._diag(wavenumber))
        K1T.destroy()
        print(f'Computation of eigenforces, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_poynting_normalization(self):
        """
        Post-process the normalization of eigenvectors and eigenforces, so that U'=U/sqrt(|P|),
        where P is the normal component of complex Poynting vector (P=-1j*omega/2*U^H*F).
        After normalization, every mode is such that |P|=1 and the attribute _poynting_normalization is set to True.
        Normalization is not mandatory but, when applied, has to be done before any response coefficient computation.
        """
        if len(self.coefficient)!=0: #response already computed
            raise NotImplementedError('Normalization has to be applied before response coefficient computation')
        if self._poynting_normalization:
            print('Poynting normalization of eigenvectors already computed')
            return
        if len(self.eigenforces)==0: #compute the eigenforces if not yet computed      
            self.compute_eigenforces()
        start = time.perf_counter()
        index = range(self.eigenvectors[0].getSize()[0])
        for i in range(len(self.eigenvalues)):
            #repeat parameter as many times as the number of eigenvalues
            omega = self._concatenate('omega', i=i)
            #Normalization
            normalization = []
            for mode in range(self.eigenvalues[i].size):
                U = self.eigenvectors[i].getColumnVector(mode)
                F = self.eigenforces[i].getColumnVector(mode)
                normalization = np.append(normalization, 1/np.sqrt(np.abs(-1j*omega[mode]/2*F.dot(U))))
            self.eigenvectors[i] = self.eigenvectors[i]*self._diag(normalization)
            self.eigenforces[i] = self.eigenforces[i]*self._diag(normalization)
        self._poynting_normalization = True
        print(f'Computation of Poynting normalization, elapsed time : {(time.perf_counter() - start):.2f}s')
        #print(np.array([-1j*omega[mode]/2*np.vdot(self.eigenvectors[i][:,mode], self.eigenforces[i][:,mode]) for mode in range(self.eigenvalues[i].size)])) #check for last iteration

    def compute_energy_velocity(self):
        """
        Post-process the energy velocity ve=Re(P)/Re(E) for every mode in the whole parameter range, where P is the
        normal component of complex Poynting vector and E is the total energy (cross-section time-averaged)
        """
        if len(self.energy_velocity)==len(self.eigenvalues):
            print('Energy velocity already computed')
            return
        
        # Compute the eigenforces if not yet computed
        if len(self.eigenforces)==0:
            self.compute_eigenforces()
        
        # Energy velocity, integration on the whole domain
        start = time.perf_counter()
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        for i, eigenvectors in enumerate(self.eigenvectors):
            #repeat parameter as many times as the number of eigenvalues
            wavenumber, omega = self._concatenate('wavenumber', 'omega', i=i)
            #time averaged kinetic energy
            E = 0.25*np.abs(omega**2)*np.real(self._dot_eigenvectors(i, self.M*eigenvectors)) 
            #add time averaged potential energy
            E = E + 0.25*np.real(self._dot_eigenvectors(i, self.K0*eigenvectors + 1j*self.K1*eigenvectors*self._diag(wavenumber)
                                                        -1j*K1T*eigenvectors*self._diag(wavenumber.conjugate()) + self.K2*eigenvectors*self._diag(np.abs(wavenumber)**2)))
            #time averaged complex Poynting vector (normal component)
            Pn = -1j*omega/2*self._dot_eigenvectors(i, self.eigenforces[i])
            #cross-section and time averaged energy velocity
            self.energy_velocity.append(np.real(Pn)/E)
        K1T.destroy()
        print(f'Computation of energy velocity, elapsed time : {(time.perf_counter() - start):.2f}s')
        
        # Warning for pml problems (integration restricted on the core is currently not possible)
        dofs_pml = np.iscomplex(self.M.getDiagonal()[:])
        if any(dofs_pml):
            print("Warning: the energy velocity is currently integrated on the whole domain including PML region")
        ## Future works: a possible trick to restrict the integration on physical dofs
        #dofs_pml = np.iscomplex(M.getDiagonal()[:]) #problem: if not stuck to the core, can include part of the exterior domain
        #Mat = M.copy(); Mat.zeroRowsColumns(dofs_pml, diag=0) #or: eigenvectors.zeroRows(dofs_pml, diag=0)

    def compute_opposite_going(self, plot=False):
        """
        Post-process the pairing of opposite-going modes, based on wavenumber and biorthogonality criteria, and store them
        as a an attribute (name: opposite_going, -1 value for unpaired modes).
        Compute their biorthogonality normalization factors, Um^T*F-m - U-m^T*Fm, where m and -m denote opposite-going
        modes, for the whole parameter range and store them as an attribute (name: _biorthogonality_factor).
        If plot is set to True, the biorthogonality factors found by the algorithm are plotted in magnitude as a function
        of frequency, allowing visual check that there is no values close to zero (a factor close to zero probably means
        a lack of biorthogonality).
        
        Notes:
        
        - when an unpaired mode is found, the value -1 is stored in opposite_going (and NaN value in _biorthogonality_factor),
          meaning that this mode will be discarded in the computation of group velocity, traveling direction, coefficient and
          excitability (NaN values stored)
        - if modes with lack of biorthogonality or two many unpaired modes occur, try to recompute the eigenproblem by
          increasing the accuracy (e.g. reducing the tolerance)
        - lack of biorthogonality may be also due to multiple modes (*); in this case, try to use an unstructured mesh instead
        - if two_sided is True, lack of biorthogonolity may occur for specific target: try another target (e.g. add a small
          imaginary part)
           
        (*) e.g. flexural modes in a cylinder with structured mesh
        """
        tol1 = 1e-4 #tolerance for relative difference between opposite wavenumbers (wavenumber criterion)
        tol2_rel = 100 #minimum ratio between the first two candidate biorthogonality factors (biorthogonality criterion)
        tol2_abs = 1e-3 #minimum biorthogonality factor to keep a given opposite pair (biorthogonality criterion)
        if self.problem_type=='wavenumber' or (self.target!=0 and not self.two_sided):
            raise NotImplementedError('Computation of biorthogonality factor is not possible: opposite-going modes cannot be paired for this kind of problem (check that the problem is of omega type and that target has been set to 0)')
        if len(self.opposite_going)==len(self.eigenvalues):
            print('Opposite-going pairs already computed')
            return
        if len(self.eigenforces)==0: #compute the eigenforces if not yet computed      
            self.compute_eigenforces()
        if not self._poynting_normalization:
            self.compute_poynting_normalization() #this normalization matters for a proper use of tol2_abs, because it ensures that the biorthogonality factor x omega/4 is equal to 1 for pairs of pure propagating modes
        start = time.perf_counter()
        for i, eigenvalues in enumerate(self.eigenvalues):
            #Loop over half of complex plane
            upper_half = np.nonzero(np.logical_and(np.angle(eigenvalues)>=-np.pi/4, np.angle(eigenvalues)<3*np.pi/4))[0]
            lower_half = np.setdiff1d(range(eigenvalues.size), upper_half, assume_unique=True)
            opposite_going = np.zeros(eigenvalues.size, dtype=int) - 1
            biorthogonality_factor = np.zeros(eigenvalues.size, dtype=complex) + (1+1j)*np.NaN
            for mode in upper_half:
                #First criterion: based on wavenumber
                criterion1 = np.abs((eigenvalues[mode]+eigenvalues[lower_half])/eigenvalues[mode])
                candidates = np.array(lower_half[np.nonzero(criterion1<=tol1)]) #candidates of lower half
                if len(candidates)!=0: #candidates found
                    #If only one candidate found: add a second candidate to allow the test of second criterion
                    if len(candidates)==1:
                        temp = eigenvalues[candidates] #store initial value
                        eigenvalues[candidates]= np.inf #trick to discard the first candidate
                        candidates2 = lower_half[np.argmin(np.abs(eigenvalues[mode]+eigenvalues[lower_half]))] #find the second candidate
                        eigenvalues[candidates] = temp #back to initial
                        candidates = np.append(candidates, candidates2)
                    #Second criterion: based on biorthogonality
                    biorthogonality_test = []
                    for c in candidates:
                        biorthogonality_test = np.append(biorthogonality_test,
                                         self.eigenforces[i].getColumnVector(c).tDot(self.eigenvectors[i].getColumnVector(mode))
                                       - self.eigenvectors[i].getColumnVector(c).tDot(self.eigenforces[i].getColumnVector(mode)))
                    #OLD: biorthogonality_test = self.eigenforces[i][:,candidates.tolist()].T @ self.eigenvectors[i][:,mode] - self.eigenvectors[i][:,candidates.tolist()].T @ self.eigenforces[i][:,mode]
                    criterion2 = np.abs(biorthogonality_test*self.omega[i]/4)
                    order = np.argsort(criterion2) #sort by ascending order
                    if criterion2[order[-1]]<tol2_rel*criterion2[order[-2]]: #relative criterion
                        raise NotImplementedError(f'Iteration {i}: Lack of biorthogonality between mode {mode} and modes [{candidates[order[-2:]]}], with respective biorthogonality factor [{criterion2[order[-2:]]}]!')
                    if criterion2[order[-1]]>tol2_abs: #absolute value criterion
                        opposite = candidates[order[-1]]
                        opposite_going[[mode, opposite]] = [opposite, mode]
                        biorthogonality_factor[[mode, opposite]] = [biorthogonality_test[order[-1]], -biorthogonality_test[order[-1]]]
            #Final check
            unique, counts = np.unique(opposite_going[opposite_going>=0], return_counts=True)
            if any(counts>1): #test for duplicate modes
                print(unique, counts)
                raise NotImplementedError(f'Iteration {i}: duplicate modes found by the pairing process!')
            unpaired = np.setdiff1d(range(eigenvalues.size), opposite_going)
            if len(unpaired)!=0: #test for unpaired modes
                print(f'Iteration {i}: unpaired modes found, index={unpaired}')
                if len(self.traveling_direction)==len(self.eigenvalues): #the traveling direction has already been computed but using energy velocity
                    self.traveling_direction[i][unpaired] = np.NaN
            #Store pairs and biorthogonality factors
            self.opposite_going.append(opposite_going)
            self._biorthogonality_factor.append(biorthogonality_factor)
        print(f'Computation of pairs of opposite-going modes, elapsed time : {(time.perf_counter() - start):.2f}s')
        #Plot the biorthogonality factors as a function of frequency
        if plot:
            omega = np.repeat(self.omega.real, [len(egv) for egv in self._biorthogonality_factor])
            biorthogonality_factor = np.concatenate(self._biorthogonality_factor)
            fig, ax = plt.subplots(1, 1)
            ax.plot(omega, np.abs(biorthogonality_factor*omega/4), marker="o", markersize=2, linestyle="", color="k")
            ax.set_xlabel('Re(omega)')
            ax.set_ylabel('|biorthogonality factor|')
            ax.set_yscale('log')
            ax.axhline(y = tol2_abs, color="r", linestyle="--")
            ax.set_title('----- threshold allowed', color='r')
            fig.tight_layout()
            return ax

    def compute_group_velocity(self):
        """
        Post-process the group velocity, vg=1/Re(dk/domega) for every mode in the whole parameter range (opposite-going modes
        required). For unpaired modes, NaN values are set.
        """
        if len(self.group_velocity)==len(self.eigenvalues):
            print('Group velocity already computed')
            return
        if len(self.opposite_going)==0: #compute opposite-going modes if not yet computed      
            self.compute_opposite_going()
        start = time.perf_counter()
        np.seterr(divide='ignore') #ignore divide by zero message (denominator may sometimes vanish) 
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        for i, (eigenvalues, eigenvectors) in enumerate(zip(self.eigenvalues, self.eigenvectors)):
            #Computation of group velocity
            wavenumber, omega = self._concatenate('wavenumber', 'omega', i=i) #repeat parameter as many times as the number of eigenvalues
            group_velocity = np.zeros(eigenvalues.size) + np.NaN
            numerator = self.M*eigenvectors
            denominator = 2*self.K2*eigenvectors #note: computing the denominator can probably be avoided if _compute_biorthogonality_factor() has already been done
            denominator = denominator*self._diag(wavenumber)
            denominator += 1j*self.K1*eigenvectors
            denominator -= 1j*K1T*eigenvectors
            for mode in range(eigenvalues.size):
                opposite = self.opposite_going[i][mode]
                if opposite>=0: #the mode has been successfully paired
                    uleft = eigenvectors.getColumnVector(opposite)
                    group_velocity[mode] = 1/np.real( 2*omega[mode]*numerator.getColumnVector(mode).tDot(uleft) / denominator.getColumnVector(mode).tDot(uleft) )
            self.group_velocity.append(group_velocity)
        K1T.destroy()
        np.seterr(divide='warn')
        print(f'Computation of group velocity, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_traveling_direction(self, delta=1e-2):
        """
        Post-process the traveling direction, +1 or -1, for every mode in the whole parameter range,
        using the sign of Im(k + 1j*delta/v) where delta is the imaginary shift used for analytical
        continuation of k, and v is the group velocity (or, if not available, the energy velocity).
        This criterion is based on the limiting absorption principle (theoretically, vg should be used
        instead of ve). For unpaired modes, NaN values are set.
        """
        if len(self.traveling_direction)==len(self.eigenvalues):
            print('Traveling direction already computed')
            return
        if len(self.group_velocity)==0 and len(self.energy_velocity)==0: #both group velocity and energy velocity have not been already computed
            if len(self.opposite_going)==0:
                self.compute_energy_velocity() #the energy velocity is simpler to compute (the pairing of opposite-going is not required)
            else:
                self.compute_group_velocity()
        start = time.perf_counter()
        np.seterr(divide='ignore') #ignore divide by zero message (denominator may sometimes vanish)
        for i in range(len(self.eigenvalues)):
            wavenumber = self._concatenate('wavenumber', i=i)
            temp = delta/(self.energy_velocity[i] if len(self.group_velocity)==0 else self.group_velocity[i])
            temp[np.abs(wavenumber.imag)+np.abs(temp)>np.abs(wavenumber.real)] = 0 #do not use the LAP if |Im(k)| + |delta/ve| is significant
            traveling_direction = np.sign((wavenumber+1j*temp).imag)
            self.traveling_direction.append(traveling_direction)
            #Check if any exponentially growing modes (in the numerical LAP, delta is user-defined, which might lead to wrong traveling directions)
            growing_modes = np.logical_and(wavenumber.imag*traveling_direction<0, np.abs(wavenumber.imag)>1e-6*np.abs(wavenumber.real))
            if any(growing_modes):
                print('Warning in computing traveling direction: exponentially growing modes found (unproper sign of Im(k) detected)')
                print(f'for iteration {i}, with |Im(k)/Re(k)| up to {(np.abs(wavenumber[growing_modes].imag/wavenumber[growing_modes].real)).max():.2e}')
        np.seterr(divide='warn')
        print(f'Computation of traveling direction, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_pml_ratio(self):
        """
        Post-process the pml ratio (useful to filter out PML mode), given by 1-Im(Ek)/|Ek| where Ek denotes
        the "complex" kinetic energy, for every mode in the whole parameter range.      
        Reminder: the pml ratio tends to 1 for mode shapes vanishing inside the PML.
        """
        if len(self.pml_ratio)==len(self.eigenvalues):
            print('PML ratio already computed')
            return
        start = time.perf_counter()
        for i, eigenvectors in enumerate(self.eigenvectors):
            omega = self._concatenate('omega', i=i)
            Ek = 0.25*np.abs(omega**2)*self._dot_eigenvectors(i, self.M*eigenvectors) #"complex" kinetic energy
            self.pml_ratio.append(1-np.imag(Ek)/np.abs(Ek))
        print(f'Computation of pml ratio, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_response_coefficient(self, F, spectrum=None, wavenumber_function=None, dof=None):
        """
        Computation of modal coefficients due to the excitation vector F for every mode in the whole omega range (opposite-going
        eigenvectors are required).
        Modal coefficients qm are defined from: U(z,omega) = sum qm(omega)*Um(omega)*exp(i*km*z), m=1...M, omega denotes the
        angular frequency.
        For unpaired modes, NaN values are set.
        Assumption: the source is centred at z=0.
        
        Note: spectrum and wavenumber_function can be specified in compute_response(...) instead of     
        compute_response_coefficient(...), but not in both functions in the same time (otherwise the excitation will be
        modulated twice)
        
        Parameters
        ----------
        F : PETSc vector
            SAFE excitation vector
        spectrum : numpy.ndarray
            when specified, spectrum is a vector of length omega  used to modulate F in terms of frequency (default: 1 for
            all frequencies)
        wavenumber_function: python function
            when specified, wavenumber_function is a python function used to modulate F in terms of wavenumber (example:
            wavenumber_function = lambda x: np.sin(x), default: 1 for all wavenumbers, i.e. source localized at z=0)
        dof : int
            when specified, it calculates the modal excitability (stored in the attribute excitability), i.e. qm*Um at
            the degree of freedom dof and for a unit excitation vector (i.e. such that the sum of the elements of F is
            equal to 1)
        """
        #Initialization
        if self.problem_type=='wavenumber':
            raise NotImplementedError('Response coefficient computation not implemented in case wavenumber is parameter')
        self.F = F
        self.coefficient = [] #re-initialized every time compute_response_coefficient(..) is executed (F is an input)
        self.excitability = [] #idem
        if spectrum is None:
            spectrum = np.ones(self.omega.size)
        if wavenumber_function is None:
            wavenumber_function = lambda k: 1+0*k
        if dof is not None:
            force = self.F.sum() #summing elements of F amounts to integrate the (normal) stress over the cross-section 
        
        #Check
        if len(spectrum) != self.omega.size:
            raise NotImplementedError('The length of spectrum must be equal to the length of omega')
        if dof is not None and not isinstance(dof, int):
            raise NotImplementedError('dof must be an integer')
        if len(self.opposite_going)==0: #compute opposite-going modes if not yet computed      
            self.compute_opposite_going()
        if len(self.traveling_direction)==0: #compute traveling direction if not yet computed
            self.compute_traveling_direction()
        
        #Modal coefficients (loop over frequency)
        start = time.perf_counter()
        for i, opposite_going in enumerate(self.opposite_going):
            coefficient = np.array(self.eigenvectors[i].copy().transpose()*F)
            mode = np.arange(opposite_going.size)
            coefficient[mode[opposite_going<0]] = (1+1j)*np.NaN #coefficients of unpaired modes are set to NaN
            mode, opposite = mode[opposite_going>=0], opposite_going[opposite_going>=0] #consider paired modes only
            coefficient[mode] = coefficient[opposite]
            coefficient[mode] = coefficient[mode]/self._biorthogonality_factor[i][mode]*self.traveling_direction[i][mode]
            coefficient = coefficient*spectrum[i]*wavenumber_function(self.eigenvalues[i])
            self.coefficient.append(coefficient)
            if dof is not None:
                self.excitability.append(coefficient*self.eigenvectors[i][dof,:]/force)
        print(f'Computation of response coefficient, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_response(self, dof, z, omega_index=None, spectrum=None, wavenumber_function=None, plot=False):
        """
        Post-process the response (modal expansion) at the degree of freedom dof and the axial coordinate z, for the whole
        frequency range.
        The outputs are frequency, a numpy 1d array of size len(omega), and response, a numpy 2d array of size len(dof or
        z)*len(omega).
        dof and z cannot be both vectors, except if omega_index is specified or omega is scalar (single frequency computation):
        in that case, the array response is of size len(z)*len(dof), which can be useful to plot the whole field at a single
        frequency.
        
        The response at each frequency omega is calculated from:
        U(z,omega) = sum qm(omega)*Um(omega)*exp(i*km*z), m=1...M,
        where z is the receiver position along the waveguide axis.
        M is the number of modes traveling in the proper direction, positive if z is positive, negative if z is negative.
        The pairing of opposite-going eigenvectors is required, unpaired modes are discarded from the expansion.
        
        The outputs frequency and response are made dimensional when values in plot_scaler are not set to 1.
        
        Assumption: the source is assumed to be centred at z=0.
        
        Warning: the response calculation is only valid if z lies oustide the source region.
        
        Note: spectrum and wavenumber_function can be specified in compute_response_coefficient(...) instead
        of compute_response(...), but not in both functions in the same time (otherwise the excitation will be modulated twice).
        
        Parameters
        ----------
        dof : numpy array of integer
            dof where the response is computed
        z : numpy array
            axial coordinate where the response is computed
        omega_index : int
            omega index to compute the response at a single frequency, allowing the consideration of multiple dof and z
        spectrum : numpy.ndarray
            when specified, spectrum is a vector of length omega  used to modulate F in terms of frequency (default: 1 for
            all frequencies)
        wavenumber_function: python function
            when specified, wavenumber_function is a python function used to modulate F in terms of wavenumber (example:
            wavenumber_function = lambda x: np.sin(x), default: 1 for all wavenumbers, i.e. source localized at z=0)
        plot : bool
            if set to True, the magnitude and phase of response are plotted as a function of frequency
        
        Returns
        -------
        frequency: numpy 1d array
            the frequency vector, i.e. omega/(2*pi)
        response : numpy array (1d or 2d)
            the matrix response
        ax : matplotlib axes when plot is set to True
            ax[0] is the matplotlib axes used for magnitude, ax[1] is the matplotlib axes used for phase
        """
        
        #Initialization
        response = []
        dof = np.array(dof)
        z = np.array(z).reshape(-1, 1)
        if omega_index is None:
            omega_index = range(self.omega.size)
        else:
            if isinstance(omega_index, int): #single element special case
                omega_index = [omega_index]
            else:
                raise NotImplementedError('omega_index must be an integer')
        if spectrum is None:
            spectrum = np.ones(self.omega.size)
        
        #Check
        if self.problem_type=='wavenumber':
            raise NotImplementedError('Response computation not implemented in case wavenumber is parameter')
        if plot and len(omega_index)==1:
            raise NotImplementedError('Plot is not possible for a single frequency computation (please set plot to False)')
        if len(spectrum) != self.omega.size:
            raise NotImplementedError('The length of spectrum must be equal to the length of omega')
        if any(z==0):
            raise NotImplementedError('z cannot contain zero values (z must lie outside the source region)')
        if wavenumber_function is not None:
            print('Reminder: z should lie outside the source region')
        if wavenumber_function is None:
            wavenumber_function = lambda k: 1+0*k
        direction = np.sign(z)
        if np.abs(direction.sum())!=direction.size:
            raise NotImplementedError('z cannot contain both negative and positive values')
        if dof.size>1 and z.size>1 and len(omega_index)>1:
            raise NotImplementedError('dof and z cannot be both vectors in the same time except for a single frequency computation')
        if len(self.coefficient)==0: #response coefficient has not yet been computed
            print('Response coefficient must be computed first (execute compute_response_coefficient(...))')
            return
        
        #Response
        start = time.perf_counter()
        direction = direction[0]
        for i in omega_index:
            imode = np.nonzero(self.traveling_direction[i]==direction)[0].tolist() #indices of modes traveling in the desired direction
            temp = self.coefficient[i][imode]*spectrum[i]*wavenumber_function(self.eigenvalues[i][imode])
            #temp = (self.eigenvectors[i][dof.tolist(),imode] @ np.diag(temp)) @ np.exp(1j*np.outer(self.eigenvalues[i][imode], z)) #OLD: np.diag(temp) may have many zeros
            temp = PETSc.Mat().createDense([dof.size, len(imode)], array=self.eigenvectors[i][dof.tolist(),imode], comm=self.comm) * self._diag(temp)
            temp = temp[:,:] @ np.exp(1j*np.outer(self.eigenvalues[i][imode], z)) #note: PETSc matrices have no exponential function (although PETSc vectors have one!) -> going back to numpy arrays here for simplicity...
            temp = temp.reshape(-1,1) #enforce vector to be column
            response.append(temp)
        response = np.squeeze(np.concatenate(response, axis=1)) #numpy array of size len(dof or z)*len(self.omega)
        if len(omega_index)==1:
            response = response.reshape(z.size, dof.size) #numpy array of size len(z)*len(dof)
        print(f'Computation of response, elapsed time : {(time.perf_counter() - start):.2f}s')
        frequency = self.omega[omega_index]/(2*np.pi)
        
        # Scaling
        xscale, yscale = self.plot_scaler['omega'], 1/self.plot_scaler['wavenumber']
        frequency, response = frequency*xscale, response*yscale
        
        #Plots
        if plot:
            if all(np.array(list(self.plot_scaler.values()))==1): #the dictionnary variables are equal to 1
                xlabel, ylabel = "normalized angular frequency", "|normalized displacement|"
            else:
                xlabel, ylabel, xscale = "frequency", "|displacement|", xscale/(2*np.pi)
            #Magnitude
            fig, ax_abs = plt.subplots(1, 1)
            ax_abs.plot(self.omega.real*xscale, np.abs(response.T), linewidth=1, linestyle="-") #color="k"
            ax_abs.set_xlabel(xlabel)
            ax_abs.set_ylabel(ylabel)
            fig.tight_layout()
            #Phase
            fig, ax_angle = plt.subplots(1, 1)
            ax_angle.plot(self.omega.real*xscale, np.angle(response.T), linewidth=1, linestyle="-") #color="k"
            ax_angle.set_xlabel(xlabel)
            ax_angle.set_ylabel('phase')
            fig.tight_layout()
            return frequency, response, [ax_abs, ax_angle]
        else:
            return frequency, response

    def plot_phase_velocity(self, **kwargs):
        """
        Plot phase velocity dispersion curves, vp vs. Re(omega), where omega is replaced with frequency
        for dimensional results. Parameters and Returns: see plot(...).
        """
        return self.plot(y=['phase_velocity', np.real], **kwargs)

    def plot_attenuation(self, **kwargs):
        """
        Plot attenuation dispersion curves, Im(wavenumber) vs. Re(omega) if omega is the parameter,
        or Im(omega) vs. Re(wavenumber) if wavenumber is the parameter, where omega is replaced with frequency
        for dimensional results. Parameters and Returns: see plot(...).
        """
        return self.plot(y=['attenuation', np.real], **kwargs)

    def plot_energy_velocity(self, **kwargs):
        """
        Plot energy velocity dispersion curves, ve vs. Re(omega), where omega is replaced with frequency
        for dimensional results. Parameters and Returns: see plot(...).
        """
        return self.plot(y=['energy_velocity', np.real], **kwargs)

    def plot_group_velocity(self, **kwargs):
        """
        Plot group velocity dispersion curves, vg vs. Re(omega), where omega is replaced with frequency
        for dimensional results. Parameters and Returns: see plot(...).
        """
        return self.plot(y=['group_velocity', np.real], **kwargs)

    def plot_coefficient(self, **kwargs):
        """
        Plot response coefficients as a function of frequency, |q| vs. Re(omega), where omega is replaced with frequency
        for dimensional results. Parameters and Returns: see plot(...).
        """
        return self.plot(y=['coefficient', np.abs], **kwargs)

    def plot_excitability(self, **kwargs):
        """
        Plot excitability as a function of frequency, |e| vs. Re(omega), where omega is replaced with frequency
        for dimensional results. Parameters and Returns: see plot(...).
        """
        return self.plot(y=['excitability', np.abs], **kwargs)

    def plot(self, x=None, y=None, c=None, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, **kwargs):
        """
        Plot dispersion curves y[1](y[0]) vs. x[1](x[0])
        
        Parameters
        ----------
        x, y, c: list
            x[0], y[0], c[0] are strings corresponding to modal properties for the x-axis, y-axis and marker colors respectively
            (these strings can be: 'omega', 'wavenumber', 'energy_velocity', 'group_velocity', 'pml_ratio', 'eigenvalues',
            'excitability', 'eigenvectors', 'eigenforces', 'coefficient', 'frequency', 'attenuation', 'phase_velocity'),
            x[1], y[1], c[1] are the functions applied to x[0], y[0] and c[0] respectively (e.g. np.abs, np.angle, np.real, np.imag, etc.).
            If x is None but not y, x is set to ['omega', np.real] if results are normalized, or set to ['frequency', np.real] if they are
            dimensional. If both x and are None, plot dispersion curves Re(omega) or Re(frequency) vs. Re(wavenumber).
            If c is None, a single color is used for coloring markers (given by the input variable color).
        direction: int
            +1 for positive-going modes, -1 for negative-going modes, None for plotting all modes
        pml_threshold: float
            threshold to filter out PML modes (modes such that pml_ratio<pml_threshold)
        ax: matplotlib axis
            the matplotlib axis on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot
        
        Returns
        -------
        sc: the matplotlib collection
        """
        
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        normalized = all(np.array(list(self.plot_scaler.values()))==1) #test if the dictionnary variables are equal to 1 (results will be normalized) or not (results will be dimensional)
        if x is None: #particular cases
            if y is None:
                x, y = ['wavenumber', np.real], ['omega' if normalized else 'frequency', np.real]
            elif y[0] == 'attenuation':
                x = ['omega' if self.problem_type=='omega' else 'wavenumber', np.real]
                if self.problem_type=='omega' and not normalized:
                    x[0] = 'frequency'
            else:
                x = ['omega' if normalized else 'frequency', np.real]
        if x[0] in ('coefficient', 'excitability') and len(getattr(self, x[0]))==0:
            raise NotImplementedError('No ' + x[0] + ' has been computed')
        if y[0] in ('coefficient', 'excitability') and len(getattr(self, y[0]))==0:
            raise NotImplementedError('No ' + y[0] + ' has been computed')
        if x[0] in ('energy_velocity', 'group_velocity') and len(getattr(self, x[0]))==0:
            eval('self.compute_' + x[0] + '()')
        if y[0] in ('energy_velocity', 'group_velocity') and len(getattr(self, y[0]))==0:
            eval('self.compute_' + y[0] + '()')
        if c is None:
            c = [None, lambda c: None]
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Scaling and labels
        xscale, yscale, cscale = self.plot_scaler[x[0]], self.plot_scaler[y[0]], self.plot_scaler[c[0]] if c[0] is not None else 1
        xlabel = 'angular frequency' if x[0]=='omega' else x[0].replace("_", " ") #take the string x and replace underscores with whitespaces
        ylabel = 'angular frequency' if y[0]=='omega' else y[0].replace("_", " ") #id
        if normalized: #add string "normalized" to labels
            xlabel, ylabel = "normalized " + xlabel, "normalized " + ylabel
        
        # Build concatenaded arrays from the string x[0] and apply functions x[1] (idem for y and c)
        x_array, y_array, c_array = self._concatenate(x[0], y[0], c[0], direction=direction, pml_threshold=pml_threshold)
        x_array, y_array, c_array = x[1](x_array*xscale), y[1](y_array*yscale), c[1](c_array*cscale)
        if c[0] is None: #single color plot (no colorbar)
            c_array = color
        
        # Re(omega) vs. Re(k)
        sc = ax.scatter(x_array, y_array, s=markersize, c=c_array, marker=marker, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        if c[0] is not None:
            plt.colorbar(sc) #label=colors
        return sc

    def set_plot_scaler(self, length=1, time=1, mass=1, dim=3):
        """
        Define the characteristic length, time and mass in order to visualize plots in a dimensional form (by default, they are equal to 1).
        Set dim=3 for three-dimensional waveguides, dim=2 for two-dimensional waveguides (e.g. plates).
        Scaling factors for 'omega', 'wavenumber', 'energy_velocity', 'group_velocity', 'pml_ratio', 'eigenvalues', 'excitability',
        'eigenvectors', 'eigenforces', 'coefficient', 'frequency', 'attenuation', 'phase_velocity' are stored in the attribute name plot_scaler.
        If poynting normalization has already been applied, then the scalers for 'eigenvectors', 'eigenforces' and 'coefficient' are such that
        the dimensional cross-section power flow of eigenmodes is equal to 1 Watt (if no poynting normalization applied, these scalers are left to 1).
        Reminder: while the dimension of U (displacement) is in meter, the dimension of F (force) is in Newton for 3D waveguides
        and in Newton/meter for 2D waveguides (F is in mass*length**(dim-2)/time**2).
        """
        force = mass*length**(dim-2)/time**2
        self.plot_scaler = {'omega':1/time, 'wavenumber':1/length, 'energy_velocity':length/time, 'group_velocity':length/time, 'pml_ratio':1,
                            'eigenvalues':1/length if self.problem_type=='omega' else 1/time, 'excitability':length/force,
                            'eigenvectors':1, 'eigenforces':1, 'coefficient':1}
        self.plot_scaler.update({'frequency':self.plot_scaler['omega'], 'attenuation':self.plot_scaler['eigenvalues'], 'phase_velocity':self.plot_scaler['energy_velocity']})
        if self._poynting_normalization:
            normalization_factor_1W = 1/np.sqrt(force*length/time) #factor to normalize eigenmodes such that their dimensional cross-section power flow is equal to 1 Watt
            self.plot_scaler.update({'eigenvectors':normalization_factor_1W*length, 'eigenforces':normalization_factor_1W*force, 'coefficient':1/normalization_factor_1W})

    def plot_spectrum(self, index=0, c=None, ax=None, color="k",
                        marker="o", markersize=2, **kwargs):
        """
        Plot the spectrum, Im(k) vs. Re(k) computed for omega[index] (if the parameter is the frequency),
        or Im(omega) vs. Re(omega) for wavenumber[index] (if the parameter is the wavenumber).
        
        Parameters
        ----------
        index: int
            parameter index
        c: list
            c[0] is a string (must be an attribute of self) and c[1] is a function used for coloring markers,
            a single color (given by the input variable color) is used if c is None
        ax: matplotlib axis
            the matplotlib axis on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot

        Returns
        -------
        ax: the plot axe used for display
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        normalized = all(np.array(list(self.plot_scaler.values()))==1)
        if self.problem_type == "wavenumber":
            scale = self.plot_scaler['omega']
            title = "normalized angular frequency" if normalized else "angular frequency"
        elif self.problem_type == "omega":
            scale = self.plot_scaler['wavenumber']
            title = "normalized wavenumber" if normalized else "wavenumber"
        c = c[1](getattr(self,c[0])[index])*self.plot_scaler[c[0]] if c is not None else color
        sc = ax.scatter(self.eigenvalues[index].real*scale, self.eigenvalues[index].imag*scale, s=markersize, c=c, marker=marker, **kwargs)
        ax.set_xlabel('real part')
        ax.set_ylabel('imaginary part')
        ax.set_title(title)
        ax.figure.tight_layout()
        if c is not color:
            plt.colorbar(sc) #label=colors
        return sc

    def _check_biorthogonality(self, i):
        """ Return and plot, for the ith parameter, the Modal Assurance Criterion (MAC) matrix based on the (bi)-orthogonality relation (for internal use)"""
        if len(self.eigenforces)==0: #the eigenforces has not yet been computed
            print('Eigenforces have not yet been computed')
            return
        if self.problem_type == "wavenumber":
            biorthogonality = self.eigenvectors[i].copy().hermitianTranspose()*self.M*self.eigenvectors[i] #hyp: K0, K1, K2, M and eigenvalues must be real here!
            # Warning for lossy problems
            dofs_complex = np.iscomplex(self.K2.getDiagonal()[:])
            if any(dofs_complex):
                print("Warning: the orthogonality relation implemented is valid for real matrices only (lossless problems)")
        elif self.problem_type == "omega":
            biorthogonality = self.eigenforces[i].copy().transpose()*self.eigenvectors[i]-self.eigenvectors[i].copy().transpose()*self.eigenforces[i]
        plt.matshow(np.abs(biorthogonality[:,:]))
        plt.title(f'MAC matrix for iteration {i}')
        return biorthogonality

    def _get_eigenpairs(self, two_sided=False):
        """
        Return all converged eigenpairs of the current EVP object (for internal use).
        Eigenvectors are stored in a PETSc dense matrix.
        If two_sided is set to True, left eigensolutions are also included in the outputs, removing any duplicates.
        """        
        nconv = self.evp.getConverged()
        #Initialization
        if self.problem_type=='omega' and two_sided: #get rid of one half of complex plane to avoid duplicates
            eigenvalues = np.array([self.evp.getEigenpair(i) for i in range(nconv)])
            modes_kept = np.nonzero(np.logical_and(np.angle(eigenvalues)>-np.pi/4, np.angle(eigenvalues)<3*np.pi/4))[0]
        else: #keep all
            modes_kept = range(nconv) 
        eigenvalues = []
        v = self.evp.getOperators()[0].createVecRight()
        index = range(v.getSize()) if v.getSize()==self.M.getSize()[0] else range(int(v.getSize()/2)) #1/2 in case of externally linearized quadratic evp
        eigenvectors = PETSc.Mat().create(comm=self.comm)
        eigenvectors.setType("dense")
        eigenvectors.setSizes([self.M.getSize()[0], 2*len(modes_kept) if two_sided else len(modes_kept)])
        eigenvectors.setFromOptions()
        eigenvectors.setUp()
        #Build eigenpairs
        for i, mode in enumerate(modes_kept):
            eigenvalues.append(self.evp.getEigenpair(mode, v))
            if self.problem_type=='omega' and two_sided: #include left eigensolutions
                eigenvectors.setValues(index, 2*i, v[index])
                self.evp.getLeftEigenvector(mode, v)
                v.conjugate() #cancel the conjugate internally applied by SLEPc
                eigenvectors.setValues(index, 2*i+1, v[index])
                eigenvalues.append(-eigenvalues[-1])
            else:
                eigenvectors.setValues(index, i, v[index])
        eigenvectors.assemble()
        eigenvalues = np.array(eigenvalues)
        if self.problem_type=="wavenumber":
            eigenvalues = np.sqrt(eigenvalues)
        return eigenvalues, eigenvectors

    def _concatenate(self, *args, direction: Union[int, None]=None, pml_threshold: Union[int, None]=None, i: Union[int, None]=None):
        """
        Return concatenated modal properties in the whole parameter range as 1D numpy arrays (for internal use).
        The arguments *args are strings which can be 'omega', 'wavenumber', 'energy_velocity', 'group_velocity',
        'pml_ratio', 'eigenvalues', 'excitability', 'eigenvectors', 'eigenforces', 'coefficient', 'frequency',
        'attenuation', 'phase_velocity'.
        The parameter value (omega or wavenumber) is repeated as many as times as the number of eigenvalues.
        If direction is specified (+1 or -1), eigenmodes traveling in the non-desired direction are filtered out. 
        If pml_threshold is specified, eigenmodes such that pml_ratio<pml_threshold are filtered out.
        If i is specified, then the function returns the results for the ith parameter only.
        """
        argout = []
        index = slice(None) if i is None else slice(i, i+1)
        if 'phase_velocity' in args: #create the temporary attribute 'phase_velocity'
            np.seterr(divide='ignore') #ignore divide by zero message (denominator may sometimes vanish) 
            if self.problem_type=="omega":
                self.phase_velocity = [self.omega[i].real/self.eigenvalues[i].real for i in range(len(self.eigenvalues))]
            else: #"wavenumber"
                self.phase_velocity = [self.eigenvalues[i].real/self.wavenumber[i].real for i in range(len(self.eigenvalues))]
            np.seterr(divide='warn')
        for arg in args:
            (arg, one_or_two_pi) = ('omega', 2*np.pi) if arg=='frequency' else (arg, 1)
            if arg=="omega" and self.problem_type=="omega":
                array = np.repeat(self.omega[index], [len(egv) for egv in self.eigenvalues[index]])/one_or_two_pi
            elif arg=="wavenumber" and self.problem_type=="wavenumber":
                array = np.repeat(self.wavenumber[index], [len(egv) for egv in self.eigenvalues[index]])
            elif arg=="attenuation":
                array = np.concatenate(getattr(self, "eigenvalues")[index]).imag
            elif arg is None:
                array = []
            else:
                if (arg=="wavenumber" and self.problem_type=="omega") or (arg=="omega" and self.problem_type=="wavenumber"):
                    arg = "eigenvalues"
                if len(getattr(self, arg))==0:
                    raise NotImplementedError(f'{arg} has not been computed: please compute it before plotting')
                array = np.concatenate(getattr(self, arg)[index])/one_or_two_pi
            argout.append(array) 
        if direction is not None:
            traveling_direction = np.concatenate(self.traveling_direction[index])
            imode = traveling_direction==direction #indices of modes traveling in the desired direction
            argout = [argout[j][imode] if len(argout[j])>0 else [] for j in range(len(argout))]
        else:
            imode = slice(None)
        if pml_threshold is not None:
            pml_ratio = np.concatenate(self.pml_ratio[index])
            iphysical = pml_ratio[imode]>=pml_threshold #indices of physical modes (i.e. excluding PML modes)
            argout = [argout[j][iphysical] if len(argout[j])>0 else [] for j in range(len(argout))]
        if len(argout)==1:
            argout = argout[0]
        if 'phase_velocity' in args: #delete the attribute 'phase_velocity'
            del(self.phase_velocity)
        return argout

    def _compute_if_necessary(self, direction, pml_threshold):
        """ Compute traveling direction and pml ratio if necessary before plot (for internal use) """
        if direction is not None and len(self.traveling_direction)==0:  #compute the traveling direction if not yet computed
            self.compute_traveling_direction()
        if pml_threshold is not None and len(self.pml_ratio)==0:  #compute the pml_ratio if not yet computed
            self.compute_pml_ratio()

    def _diag(self, vec):
        """ Return the PETSc diagonal matrix with diagonal entries given by vector vec (for internal use)"""
        diag = PETSc.Mat().createAIJ(vec.size, nnz=1, comm=self.comm)
        diag.setUp()
        diag.setDiagonal(PETSc.Vec().createWithArray(vec, comm=self.comm))
        diag.assemble()
        return diag

    def _dot_eigenvectors(self, i, eigenfield):
        """
        Return the dot product, mode by mode, between eigenvectors[i] (taking their conjugate) and a given
        eigenfield (for internal use).
        The matrix eigenfield must have the same size as eigenvectors[i]
        """
        res = []
        for mode in range(self.eigenvectors[i].getSize()[1]):
            res.append(eigenfield.getColumnVector(mode).dot(self.eigenvectors[i].getColumnVector(mode))) #dot: conjugate, tDot: without
        res = np.array(res)
        return res

    def _build_block_matrix(self, A, B, C, D):
        """ Return the block matrix [[A, B], [C, D]] given the equal-sized blocks A, B, C, D (for internal use) """    
        bs = A.getSize() #block size
        #Block A
        csr = A.getValuesCSR()
        Mat = PETSc.Mat().createAIJWithArrays([2*bs[0], 2*bs[1]], [np.insert(csr[0], len(csr[0]), np.full(bs[0], csr[0][-1])), csr[1], csr[2]], comm=self.comm)
        #Block B
        csr = B.getValuesCSR()
        Mat = Mat + PETSc.Mat().createAIJWithArrays([2*bs[0], 2*bs[1]], [np.insert(csr[0], len(csr[0]), np.full(bs[0], csr[0][-1])), csr[1]+bs[1], csr[2]], comm=self.comm)
        #Block C
        csr = C.getValuesCSR()
        Mat = Mat + PETSc.Mat().createAIJWithArrays([2*bs[0], 2*bs[1]], [np.insert(csr[0], 0, np.zeros(bs[0])), csr[1], csr[2]], comm=self.comm)
        #Block D
        csr = D.getValuesCSR()
        Mat = Mat + PETSc.Mat().createAIJWithArrays([2*bs[0], 2*bs[1]], [np.insert(csr[0], 0, np.zeros(bs[0])), csr[1]+bs[1], csr[2]], comm=self.comm)
        return Mat


class Signal:
    """
    A class for handling signals in the time domain and in the frequency domain
    
    Reminder:
    
    - the sampling frequency fs must be at least twice the highest excited frequency (fs>=2fmax)
    - the time duration T must be large enough to capture the slowest wave at z, the source-receiver distance
    
    Fourier transform definition used: X(f) = 2/T * integral of x(t)*exp(+i*omega*t)*dt
    
    Two remarks:
    
    1. this is not the numpy fft function convention, which is in exp(-i*omega*t)
    2. the true amplitude of the Fourier transform, when needed, has to be obtained by
       multiplying the output (spectrum) by the scalar T/2, where T is the duration of the time signal
       (with the above definition: the division by T simplifies dimensionless analyses,
       and the factor 2 is used because only the positive part of the spectrum is considered)
    
    Complex Fourier transform:
    
    A complex Fourier transform is applied if alpha is set to a nonzero value.
    The frequency vector has then an imaginary part, constant and equal to alpha/(2*pi).
    Complex frequency computations can be useful for the analysis of long time duration signals (avoids aliasing).
    A good choice is alpha = log(50)/T.
    Note that the first frequency component is kept in that case (the frequency has a zero real part
    but non-zero imaginary part).
    
    Example::
    
        mysignal = Signal(alpha=0*np.log(50)/5e-4)
        mysignal.toneburst(fs=5000e3, T=5e-4, fc=100e3, n=5)
        mysignal.plot()
        mysignal.fft()
        mysignal.plot_spectrum()
        mysignal.ifft(coeff=1)
        mysignal.plot()
        plt.show()
    
    Attributes
    ----------
    time : numpy 1d array
        time vector
    waveform : numpy nd array
        waveform vectors stacked as rows (waveform is an array of size number_of_signals*len(time))
    frequency : numpy 1d array
        frequency vector
    spectrum : numpy nd array
        spectrum vectors stacked as rows (spectrum is an array of size number_of_signals*len(frequency))
    alpha : float
        decaying parameter to apply complex Fourier transform (useful for long time duration signal)
    
    Methods
    -------
    __init__(time=None, waveform=None, frequency=None, spectrum=None, alpha=0):
        Constructor, initialization of signal (specify either waveform vs. time or spectrum vs. frequency)
    fft():
        Compute Fourier transform, results are stored as attributes (names: frequency, spectrum) 
    ifft(coeff=1):
        Compute inverse Fourier transform, results are stored as attributes (names: time, waveform)
    ricker(fs, T, fc):
        Generate a Ricker signal
    toneburst(fs, T, fc, n):
        Generate a toneburst signal
    chirp(fs, T, f0, f1, chirp_duration):
        Generate a chirp signal
    plot():
        Plot time waveform (waveform vs. time)
    plot_spectrum():
        Plot the spectrum (spectrum vs. frequency), in magnitude and phase
    """
    def __init__(self, time=None, waveform=None, frequency=None, spectrum=None, alpha=0):
        """
        Constructor
        
        Parameters
        ----------
        time : numpy 1d array
            time vector
        waveform : numpy array (1d or 2d)
            amplitude of signals in the time domain
        frequency : numpy 1d array
            frequency vector
        spectrum : numpy array (1d or 2d)
            amplitude of signals in the frequency domain
        alpha : float
            decaying parameter to apply complex Fourier transform (useful for long time duration signal)
        """
        self.time = time
        self.waveform = waveform
        self.frequency = frequency
        self.spectrum = spectrum
        self.alpha = alpha
        
        if (time is None) ^ (waveform is None):
            raise NotImplementedError('Please specify both time and waveform')
        if (frequency is None) ^ (spectrum is None):
            raise NotImplementedError('Please specify both frequency and spectrum')

    def fft(self):
        """
        Compute Fourier transform (positive frequency part only, time waveform are assumed to be real).
        If the number of time steps is odd, one point is added.
        The zero frequency, if any, is suppressed.
        Results are stored as attributes (names: frequency, spectrum).
        spectrum is an array of size number_of_signals*len(frequency)
        """
        # Check waveform
        if self.time is None:
            raise ValueError("Time waveform is missing")
        self.waveform = self.waveform.reshape(-1, len(self.time))
        dt = np.mean(np.diff(self.time))
        if len(self.time)%2 != 0:  #if the number of points is odd, complete with one point
            print("One point added in order to have length of t even")
            self.time = np.append(self.time, self.time[-1] + dt)
            self.waveform = np.append(self.waveform, np.array([0]*self.waveform.shape[0]).reshape(-1, 1), axis=1)  #complete with one zero
        self.waveform = np.squeeze(self.waveform)
        temp = np.diff(self.time)
        if np.max(np.abs(temp-dt))/dt >= 1e-3:
            raise ValueError("Time steps might be unequally spaced! Please check")
        
        # FFT of excitation (the time signal x is multiplied by exp(-alpha*t) for complex Fourier transform)
        #T = self.time[-1] #time duration
        N = len(self.time)
        fs = 1/dt #sampling frequency
        self.spectrum = np.fft.rfft(self.waveform * np.exp(-self.alpha * self.time)[np.newaxis, :], N).conj() / N  #conj() because our convention is +i*omega*t, as opposed to fft function
        Np = N//2+1  #number of points of the positive part of the spectrum (N is even)
        self.frequency = fs/2*np.linspace(0, 1, Np)  #frequency vector
        self.frequency = self.frequency + 1j*self.alpha/(2*np.pi)  #complex frequency
        self.spectrum = 2*self.spectrum.reshape(-1, Np)
        if self.frequency[0]==0:  #suppress first frequency if zero
            self.frequency = self.frequency[1:]
            self.spectrum = self.spectrum[:, 1:]
        self.spectrum = np.squeeze(self.spectrum)

    def ifft(self, coeff=1):
        """
        Compute inverse Fourier transform (only the positive frequency part is needed, time waveform are assumed to be real).
        Zero padding is applied in the low-frequency range (if missing) and in the high-frequency range (if coeff is greater than 1).
        Zero padding in the high frequency range is applied up to the frequency coeff*max(frequency).
        Results are stored as attributes (names: time, waveform).
        waveform is an array of size number_of_signals*len(time).
        """    
        # Check spectrum
        if self.frequency is None:
            raise ValueError("Frequency spectrum is missing")
        if len(np.unique(np.imag(self.frequency))) > 1:
            raise ValueError('The imaginary part of the frequency vector must remain constant')
        
        # Zero padding in low and high frequencies
        frequency = np.real(self.frequency)
        df = np.mean(np.diff(frequency))  #frequency step
        if abs(frequency[0]) < 1e-3*df:  #the first frequency is zero
            frequency[0] = 0
            f_low = np.array([])
        else:  #non-zero first frequency
            f_low = np.arange(0, frequency[0]-1e-6*df, df)  #low frequency
        f_high = np.arange(frequency[-1]+df, coeff*frequency[-1]+1e-6*df, df)  #high frequency
        frequency = np.concatenate([f_low, frequency, f_high])
        spectrum = self.spectrum.reshape(-1, len(self.frequency))
        spectrum = np.concatenate([np.zeros((spectrum.shape[0], len(f_low))), spectrum, np.zeros((spectrum.shape[0], len(f_high)))], axis=1)
        if len(f_low) > 0:
            print('Zero padding applied in the missing low-frequency range')
        if len(f_high) > 0:
            print('Zero padding applied in the high-frequency range')
        temp = np.diff(frequency)
        if np.max(np.abs(temp-df))/df >= 1e-3:
            raise ValueError('Frequency steps might be unequally spaced! Please check')
        
        # IFFT of response
        Np = spectrum.shape[1]  #number of points of the spectrum (positive part of the spectrum)
        N = 2*(Np-1)  #number of points for the IFFT
        dt = 1/(N*df)  #sample time
        self.time = np.arange(0, N)*dt
        self.waveform = np.fft.irfft(spectrum.conj(), N) * Np
        self.waveform *= np.exp(self.alpha*self.time[np.newaxis, :]) #for complex Fourier transform
        self.waveform = np.squeeze(self.waveform)

    def ricker(self, fs, T, fc):
        """
        Generate a Ricker wavelet signal of unit amplitude (fs: sampling frequency, T: time duration, fc: Ricker central frequency)
        
        Note that for better accuracy:
        
        - fs is rounded so that fs/fc is an integer
        - T is adjusted so that the number of points is even
        """
        # Time
        fs = np.ceil(fs/fc)*fc  #redefine fs so that fs/fc is an integer
        dt = 1/fs  #time step
        T = np.round(T/2/dt)*2*dt + dt  #redefine T so that the number of points is equal to an even integer
        self.time = np.arange(0, T+1e-6*dt, dt)  #time vector
        #N = len(self.time)  # number of points (N=T/dt+1, even)
        
        # Ricker waveform
        t0 = 1/fc
        self.waveform = (1-2*(self.time-t0)**2*np.pi**2*fc**2)*np.exp(-(self.time-t0)**2*np.pi**2*fc**2)
        self.fft()

    def toneburst(self, fs, T, fc, n):
        """
        Generate a toneburst signal (fs: sampling frequency, T: time duration, fc: central frequency, n: number of cycles).
        This signal is a Hanning-modulated n cycles sinusoidal toneburst centred at fc Hz (with unit amplitude).
        For this kind of excitation, fmax can be considered as 2*fc roughly, hence one should choose fs>=4fc.
        
        Note that for better accuracy:
        
        - fs is rounded so that fs/fc is an integer
        - T is adjusted so that the number of points is even
        """
        # Time
        fs = np.ceil(fs/fc)*fc  #redefine fs so that fs/fc is an integer
        dt = 1/fs  #time step
        T = np.round(T/2/dt)*2*dt + dt  #redefine T so that the number of points is equal to an even integer
        self.time = np.arange(0, T+1e-6*dt, dt)  #time vector
        #N = len(self.time)  # number of points (N=T/dt+1, even)
        
        # Toneburst waveform
        t = np.arange(0, n/fc+1e-6*dt, dt)  #n/fc yields an integer number of time steps because fs/fc is an integer
        x = np.sin(2*np.pi*fc*t)
        x *= np.hanning(len(x))  #hanning window
        self.waveform = np.zeros(len(self.time))
        self.waveform[:len(x)] = x  #time amplitude vector
        self.fft()

    def chirp(self, fs, T, f0, f1, chirp_duration):
        """
        Generate a chirp of unit amplitude (fs: sampling frequency, T: time duration, f0: first frequency, f1: last frequency, chirp_duration: time to sweep from f0 to f1).
        Note that for better accuracy, T is adjusted so that the number of points is even.
        """
        # Time
        dt = 1/fs  #time step
        T = np.floor(T/2/dt)*2*dt + dt  #redefine T so that the number of points is equal to an even integer
        self.time = np.arange(0, T+1e-6*dt, dt)  #time vector
        
        # Chirp waveform
        index = np.argmin(np.abs(self.time-chirp_duration))
        t = self.time[:index]
        x = np.sin(2*np.pi*f0*t + np.pi*(f1-f0)/chirp_duration*t**2)
        self.waveform = np.zeros(len(self.time))
        self.waveform[:len(x)] = x  #time amplitude vector
        self.fft()

    def plot(self, ax=None, color="k", linewidth=1, linestyle="-", **kwargs):
        """ Plot time waveform (waveform vs. time) """
        # Initialization
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # Plot waveform vs. time
        ax.plot(self.time, self.waveform.T, color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.figure.tight_layout()
        return ax

    def plot_spectrum(self, color="k", linewidth=2, linestyle="-", **kwargs):
        """ Plot the spectrum (spectrum vs. frequency), in magnitude and phase """
        # Plot spectrum magnitude vs. frequency
        fig, ax_abs = plt.subplots(1, 1)
        ax_abs.plot(self.frequency.real, np.abs(self.spectrum.T), color=color, linewidth=1, linestyle=linestyle, **kwargs)
        ax_abs.set_xlabel('f')
        ax_abs.set_ylabel('|X|')
        fig.tight_layout()
        
        # Plot spectrum phase vs. frequency
        fig, ax_angle = plt.subplots(1, 1)
        ax_angle.plot(self.frequency.real, np.angle(self.spectrum.T), color=color, linestyle=linestyle, **kwargs)
        ax_angle.set_xlabel('f')
        ax_angle.set_ylabel('arg(X)')
        fig.tight_layout()
        
        return ax_abs, ax_angle
