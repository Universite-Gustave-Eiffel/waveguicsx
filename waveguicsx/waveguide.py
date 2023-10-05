from typing import Union, List
from petsc4py import PETSc
from slepc4py import SLEPc

import matplotlib.pyplot as plt
import numpy as np
import time

class Waveguide:
    """
    A class for solving waveguide problems (based on SLEPc eigensolver)
    
    The problem must be based on the so-called SAFE (Semi-Analytical Finite Element) formulation:
    (K0-omega**2*M + 1j*k*(K1-K1^T) + k**2*K2)*U=0
    The varying parameter can be the angular frequency omega or the wavenumber k.
    In the former case, the eigenvalue is k, while in the latter case, the eigenvalue is omega**2.
    
    Example:
    import waveguicsx
    param = np.arange(0.1, 2, 0.1)
    waveguide = waveguicsx.Waveguide(MPI.COMM_WORLD, M, K0, K1, K2)
    waveguide.set_parameters(wavenumber=param) #or: waveguide.setParameters(omega=param)
    waveguide.solve(nev)
    waveguide.plot()
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
        target around which eigenpairs are looked for and set from solve(...)
    omega or wavenumber : numpy.ndarray
        the parameter range specified by the user (see method setParameters)
    evp : PEP or EPS instance (SLEPc object)
        eigensolver parameters (EPS if problem_type is "wavenumber", PEP otherwise)
    eigenvalues : list of numpy arrays
        list of wavenumbers or angular frequencies
        access to components with eigenvalues[ip][imode] (ip: parameter index, imode: mode index)
    eigenvectors : list of PETSc matrices
        list of mode shapes
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
    
    Methods
    -------
    set_parameters(omega=None, wavenumber=None, two_sided=False):
        Set problem type (problem_type), the parameter range (omega or wavenumber) as well as default parameters of SLEPc eigensolver (evp)
        Set two_sided to True if left eigenvectors are needed (for response and group velocity computation)
    solve(nev=1, target=0):
        Solve the eigenvalue problem repeatedly for the parameter range, solutions are stored as attributes (names: eigenvalues,
        eigenvectors)
    compute_eigenforce():
        Compute the eigenforces for the whole parameter range and store them as an attribute (name: eigenforces)
    compute_poynting_normalization():
        Normalization of eigenvectors and eigenforces, so that U'=U/sqrt(|P|), where P is the normal component of complex Poynting vector
    compute_opposite_going(plot=False):
        Compute opposite-going mode pairs based on on wavenumber and biorthogonality for the whole parameter range and store them as
        attributes (name: opposite_going), set plot to True to visualize the biorthogonality values of detected pairs        
    compute_energy_velocity():    
        Compute the energy velocities for the whole parameter range and store them as an attribute (name: energy_velocity)
    compute_group_velocity():
        Compute the group velocities for the whole parameter range and store them as an attribute (name: energy_velocity)
        Left eigenvectors are required (two_sided must be set to True)
    compute_traveling_direction():
        Compute the traveling directions for the whole parameter range and store them as an attribute (name: traveling_direction)
    compute_pml_ratio():
        Compute the pml ratios for the whole parameter range and store them as an attribute (name: pml_ratio)
    compute_response_coefficient(F, spectrum=None, wavenumber_function=None, dof=None):
        Compute the response coefficients due to excitation vector F for the whole parameter range and store them as an attribute (name: coefficient)
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
        self._poynting_normalization = None
        self._biorthogonality_factor: list = []
        
        # Print the number of degrees of freedom
        print(f'Total number of degrees of freedom: {self.M.size[0]}')

    def set_parameters(self, omega: Union[np.ndarray, None]=None, wavenumber:Union[np.ndarray, None]=None, two_sided=False):
        """
        Set the parameter range (omega or wavenumber) as well as default parameters of the SLEPc eigensolver (evp)
        The user must specify the parameter omega or wavenumber, but not both
        This method generates the attributes omega (or wavenumber) and evp
        After this method call, different SLEPc parameters can be set by changing the attribute evp manually
        Set two_sided=True for solving left eigenvectors also (for response and group velocity computation)
        
        Parameters
        ----------
        omega or wavenumber : numpy.ndarray
            the parameter range specified by the user
        two_sided : bool
            False if left eigenvectiors are not needed, True if they must be solved also (for response and group velocity computation)
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
        Solve the dispersion problem, i.e. the eigenvalue problem repeatedly for the parameter range (omega or wavenumber)
        The solutions are stored in the attributes eigenvalues and eigenvectors
        
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
            wavenumber, _ = self._concatenate(i=i) #repeat parameter as many times as the number of eigenvalues
            self.eigenforces.append(K1T*eigenvectors+1j*self.K2*eigenvectors*self._diag(wavenumber))
        K1T.destroy()
        print(f'Computation of eigenforces, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_poynting_normalization(self):
        """
        Post-process the normalization of eigenvectors and eigenforces, so that U'=U/sqrt(|P|),
        where P is the normal component of complex Poynting vector
        After normalization, every mode is such that |P|=1 and the attribute _poynting_normalization is set to True
        Normalization is not mandatory but, when applied, has to be done before any response coefficient computation
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
            _, omega = self._concatenate(i=i)
            #Normalization
            normalization = []
            for mode in range(self.eigenvalues[i].size):
                U = self.eigenvectors[i].getColumnVector(mode)
                F = self.eigenforces[i].getColumnVector(mode)
                normalization.append(1/np.sqrt(np.abs(-1j*omega[mode]/2*F.dot(U))))
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
            wavenumber, omega = self._concatenate(i=i)
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
        as a an attribute (name: opposite_going, -1 value for unpaired modes)
        Compute their biorthogonality normalization factors, Um^T*F-m - U-m^T*Fm, where m and -m denote opposite-going
        modes, for the whole parameter range and store them as an attribute (name: _biorthogonality_factor)
        If plot is set to True, the biorthogonality factors found by the algorithm are plotted in magnitude as a function
        of frequency, allowing visual check that there is no values close to zero (a factor close to zero probably means
        a lack of biorthogonality)
        Notes:
        - when an unpaired mode is found, the value -1 is stored in opposite_going (and NaN value in _biorthogonality_factor), meaning that
          this mode will be discarded in the computation of group velocity, traveling direction, coefficient and excitability (a NaN value will be stored)
        - if modes with lack of biorthogonality or two many unpaired modes occur, try to recompute the eigenproblem by increasing the accuracy (e.g. reducing the tolerance)
        - lack of biorthogonality may be also due to multiple modes (*): in this case, try to use an unstructured mesh instead
        (*) e.g. flexural modes in a cylinder with structured mesh
        """
        tol1 = 1e-4 #tolerance for relative difference between opposite wavenumbers (wavenumber criterion)
        tol2_rel = 100 #minimum ratio between the first two candidate biorthogonality factors (biorthogonality criterion)
        tol2_abs = 1e-3 #minimum biorthogonality factor to keep a given opposite pair (biorthogonality criterion)
        if self.problem_type=='wavenumber' or self.target!=0:
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
                        candidates2 = lower_half[np.argmin(np.abs((eigenvalues[mode]+eigenvalues[lower_half])/eigenvalues[mode]))] #find the second candidate
                        eigenvalues[candidates] = temp #back to initial
                        candidates = np.append(candidates, candidates2)
                    #Second criterion: based on biorthogonality
                    biorthogonality_test = []
                    for c in candidates:
                        biorthogonality_test = np.append(biorthogonality_test,
                                         self.eigenforces[i].getColumnVector(c).tDot(self.eigenvectors[i].getColumnVector(mode))
                                       - self.eigenvectors[i].getColumnVector(c).tDot(self.eigenforces[i].getColumnVector(mode)))
                    #OLD: biorthogonality_test = self.eigenforces[i][:,candidates.tolist()].T @ self.eigenvectors[i][:,mode] - self.eigenvectors[i][:,candidates.tolist()].T @ self.eigenforces[i][:,mode]
                    criterion2 = np.abs(biorthogonality_test)*self.omega[i]/4
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
                    self.traveling_direction[i][unpaired] = np.NaN #or 0?! -> try traveling_direction as integer?????
            #Store pairs and biorthogonality factors
            self.opposite_going.append(opposite_going)
            self._biorthogonality_factor.append(biorthogonality_factor)
        print(f'Computation of pairs of opposite-going modes, elapsed time : {(time.perf_counter() - start):.2f}s')
        #Plot the biorthogonality factors as a function of frequency
        if plot:
            omega = np.repeat(self.omega.real, [len(egv) for egv in self._biorthogonality_factor])
            biorthogonality_factor = np.concatenate(self._biorthogonality_factor)
            fig, ax = plt.subplots(1, 1)
            ax.plot(omega, np.abs(biorthogonality_factor)*omega/4, marker="o", markersize=2, linestyle="", color="k")
            ax.set_xlabel('Re(omega)')
            ax.set_ylabel('|biorthogonality factor|')
            ax.set_yscale('log')
            ax.axhline(y = tol2_abs, color="r", linestyle="--")
            ax.set_title('----- threshold allowed', color='r')
            fig.tight_layout()
            return ax

    def compute_group_velocity(self):
        """
        Post-process the group velocity, vg=1/Re(dk/domega) for every mode in the whole parameter range (opposite-going modes required)
        For unpaired modes, NaN values are set
        The traveling direction is also determined
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
            wavenumber, omega = self._concatenate(i=i) #repeat parameter as many times as the number of eigenvalues
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
        continuation of k, and v is the group velocity (or, if not available, the energy velocity)
        This criterion is based on the limiting absorption principle (theoretically, vg should be used
        instead of ve)
        For unpaired modes, NaN values are set 
        """
        if len(self.traveling_direction)==len(self.eigenvalues):
            print('Traveling direction already computed')
            return
        if len(self.group_velocity)==0 and len(self.energy_velocity)==0: #both group velocity and energy velocity have not been already computed
            self.compute_energy_velocity() #use the energy velocity (simpler to compute: the pairing of opposite-going is not required) 
        start = time.perf_counter()
        for i in range(len(self.eigenvalues)):
            wavenumber, _ = self._concatenate(i=i)
            temp = delta/(self.energy_velocity[i] if len(self.group_velocity)==0 else self.group_velocity[i])
            temp[np.abs(wavenumber.imag)+np.abs(temp)>np.abs(wavenumber.real)] = 0 #do not use the LAP if |Im(k)| + |delta/ve| is significant
            traveling_direction = np.sign((wavenumber+1j*temp).imag)
            self.traveling_direction.append(traveling_direction)
            #Check if any exponentially growing modes (in the numerical LAP, delta is user-defined, which might lead to wrong traveling directions)
            growing_modes = np.logical_and(wavenumber.imag*traveling_direction<0, np.abs(wavenumber.imag)>1e-6*np.abs(wavenumber.real))
            if any(growing_modes):
                print('Warning in computing traveling direction: exponentially growing modes found (unproper sign of Im(k) detected)')
                print(f'for iteration {i}, with |Im(k)/Re(k)| up to {(np.abs(wavenumber[growing_modes].imag/wavenumber[growing_modes].real)).max():.2e}')
        print(f'Computation of traveling direction, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_pml_ratio(self):
        """
        Post-process the pml ratio (useful to filter out PML mode), given by 1-Im(Ek)/|Ek| where Ek denotes
        the "complex" kinetic energy, for every mode in the whole parameter range        
        Note that in the absence of PML, the pml ratio is equal to 1
        """
        if len(self.pml_ratio)==len(self.eigenvalues):
            print('PML ratio already computed')
            return
        start = time.perf_counter()
        for i, eigenvectors in enumerate(self.eigenvectors):
            _, omega = self._concatenate(i=i)
            Ek = 0.25*np.abs(omega**2)*self._dot_eigenvectors(i, self.M*eigenvectors) #"complex" kinetic energy
            self.pml_ratio.append(1-np.imag(Ek)/np.abs(Ek))
        print(f'Computation of pml ratio, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_response_coefficient(self, F, spectrum=None, wavenumber_function=None, dof=None):
        """
        Computation of modal coefficients due to the excitation vector F for every mode in the whole omega range (opposite-going eigenvectors are required)
        Modal coefficients qm are defined from: U(z,omega) = sum qm(omega)*Um(omega)*exp(i*km*z), m=1...M, omega denotes the angular frequency
        For unpaired modes, NaN values are set
        Assumption: the source is centred at z=0
        Note: spectrum and wavenumber_function can be specified in compute_response(...) instead of compute_response_coefficient(...),
        but not in both functions in the same time (otherwise the excitation would be modulated twice)
        
        Parameters
        ----------
        F : PETSc vector
            SAFE excitation vector
        spectrum : numpy.ndarray
            when specified, spectrum is a vector of length omega  used to modulate F in terms of frequency (default: 1 for all frequencies)
        wavenumber_function: python function
            when specified, wavenumber_function is a python function used to modulate F in terms of wavenumber (example:
            wavenumber_function = lambda x: np.sin(x), default: 1 for all wavenumbers, i.e. source localized at z=0)
        dof : int
            when specified, it calculates the modal contribution qm*Um at the degree of freedom dof (equal to the so-defined modal excitability
            if spectrum and wavenumber_function are equal to 1, i.e. unit force localized at a single degree of freedom),
            stored in the attribute excitability  
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
                self.excitability.append(self.coefficient[-1]*self.eigenvectors[i][dof,:])
        print(f'Computation of response coefficient, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_response(self, dof, z, omega_index=None, spectrum=None, wavenumber_function=None, plot=False):
        """
        Post-process the response (modal expansion) at the degree of freedom dof and the axial coordinate z, for the whole frequency range
        The outputs are frequency, a numpy 1d array of size len(omega), and response, a numpy 2d array of size len(dof or z)*len(omega)
        dof and z cannot be both vectors, except if omega_index is specified or omega is scalar (single frequency computation): in that case,
        the array response is of size len(z)*len(dof), which can be useful to plot the whole field at a single frequency
        The response at each frequency omega is calculated from:
        U(z,omega) = sum qm(omega)*Um(omega)*exp(i*km*z), m=1...M,
        where z is the receiver position along the waveguide axis
        M is the number of modes traveling in the proper direction, positive if z is positive, negative if z is negative
        The pairing of opposite-going eigenvectors is required, unpaired modes are discarded from the expansion
        Assumption: the source is assumed to be centred at z=0
        Warning: the response calculation is only valid if z lies oustide the source region
        Note: spectrum and wavenumber_function can be specified in compute_response_coefficient(...) instead
        of compute_response(...), but not in both functions in the same time (otherwise the excitation would be modulated twice)
        
        Parameters
        ----------
        dof : numpy array of integer
            dof where the response is computed
        z : numpy array
            axial coordinate where the response is computed
        omega_index : int
            omega index to compute the response at a single frequency, allowing the consideration of multiple dof and z
        spectrum : numpy.ndarray
            when specified, spectrum is a vector of length omega  used to modulate F in terms of frequency (default: 1 for all frequencies)
        wavenumber_function: python function
            when specified, wavenumber_function is a python function used to modulate F in terms of wavenumber (example:
            wavenumber_function = lambda x: np.sin(x), default: 1 for all wavenumbers, i.e. source localized at z=0)
        plot : bool
            if set to True, the magnitude and phase of response are plotted as a function of frequency
        
        Returns
        -------
        frequency: numpy 1d array, the frequency vector, i.e. omega/(2*pi)
        response : numpy array, the matrix response
        ax : when plot is set to True, ax[0] is the matplotlib axes used for magnitude, ax[1] is the matplotlib axes used for phase
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
        
        #Plots
        if plot:
            #Magnitude
            fig, ax_abs = plt.subplots(1, 1)
            ax_abs.plot(self.omega.real, np.abs(response.T), linewidth=1, linestyle="-") #color="k"
            ax_abs.set_xlabel('Re(omega)')
            ax_abs.set_ylabel('|u|')
            fig.tight_layout()
            #Phase
            fig, ax_angle = plt.subplots(1, 1)
            ax_angle.plot(self.omega.real, np.angle(response.T), linewidth=1, linestyle="-") #color="k"
            ax_angle.set_xlabel('Re(omega)')
            ax_angle.set_ylabel('arg(u)')
            fig.tight_layout()
            return frequency, response, [ax_abs, ax_angle]
        else:
            return frequency, response

    def plot(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot dispersion curves Re(omega) vs. Re(wavenumber)
        
        Parameters
        ----------
        direction: +1 for positive-going modes, -1 for negative-going modes, None for plotting all modes
        pml_threshold: threshold to filter out PML modes (modes such that pml_ratio<pml_threshold)
        ax: the matplotlib axes on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot

        Returns
        -------
        ax: the matplotlib axes used for display
        """
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        wavenumber, omega = self._concatenate(direction=direction, pml_threshold=pml_threshold)
        
        # Re(omega) vs. Re(k)
        ax.plot(wavenumber.real, omega.real, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        ax.set_xlim(wavenumber.real.min(), wavenumber.real.max())
        ax.set_ylim(omega.real.min(), omega.real.max())
        ax.set_xlabel('Re(k)')
        ax.set_ylabel('Re(omega)')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax
 
    def plot_phase_velocity(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot phase velocity dispersion curves, vp=Re(omega)/Re(wavenumber) vs. Re(omega)
        Parameters and Returns: see plot(...)
        """
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        wavenumber, omega = self._concatenate(direction=direction, pml_threshold=pml_threshold)
        
        # vp vs. Re(omega)
        vp = omega.real/wavenumber.real
        ax.plot(omega.real, vp, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        ax.set_xlim(omega.real.min(), omega.real.max())
        ylim = omega.real.max()/np.abs(wavenumber.real).mean()
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel('Re(omega)')
        ax.set_ylabel('vp')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax

    def plot_attenuation(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot attenuation dispersion curves, Im(wavenumber) vs. Re(omega) if omega is the parameter,
        or Im(omega) vs. Re(omega) if wavenumber is the parameter
        Parameters and Returns: see plot(...)
        """
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        wavenumber, omega = self._concatenate(direction=direction, pml_threshold=pml_threshold)
        
        # Im(k) vs Re(omega)
        if self.problem_type == "wavenumber":
            ax.plot(omega.real, omega.imag, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
            ax.set_ylim(omega.imag.min(), omega.imag.max()+1e-6)
            ax.set_ylabel('Im(omega)')
        elif self.problem_type == "omega":
            ax.plot(omega.real, wavenumber.imag, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
            ax.set_ylim(wavenumber.imag.min(), wavenumber.imag.max()+1e-6)
            ax.set_ylabel('Im(k)')
        ax.set_xlim(omega.real.min(), omega.real.max())
        ax.set_xlabel('Re(omega)')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax

    def plot_energy_velocity(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot energy velocity dispersion curves, ve vs. Re(omega)
        Parameters and Returns: see plot(...)
        """
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if len(self.energy_velocity)==0:  #compute the energy velocity if not yet computed
            self.compute_energy_velocity()
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        _, omega, ve = self._concatenate('energy_velocity', direction=direction, pml_threshold=pml_threshold)
        
        # vp vs. Re(omega)
        ax.plot(omega.real, ve, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        ax.set_xlim(omega.real.min(), omega.real.max())
        ax.set_ylim(ve.min(), ve.max())
        ax.set_xlabel('Re(omega)')
        ax.set_ylabel('ve')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax

    def plot_group_velocity(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot group velocity dispersion curves, ve vs. Re(omega)
        Parameters and Returns: see plot(...)
        """
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if len(self.group_velocity)==0:  #compute the group velocity if not yet computed
            self.compute_group_velocity()
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        _, omega, vg = self._concatenate('group_velocity', direction=direction, pml_threshold=pml_threshold)
        
        # vp vs. Re(omega)
        ax.plot(omega.real, vg, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        ax.set_xlim(omega.real.min(), omega.real.max())
        ax.set_xlabel('Re(omega)')
        ax.set_ylabel('vg')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax

    def plot_coefficient(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot response coefficients as a function of frequency, |q| vs. Re(omega)
        Parameters and Returns: see plot(...)
        """
        if len(self.coefficient)==0:
            raise NotImplementedError('No response coefficient has been computed')
        
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        _, omega, coefficient = self._concatenate('coefficient', direction=direction, pml_threshold=pml_threshold)
        
        # vp vs. Re(omega)
        coefficient = np.abs(coefficient) #np.angle(coefficient)
        ax.plot(omega.real, coefficient, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        ax.set_xlim(omega.real.min(), omega.real.max())
        ax.set_xlabel('Re(omega)')
        ax.set_ylabel('|q|')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax

    def plot_excitability(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot excitability as a function of frequency, |e| vs. Re(omega)
        Parameters and Returns: see plot(...)
        """
        if len(self.excitability)==0:
            raise NotImplementedError('No excitability has been computed')
        
        # Initialization
        self._compute_if_necessary(direction, pml_threshold) #compute traveling direction and pml ratio if necessary
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        
        # Build concatenaded arrays
        _, omega, excitability = self._concatenate('excitability', direction=direction, pml_threshold=pml_threshold)
        
        # vp vs. Re(omega)
        excitability = np.abs(excitability) #np.angle(excitability)
        ax.plot(omega.real, excitability, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        ax.set_xlim(omega.real.min(), omega.real.max())
        ax.set_xlabel('Re(omega)')
        ax.set_ylabel('|e|')
        
        fig.tight_layout()
        # plt.show()  #let user decide whether he wants to interrupt the execution for display, or save to figure...
        return ax

    def plot_spectrum(self, index=0, ax=None, color="k",
                        marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot the spectrum, Im(k) vs. Re(k) computed for omega[index] (if the parameter is the frequency),
        or Im(omega) vs. Re(omega) for wavenumber[index] (if the parameter is the wavenumber)
        
        Parameters
        ----------
        index: parameter index
        ax: the matplotlib axe on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot

        Returns
        -------
        ax: the plot axe used for display
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.eigenvalues[index].real, self.eigenvalues[index].imag, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        if self.problem_type == "wavenumber":
            ax.set_xlabel('Re(omega)')
            ax.set_ylabel('Im(omega)')
        elif self.problem_type == "omega":
            ax.set_xlabel('Re(k)')
            ax.set_ylabel('Im(k)')
        fig.tight_layout()
        return ax

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
        Return all converged eigenpairs of the current EVP object (for internal use)
        Eigenvectors are stored in a PETSc dense matrix
        If two_sided is set to True, left eigensolutions are also included in the outputs, avoiding duplicates
        """        
        nconv = self.evp.getConverged()
        if self.target==0: #round to the nearest lower even integer to get as many positive-going modes as negatige-going ones
            nconv -= nconv%2 
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
        Return concatenated wavenumber and omega in the whole parameter range as 1D numpy arrays (for internal use)
        The arguments *args are optional strings to concatenate additional results (attribute names, e.g. 'energy_velocity')
        The parameter value (omega or wavenumber) is repeated as many as times as the number of eigenvalues
        If direction is specified (+1 or -1), eigenmodes traveling in the non-desired direction are filtered out 
        If pml_threshold is specified, eigenmodes such that pml_ratio<pml_threshold are filtered out
        If i is specified, then the function returns the results for the ith parameter only
        """
        argout = []
        index = slice(None) if i is None else slice(i, i+1)
        if self.problem_type == "wavenumber":
            wavenumber = np.repeat(self.wavenumber[index], [len(egv) for egv in self.eigenvalues[index]])
            omega = np.concatenate(self.eigenvalues[index])
        elif self.problem_type == "omega":
            omega = np.repeat(self.omega[index], [len(egv) for egv in self.eigenvalues[index]])
            wavenumber = np.concatenate(self.eigenvalues[index])
        argout.extend([wavenumber, omega])
        for arg in args:
            argout.append(np.concatenate(getattr(self, arg)[index])) 
        if direction is not None:
            traveling_direction = np.concatenate(self.traveling_direction[index])
            imode = traveling_direction==direction #indices of modes traveling in the desired direction
            argout = [argout[j][imode] for j in range(len(argout))]
        else:
            imode = slice(None)
        if pml_threshold is not None:
            pml_ratio = np.concatenate(self.pml_ratio[index])
            iphysical = pml_ratio[imode]>=pml_threshold #indices of physical modes (i.e. excluding PML modes)
            argout = [argout[j][iphysical] for j in range(len(argout))]
        return argout

    def _compute_if_necessary(self, direction, pml_threshold):
        """ Compute traveling direction and pml ratio if necessary before plot (for internal use) """
        if direction is not None and len(self.traveling_direction)==0:  #compute the traveling direction if not yet computed
            self.compute_traveling_direction()
        if pml_threshold is not None and len(self.pml_ratio)==0:  #compute the pml_ratio if not yet computed
            self.compute_pml_ratio()

    def _diag(self, vec):
        """ Return the PETSc diagonal matrix with diagonal entries given by vector vec (for internal use)"""
        diag = PETSc.Mat().createAIJ(len(vec), comm=self.comm)
        diag.setUp()
        diag.assemble()
        diag.setDiagonal(PETSc.Vec().createWithArray(vec, comm=self.comm))
        return diag

    def _dot_eigenvectors(self, i, eigenfield):
        """
        Return the dot product, mode by mode, between eigenvectors[i] (taking their conjugate) and a given eigenfield (for internal use)
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
