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
    left_eigenvectors: list of PETSc matrices
        list of left mode shapes (only computed if two_sided is set to True by user)
    eigenforces : list of PETSc matrices
        list of eigenforces (acces to components: see eigenvectors)
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
        eigenvectors, as well as left_eigenvectors if two_sided is True)
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
    plot_spectrum(index=0, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot the spectrum, Im(eigenvalues) vs. Re(eigenvalues), for the parameter index specified by the user
    compute_eigenforce():
        Compute the eigenforces for the whole parameter range and store them as an attribute (name: eigenforces)
    compute_energy_velocity():
        Compute the energy velocities for the whole parameter range and store them as an attribute (name: energy_velocity)
    compute_group_velocity():
        Compute the group velocities for the whole parameter range and store them as an attribute (name: energy_velocity)
        Left eigenvectors are required (two_sided must be set to True)
    compute_traveling_direction():
        Compute the traveling directions for the whole parameter range and store them as an attribute (name: traveling_direction)
    compute_pml_ratio():
        Compute the pml ratios for the whole parameter range and store them as an attribute (name: pml_ratio)
    compute_response_coefficient(F, amplitude_omega=None, amplitude_wavenumber=None, dof=None):
        Compute the response coefficients due to excitation vector F for the whole parameter range and store them as an attribute (name: coefficient)
    compute_response(dof, z, amplitude_omega=None, amplitude_wavenumber=None, plot=False):
        Compute the response at the degree of freedom dof and the axial coordinate z for the whole frequency range
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
        self.left_eigenvectors: list = []
        self.eigenforces: list = []
        self.energy_velocity: list = []
        self.group_velocity: list = []
        self.traveling_direction: list = []
        self.pml_ratio: list = []
        self.coefficient: list = []
        self.excitability: list = []
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
        self.evp.setTolerances(tol=1e-6, max_it=20)
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
        
        # Loop over the parameter
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        parameters = {"omega": self.omega, "wavenumber": self.wavenumber}[self.problem_type]
        print(f'Waveguide parameter: {self.problem_type} ({len(parameters)} iterations)')
        for i, parameter_value in enumerate(parameters):
            start = time.perf_counter()
            self.evp.setTarget(target(parameter_value))
            if self.problem_type=="wavenumber":
                 self.evp.setOperators(self.K0 + PETSc.ScalarType(1j*parameter_value)*(self.K1-K1T) + PETSc.ScalarType(parameter_value**2)*self.K2, self.M)
            elif self.problem_type == "omega":
                if not self.two_sided: #left eigenvectors not required -> PEP class is used
                    self.evp.setOperators([self.K0-PETSc.ScalarType(parameter_value)**2*self.M, PETSc.ScalarType(1j)*(self.K1-K1T), self.K2])
                else: #left eigenvectors are required -> linearize the quadratic evp and use EPS class (PEP class is not possible)
                    Zero = PETSc.Mat().createAIJ(self.M.getSize(), comm=self.comm)
                    Zero.setUp()
                    Zero.assemble()
                    Id = PETSc.Mat().createAIJ(self.M.getSize(), comm=self.comm)
                    Id.setUp()
                    Id.setDiagonal(self.M.createVecRight()+1)
                    Id.assemble()
                    coeff = 1 #self.K2.norm(norm_type=PETSc.NormType.FROBENIUS) #NORM_1, FROBENIUS (same as NORM_2 for vectors), INFINITY
                    A = self._build_block_matrix(-(self.K0-PETSc.ScalarType(parameter_value)**2*self.M), -PETSc.ScalarType(1j)*(self.K1-K1T), Zero, coeff*Id)
                    B = self._build_block_matrix(Zero, self.K2, coeff*Id, Zero)
                    #Note: the operators A and B below enable to get the eigenforces but increase computation time -> discarded...
                    #A = self._build_block_matrix(self.K0-PETSc.ScalarType(parameter_value)**2*self.M, Zero, -K1T, Id)
                    #B = self._build_block_matrix(-PETSc.ScalarType(1j)*self.K1, PETSc.ScalarType(1j)*Id, PETSc.ScalarType(1j)*self.K2, Zero)
                    self.evp.setOperators(A, B)
            self.evp.solve()
            #self.evp.errorView()
            #self.evp.valuesView()
            eigenvalues, eigenvectors = self._get_eigenpairs(two_sided=self.two_sided)
            self.eigenvalues.append(eigenvalues)
            self.eigenvectors.append(eigenvectors)
            print(f'Iteration {i}, elapsed time :{(time.perf_counter() - start):.2f}s')
            #self.evp.setInitialSpace(self.eigenvectors[-1]) #self.evp.setLeftInitialSpace(....) #try to use current modal basis to compute next, but may be only the first eigenvector...
        #print('\n---- SLEPc setup (based on last iteration) ----\n')
        #self.evp.view()
        K1T.destroy()
        if self.problem_type=="omega" and self.two_sided:
            Zero.destroy()
            Id.destroy()
            A.destroy()
            B.destroy()
        print('')

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
        ax.set_ylim(vg.min(), vg.max())
        ax.set_xlabel('Re(omega)')
        ax.set_ylabel('vg')
        
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

    def compute_eigenforces(self):
        """ Post-process the eigenforces F=(K1^T+1j*k*K2)*U for every mode in the whole parameter range"""
        if len(self.eigenforces)==len(self.eigenvalues):
            print('Eigenforces already computed')
            return
        start = time.perf_counter()
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        for i in range(len(self.eigenvectors)):
            wavenumber, _ = self._concatenate(i=i) #repeat parameter as many times as the number of eigenvalues
            self.eigenforces.append(K1T*self.eigenvectors[i]+PETSc.ScalarType(1j)*self.K2*self.eigenvectors[i]*self._diag(wavenumber))
        K1T.destroy()
        print(f'Computation of eigenforces, elapsed time : {(time.perf_counter() - start):.2f}s')

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
        for i in range(len(self.eigenvectors)):
            #repeat parameter as many times as the number of eigenvalues
            wavenumber, omega = self._concatenate(i=i)
            #time averaged kinetic energy
            E = 0.25*np.abs(omega**2)*np.real(self._dot_eigenvectors(i, self.M*self.eigenvectors[i])) 
            #add time averaged potential energy
            temp = (self.K0*self.eigenvectors[i] + 1j*self.K1*self.eigenvectors[i]*self._diag(wavenumber)
                    -1j*K1T*self.eigenvectors[i]*self._diag(wavenumber.conjugate()) + self.K2*self.eigenvectors[i]*self._diag(np.abs(wavenumber)**2))
            E = E + 0.25*np.real(self._dot_eigenvectors(i, temp))
            #time averaged complex Poynting vector (normal component)
            Pn = -1j*omega/2*self._dot_eigenvectors(i, self.eigenforces[i])
            #cross-section and time averaged energy velocity
            self.energy_velocity.append(np.real(Pn)/E)
        K1T.destroy()
        print(f'Computation of energy velocity, elapsed time : {(time.perf_counter() - start):.2f}s')
        
        # Warning for pml problems (integration restricted on the core is currently not possible)
        dofs_pml = np.nonzero(np.iscomplex(self.M.getDiagonal()[:]))[0]
        if dofs_pml.size!=0:
            print("Warning: the energy velocity is currently integrated on the whole domain including PML region")

    def compute_traveling_direction(self, delta=1e-2):
        """
        Post-process the traveling direction, +1 or -1, for every mode in the whole parameter range,
        using the sign of Im(k + 1j*delta/v) where delta is the imaginary shift used for analytical
        continuation of k, and v is the group velocity (or, if not available, the energy velocity)
        This criterion is based on the limiting absorption principle (theoretically, vg should be used
        instead of ve)
        """
        if len(self.traveling_direction)==len(self.eigenvalues):
            print('Traveling direction already computed')
            return
        if len(self.group_velocity)==0:
            if (self.target==0 or self.two_sided) and self.problem_type=='omega':
                self.compute_group_velocity() #compute the group velocity
            elif len(self.energy_velocity)==0: #or, if not available, compute the energy velocity
                self.compute_energy_velocity()
        start = time.perf_counter()
        for i in range(len(self.eigenvalues)):
            wavenumber, _ = self._concatenate(i=i)
            temp = delta/(self.energy_velocity[i] if len(self.group_velocity)==0 else self.group_velocity[i])
            temp[np.nonzero(np.abs(wavenumber.imag)+np.abs(temp)>np.abs(wavenumber.real))] = 0 #do not use the LAP if |Im(k)| + |delta/ve| is significant
            traveling_direction = np.sign((wavenumber+1j*temp).imag)
            self.traveling_direction.append(traveling_direction)
            #Check if any exponentially growing modes (in the numerical LAP, delta is user-defined, which could sometimes lead to wrong traveling directions)
            growing_modes = np.nonzero(wavenumber.imag*traveling_direction<0)[0]
            if len(growing_modes)!=0:
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
        for i in range(len(self.eigenvectors)):
            _, omega = self._concatenate(i=i)
            Ek = 0.25*np.abs(omega**2)*self._dot_eigenvectors(i, self.M*self.eigenvectors[i]) #"complex" kinetic energy
            self.pml_ratio.append(1-np.imag(Ek)/np.abs(Ek))
        print(f'Computation of pml ratio, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_group_velocity(self):
        """
        Post-process the group velocity, vg=1/Re(dk/domega) for every mode in the whole parameter range
        Opposite-going eigenvectors are required: to properly work, this function assumes that opposite-going modes
        have been sorted as consecutive pairs in eigenvalues and eigenvectors (this can be ensured by setting two_sided to True)
        """
        if len(self.group_velocity)==len(self.eigenvalues):
            print('Group velocity already computed')
            return
        if self.problem_type=='wavenumber' or (self.target!=0 and not self.two_sided):
            raise NotImplementedError('Group velocity computation is not possible: opposite-going modes cannot be paired for this kind of problem (check that the problem is of omega type and that target has been set to 0 or two_sided is True)')
        start = time.perf_counter()
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        for i in range(len(self.eigenvalues)):
            if (self.eigenvalues[i].size%2)!=0: #test if the number of eigensolutions is an even integer
                raise NotImplementedError(f'Group velocity computation is not possible at iteration {i}: the number of positive-going modes must be the same as the number of negative-going modes')
            wavenumber, omega = self._concatenate(i=i) #repeat parameter as many times as the number of eigenvalues
            group_velocity = np.zeros(self.eigenvalues[i].size)
            numerator = self.M*self.eigenvectors[i]
            denominator = 1j*(self.K1-K1T)*self.eigenvectors[i] + 2*self.K2*self.eigenvectors[i]*self._diag(wavenumber) #note: this line could probably be avoided if _compute_biorthogonality_factor() has already been done
            for mode in range(0,self.eigenvalues[i].size,2):
                uleft = self.eigenvectors[i].getColumnVector(mode+1)
                group_velocity[mode] = 1/np.real( 2*omega[mode]*numerator.getColumnVector(mode).tDot(uleft) / denominator.getColumnVector(mode).tDot(uleft) )
                group_velocity[mode+1] = -group_velocity[mode]
            self.group_velocity.append(group_velocity)
        K1T.destroy()
        print(f'Computation of group velocity, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_response_coefficient(self, F, amplitude_omega=None, amplitude_wavenumber=None, dof=None):
        """
        Computation of modal coefficients due to the excitation vector F for every mode in the whole omega range (omega denotes the angular frequency)
        Modal coefficients qm are defined from: U(z,omega) = sum qm(omega)*Um(omega)*exp(i*km*z), m=1...M,
        Opposite-going eigenvectors are required: to properly work, this function assumes that opposite-going modes
        have been sorted as consecutive pairs in eigenvalues and eigenvectors (this can be ensured by setting two_sided to True)
        Assumption: the source is centred at z=0
        Note: amplitude_omega and amplitude_wavenumber can be specified in compute_response(...) instead of compute_response_coefficient(...),
        but not in both functions in the same time (otherwise the excitation would be modulated twice)
        
        Parameters
        ----------
        F : PETSc vector
            SAFE excitation vector
        amplitude_omega : numpy.ndarray
            when specified, amplitude_omega is a vector of length omega  used to modulate F in terms of frequency (default: 1 for all frequencies)
        amplitude_wavenumber: python function
            when specified, amplitude_wavenumber is a python function used to modulate F in terms of wavenumber (example:
            amplitude_wavenumber = lambda x: np.sin(x), default: 1 for all wavenumbers, i.e. source localized at z=0)
        dof : int
            when specified, it calculates the modal contribution qm*Um at the degree of freedom dof (equal to the so-defined modal excitability
            if amplitude_omega and amplitude_wavenumber are equal to 1, i.e. unit force localized at a single degree of freedom),
            stored in the attribute excitability  
        """
        #Initialization
        if self.problem_type=='wavenumber':
            raise NotImplementedError('Response coefficient computation not implemented in case wavenumber is parameter')
        self.F = F
        self.coefficient = [] #re-initialized every time compute_response_coefficient(..) is executed (F is an input)
        self.excitability = [] #idem
        if amplitude_omega is None:
            amplitude_omega = np.ones(self.omega.size)
        if amplitude_wavenumber is None:
            amplitude_wavenumber = lambda k: 1+0*k
        
        #Check
        if len(amplitude_omega) != self.omega.size:
            raise NotImplementedError('The length of amplitude_omega must be equal to the length of omega')
        if dof is not None and not isinstance(dof, int):
            raise NotImplementedError('dof must be an integer')
        if len(self._biorthogonality_factor)==0: #biorthogonality normalization factor has not yet been computed
            self._compute_biorthogonality_factor()
        if len(self.traveling_direction)==0: #compute the traveling direction if not yet computed
            self.compute_traveling_direction()
        
        #Modal coefficients (loop over frequency)
        start = time.perf_counter()
        for i in range(self.omega.size):
            index = []
            coefficient = np.array(self.eigenvectors[i].copy().transpose()*F)
            [index.extend([j+1, j]) for j in range(0,len(self.eigenvalues[i]),2)] #index for assigning opposite-going modes (sorted as consecutive pairs)
            coefficient = coefficient[index]
            coefficient = coefficient/self._biorthogonality_factor[i]*self.traveling_direction[i]
            coefficient = coefficient*amplitude_omega[i]*amplitude_wavenumber(self.eigenvalues[i])
            self.coefficient.append(coefficient)
            if dof is not None:
                self.excitability.append(self.coefficient[-1]*self.eigenvectors[i][dof,:])
        print(f'Computation of response coefficient, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_response(self, dof, z, omega_index=None, amplitude_omega=None, amplitude_wavenumber=None, plot=False):
        """
        Post-process the response (modal expansion) at the degree of freedom dof and the axial coordinate z, for the whole frequency range
        The outputs are response, a numpy 2d array of size len(omega)*len(dof or z)
        dof and z cannot be both vectors, except if omega_index is specified or omega is scalar (single frequency computation): in that case,
        the array response is of size len(dof)*len(z), which can be useful to plot the whole field at a single frequency
        The response at each frequency omega is calculated from:
        U(z,omega) = sum qm(omega)*Um(omega)*exp(i*km*z), m=1...M,
        where z is the receiver position along the waveguide axis
        M is the number of modes traveling in the proper direction, positive if z is positive, negative if z is negative
        Assumption: the source is assumed to be centred at z=0
        Warning: the response calculation is only valid if z lies oustide the source region
        Note: amplitude_omega and amplitude_wavenumber can be specified in compute_response_coefficient(...) instead
        of compute_response(...), but not in both functions in the same time (otherwise the excitation would be modulated twice)
        
        Parameters
        ----------
        dof : numpy array of integer
            dof where the response is computed
        z : numpy array
            axial coordinate where the response is computed
        omega_index : int
            omega index to compute the response at a single frequency, allowing the consideration of multiple dof and z
        amplitude_omega : numpy.ndarray
            when specified, amplitude_omega is a vector of length omega  used to modulate F in terms of frequency (default: 1 for all frequencies)
        amplitude_wavenumber: python function
            when specified, amplitude_wavenumber is a python function used to modulate F in terms of wavenumber (example:
            amplitude_wavenumber = lambda x: np.sin(x), default: 1 for all wavenumbers, i.e. source localized at z=0)
        plot : bool
            if set to True, the magnitude and phase of response are plotted as a function of frequency
        
        Returns
        -------
        response : numpy array, the matrix response
        ax : when plot is set to True, ax[0] is the matplotlib axes used for magnitude, ax[1] is the matplotlib axes used for phase
        """
        
        #Initialization
        response = []
        dof = np.array(dof)
        z = np.array(z)
        if omega_index is None:
            omega_index = range(self.omega.size)
        else:
            if isinstance(omega_index, int): #single element special case
                omega_index = [omega_index]
            else:
                raise NotImplementedError('omega_index must be an integer')
        if amplitude_omega is None:
            amplitude_omega = np.ones(self.omega.size)
        
        #Check
        if self.problem_type=='wavenumber':
            raise NotImplementedError('Response computation not implemented in case wavenumber is parameter')
        if plot and len(omega_index)==1:
            raise NotImplementedError('Plot is not possible for a single frequency computation (please set plot to False)')
        if len(amplitude_omega) != self.omega.size:
            raise NotImplementedError('The length of amplitude_omega must be equal to the length of omega')
        if len(np.nonzero(z==0)[0])!=0:
            raise NotImplementedError('z cannot contain zero values (z must lie outside the source region)')
        if amplitude_wavenumber is not None:
            print('Reminder: z should lie outside the source region')
        if amplitude_wavenumber is None:
            amplitude_wavenumber = lambda k: 1+0*k
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
        direction = direction[0] if direction.size>1 else direction
        for i in omega_index:
            imode = np.nonzero(self.traveling_direction[i]==direction) #indices of modes traveling in the desired direction
            temp = self.coefficient[i][imode]*amplitude_omega[i]*amplitude_wavenumber(self.eigenvalues[i][imode])
            temp = (np.squeeze(self.eigenvectors[i][dof.tolist(),imode]) @ np.diag(temp)) @ np.exp(1j*np.outer(self.eigenvalues[i][imode], z)) #np.diag(temp) has many zeros (use sparse?)
            temp = temp.reshape(-1,1) #enforce vector to be column
            response.append(temp)
        response = np.squeeze(np.concatenate(response, axis=1)).T #numpy array of size len(self.omega)*len(dof or z)
        if len(omega_index)==1:
            response = response.reshape(dof.size,z.size) #numpy array of size len(dof)*len(z)
        print(f'Computation of response, elapsed time : {(time.perf_counter() - start):.2f}s')
        
        #Plots
        if plot:
            #Magnitude
            fig, ax_abs = plt.subplots(1, 1)
            ax_abs.plot(self.omega.real, np.abs(response), marker="o", markersize=2, linestyle="") #color="k"
            ax_abs.set_xlabel('Re(omega)')
            ax_abs.set_ylabel('|u|')
            fig.tight_layout()
            #Phase
            fig, ax_angle = plt.subplots(1, 1)
            ax_angle.plot(self.omega.real, np.angle(response), marker="o", markersize=2, linestyle="") #color="k"
            ax_angle.set_xlabel('Re(omega)')
            ax_angle.set_ylabel('arg(u)')
            fig.tight_layout()
            return response, [ax_abs, ax_angle]
        else:
            return response

    def _compute_biorthogonality_factor(self):
        """
        Post-process the biorthogonality normalization factors, Um^T*F-m - U-m^T*Fm),
        where m and -m denote opposite-going modes, for the whole parameter range and store them as
        an attribute (name: _biorthogonality_factor)
        To properly work, this method assumes that opposite-going modes have been sorted as
        consecutive pairs in eigenvalues and eigenvectors (this can be ensured by setting two_sided to True)
        """
        tol = 1e-6
        if self.problem_type=='wavenumber' or (self.target!=0 and not self.two_sided):
            raise NotImplementedError('Computation of biorthogonality factor is not possible: opposite-going modes cannot be paired for this kind of problem (check that the problem is of omega type and that target has been set to 0 or two_sided is True)')
        if len(self._biorthogonality_factor)!=0:
            print('Biorthogonality factor already computed')
            return
        if len(self.eigenforces)==0: #compute the eigenforces if not yet computed      
            self.compute_eigenforces()
        start = time.perf_counter()
        index = range(self.eigenvectors[0].getSize()[0])
        for i in range(len(self.eigenvalues)):
            if (self.eigenvalues[i].size%2)!=0: #test if the number of eigensolutions is an even integer
                raise NotImplementedError(f'Computation of biorthogonality factor is not possible at iteration {i}: the number of positive-going modes must be the same as the number of negative-going modes')
            #Quick check that opposite-going modes seem to be correctly paired
            if any(np.abs((self.eigenvalues[i][0::2]+self.eigenvalues[i][1::2])/self.eigenvalues[i][0::2])>tol):
                print(f'Warning at iteration {i}: opposite-going modes may be not correctly paired (tolerance exceeded)')
            #Biorthogonality normalization factor
            for mode in range(0,self.eigenvalues[i].size,2):
                U = self.eigenvectors[i].getColumnVector(mode)
                F = self.eigenforces[i].getColumnVector(mode)
                Uopposite = self.eigenvectors[i].getColumnVector(mode+1)
                Fopposite = self.eigenforces[i].getColumnVector(mode+1)
                self._biorthogonality_factor.append(U.tDot(Fopposite) - Uopposite.tDot(F))
                self._biorthogonality_factor.append(-self._biorthogonality_factor[-1]) #the opposite-going mode
        print(f'Computation of biorthogonality normalization factor, elapsed time : {(time.perf_counter() - start):.2f}s')

    def _check_biorthogonality(self, i):
        """ Return and plot, for the ith parameter, the Modal Assurance Criterion (MAC) matrix based on the (bi)-orthogonality relation (for internal use)"""
        if len(self.eigenforces)==0: #the eigenforces has not yet been computed
            print('Eigenforces have not yet been computed')
            return
        if self.problem_type == "wavenumber":
            biorthogonality = self.eigenvectors[i].copy().hermitianTranspose()*self.M*self.eigenvectors[i] #hyp: K0, K1, K2, M and eigenvalues must be real here!
            # Warning for lossy problems
            dofs_complex = np.nonzero(np.iscomplex(self.K2.getDiagonal()[:]))[0]
            if dofs_complex.size!=0:
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
            imode = np.nonzero(traveling_direction==direction)[0] #indices of modes traveling in the desired direction
            argout = [argout[j][imode] for j in range(len(argout))]
        else:
            imode = slice(None)
        if pml_threshold is not None:
            pml_ratio = np.concatenate(self.pml_ratio[index])
            iphysical = np.nonzero(pml_ratio[imode]>=pml_threshold)[0] #indices of physical modes (i.e. excluding PML modes)
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
