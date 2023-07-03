from typing import Union, List
from petsc4py import PETSc
from slepc4py import SLEPc

import matplotlib.pyplot as plt
import numpy as np
import time

# TO KEEP IN MIND: memory usage (-> check with A is B, with import sys; sys.getsizeof(eigenvectors), or with memory profiler, etc.)

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
    problem_type : str
        problem_type is "omega" if the varying parameter is omega, "wavenumber" if this is k
    two_sided : bool
        if True, left eigenvectors will be also computed (otherwise, only right eigenvectors are computed)
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
        list of pml ratio, useful for filtering out PML modes (access to component: see eigenvalues)
    
    Methods
    -------
    set_parameters(omega=None, wavenumber=None, two_sided=False):
        Set problem type (problem_type), the parameter range (omega or wavenumber) as well as default parameters of SLEPc eigensolver (evp)
        Set two_sided to True if left eigenvectors are needed (e.g. for normalization or group velocity computation)
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
        Compute the energy velocity for the whole parameter range and store them as an attribute (name: energy_velocity)
    compute_group_velocity():
        Compute the group velocity for the whole parameter range and store them as an attribute (name: energy_velocity)
        Left eigenvectors are required (two_sided must be set to True)
    compute_traveling_direction():
        Compute the traveling_direction for the whole parameter range and store them as an attribute (name: traveling_direction)
    compute_pml_ratio():
        Compute the pml ratio for the whole parameter range and store them as an attribute (name: pml_ratio)
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

        # Set the default values for the internal attributes used in this class
        self.problem_type: str = ""  # "wavenumber" or "omega"
        self.omega: Union[np.ndarray, None] = None
        self.wavenumber: Union[np.ndarray, None] = None
        self.two_sided = None
        self.evp: Union[SLEPc.PEP, SLEPc.EPS, None] = None
        self.eigenvalues: list = []
        self.eigenvectors: list = []
        self.left_eigenvectors: list = []
        self.eigenforces: list = []
        self.energy_velocity: list = []
        self.group_velocity: list = []
        self.traveling_direction: list = []
        self.pml_ratio: list = []
        
        # Print the number of degrees of freedom
        print(f'Total number of degrees of freedom: {self.M.size[0]}')

    def set_parameters(self, omega: Union[np.ndarray, None]=None, wavenumber:Union[np.ndarray, None]=None, two_sided=False):
        """
        Set the parameter range (omega or wavenumber) as well as default parameters of the SLEPc eigensolver (evp)
        The user must specify the parameter omega or wavenumber, but not both
        This method generates the attributes omega (or wavenumber) and evp
        After this method call, different SLEPc parameters can be set by changing the attribute evp manually
        Set two_sided=True for solving left eigenvectors also
        
        Parameters
        ----------
        omega or wavenumber : numpy.ndarray
            the parameter range specified by the user
        two_sided : bool
            False if left eigenvectiors are not needed, True if they must be solved also (e.g. for mode normalization and group velocity)
        """
        if len(self.eigenvalues)!=0:
            print('Eigenvalue problem already solved (re-initialize the Waveguide object to solve a new eigenproblem)')
            return
        if not (wavenumber is None) ^ (omega is None):
            raise NotImplementedError('Please specify omega or wavenumber (and not both)')
        
        # The parameter is the frequency omega, the eigenvalue is the wavenumber k
        if wavenumber is None:
            self.problem_type = "omega"
            if isinstance(omega, (float, int, complex)): #single element special case
                omega = [omega]
            self.omega = omega
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
            self.problem_type = "wavenumber"
            if isinstance(wavenumber, (float, int, complex)): #single element special case
                wavenumber = [wavenumber]
            self.wavenumber = wavenumber
            # Setup the SLEPc solver for the generalized eigenvalue problem
            self.evp = SLEPc.EPS()
            self.evp.create(comm=self.comm)
            self.evp.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #note: GHEP (generalized Hermitian) is surprinsingly a little bit slower...
            self.evp.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #note: ARNOLDI also works although slightly slower
            self.evp.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            self.evp.setTwoSided(two_sided)
        
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
                    A = self._build_block_matrix(-(self.K0-PETSc.ScalarType(parameter_value)**2*self.M), -PETSc.ScalarType(1j)*(self.K1-K1T), Zero, Id)
                    B = self._build_block_matrix(Zero, self.K2, Id, Zero)
                    #Note: the operators A and B below enable to get the eigenforces but increase computation time -> discarded...
                    #A = self._build_block_matrix(self.K0-PETSc.ScalarType(parameter_value)**2*self.M, Zero, -K1T, Id)
                    #B = self._build_block_matrix(-PETSc.ScalarType(1j)*self.K1, PETSc.ScalarType(1j)*Id, PETSc.ScalarType(1j)*self.K2, Zero)
                    self.evp.setOperators(A, B)
            self.evp.solve()
            #self.evp.errorView()
            #self.evp.valuesView()
            self.eigenvalues.append(self._get_eigenvalues())
            self.eigenvectors.append(self._get_eigenvectors())
            if self.two_sided: #get left eigenvectors also
                self.left_eigenvectors.append(self._get_eigenvectors(side='left'))
            print(f'Iteration {i}, elapsed time :{(time.perf_counter() - start):.2f}s')
            #self.evp.setInitialSpace(self.eigenvectors[-1]) #try to use current modal basis to compute next, but may be only the first eigenvector...
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
            if self.two_sided:
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
        Post-process the group velocity, vg=Re(dk/domega) for every mode in the whole parameter range
        Left eigenvectors are required
        """
        if len(self.group_velocity)==len(self.eigenvalues):
            print('Group velocity already computed')
            return
        if not self.two_sided:
            raise NotImplementedError('The attribute two_sided is False, please specify set_parameter(..., two_sided=True): left eigenvectors are needed to compute the group velocity')        
        start = time.perf_counter()
        K1T = self.K1.copy().transpose() #K1^T is stored before loop (faster computations)
        for i in range(len(self.eigenvectors)):
            #repeat parameter as many times as the number of eigenvalues
            wavenumber, omega = self._concatenate(i=i)
            #numerator
            numerator = 2*omega*self._dot_eigenvectors(i, self.M*self.eigenvectors[i], side='left') 
            #denominator
            denominator = 1j*(self.K1-K1T)*self.eigenvectors[i] + 2*self.K2*self.eigenvectors[i]*self._diag(wavenumber)
            denominator = self._dot_eigenvectors(i, denominator, side='left')
            #group velocity
            self.group_velocity.append(1/np.real(numerator/denominator))
        K1T.destroy()
        print(f'Computation of group velocity, elapsed time : {(time.perf_counter() - start):.2f}s')

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
        return biorthogonality

    def _get_eigenvalues(self):
        """ Return all converged eigenvalues of the current EVP object (for internal use) """
        eigenvalues = np.array([self.evp.getEigenpair(i) for i in range(self.evp.getConverged())])
        if self.problem_type=="wavenumber":
            eigenvalues = np.sqrt(eigenvalues)
        return eigenvalues

    def _get_eigenvectors(self, side='right'):
        """
        Return all converged eigenvectors of the current EVP object in a PETSc dense matrix (for internal use)
        Return left eigenvectors if side is set to 'left', right eigenvectors otherwise
        """
        nconv = self.evp.getConverged()
        v = self.evp.getOperators()[0].createVecRight()
        index = range(v.getSize()) if v.getSize()==self.M.getSize()[0] else range(int(v.getSize()/2)) #1/2 in case of externally linearized quadratic evp
        eigenvectors = PETSc.Mat().create(comm=self.comm)
        eigenvectors.setType("dense")
        eigenvectors.setSizes([self.M.getSize()[0], nconv])
        eigenvectors.setFromOptions()
        eigenvectors.setUp()
        for i in range(nconv):
            if side != 'left':
                self.evp.getEigenpair(i, v)
            else:
                self.evp.getLeftEigenvector(i, v)
                v.conjugate() #cancel the conjugate internally applied by SLEPc
            eigenvectors.setValues(index, i, v[index])
        eigenvectors.assemble()
        return eigenvectors

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
        """ Return the diagonal matrix from a given vector vec (for internal use)"""
        diag = PETSc.Mat().createAIJ(len(vec), comm=self.comm)
        diag.setUp()
        diag.assemble()
        diag.setDiagonal(PETSc.Vec().createWithArray(vec, comm=self.comm))
        return diag

    def _dot_eigenvectors(self, i, eigenfield, side='right'):
        """
        Return the dot product, mode by mode, between eigenvectors[i] and a given eigenfield (for internal use)
        The matrix eigenfield must have the same size as eigenvectors[i]
        Dot product using left eigenvectors (without taking their conjugate) if side is set to 'left',
        right eigenvectors (taking their conjugate) otherwise
        """
        res = []
        for mode in range(self.eigenvectors[i].getSize()[1]):
            if side != 'left':
                res.append(eigenfield.getColumnVector(mode).dot(self.eigenvectors[i].getColumnVector(mode))) #or: np.vdot(self.eigenvectors[i][:,mode],eigenfield[:,mode]))
            else:
                res.append(eigenfield.getColumnVector(mode).tDot(self.left_eigenvectors[i].getColumnVector(mode)))
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
