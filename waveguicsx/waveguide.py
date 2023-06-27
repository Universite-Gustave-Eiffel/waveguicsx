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
    (K1-omega**2*M + 1j*k*(K2-K2^T) + k**2*K3)*U=0
    The varying parameter can be the angular frequency omega or the wavenumber k.
    In the former case, the eigenvalue is k, while in the latter case, the eigenvalue is omega**2.
    
    Example:
    import waveguicsx
    param = np.arange(0.1, 2, 0.1)
    waveguide = waveguicsx.Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
    waveguide.set_parameters(wavenumber=param) #or: waveguide.setParameters(omega=param)
    waveguide.solve(nev)
    waveguide.plot()
    plt.show()
    
    Attributes
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI communicator (parallel processing)
    M, K1, K2, K3 : petsc4py.PETSc.Mat
        SAFE matrices
    problem_type : str
        problem_type is "omega" if the varying parameter is omega, "wavenumber" if this is k
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
    energy_velocity : list of numpy arrays
        list of energy velocity (access to component: see eigenvalues)
    traveling_direction : list of numpy arrays
        list of traveling_direction (access to component: see eigenvalues)
    pml_ratio : list of numpy arrays
        list of pml ratio, useful for filtering out PML modes (access to component: see eigenvalues)
    
    Methods
    -------
    set_parameters(omega=None, wavenumber=None):
        Set problem type (problem_type), the parameter range (omega or wavenumber) as well as default parameters of SLEPc eigensolver (evp)
    solve(nev=1, target=0):
        Solve the eigenvalue problem repeatedly for the parameter range, solutions are stored as attributes (names: eigenvalues, eigenvectors)
    plot(ax=None, color="k",  marker="o", markersize=2, linestyle="", **kwargs):
        Plot dispersion curves Re(omega) vs. Re(wavenumber) using matplotlib
    plot_phase_velocity(ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot phase velocity dispersion curves, vp=Re(omega)/Re(wavenumber) vs. Re(omega)
    plot_attenuation(ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot attenuation dispersion curves, Im(wavenumber) vs. Re(omega) if omega is the parameter,
        or Im(omega) vs. Re(omega) if wavenumber is the parameter
    plot_energy_velocity(ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot energy velocity dispersion curves, ve vs. Re(omega)
    plot_spectrum(index=0, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot the spectrum, Im(eigenvalues) vs. Re(eigenvalues), for the parameter index specified by the user
    compute_eigenforce():
        Compute the eigenforces for the whole parameter range and store them as an attribute (name: eigenforces)
    compute_energy_velocity():
        Compute the energy velocity for the whole parameter range and store them as an attribute (name: energy_velocity)
    compute_traveling_direction():
        Compute the traveling_direction for the whole parameter range and store them as an attribute (name: traveling_direction)
    compute_pml_ratio():
        Compute the pml ratio for the whole parameter range and store them as an attribute (name: pml_ratio)
    """
    def __init__(self, comm:'_MPI.Comm', M:PETSc.Mat, K1:PETSc.Mat, K2:PETSc.Mat, K3:PETSc.Mat):
        """
        Constructor
        
        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator (parallel processing)
        M, K1, K2, K3 : petsc4py.PETSc.Mat
            SAFE matrices
        """
        self.comm = comm
        self.M = M
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3

        # Set the default values for the internal attributes used in this class
        self.problem_type: str = ""  # "wavenumber" or "omega"
        self.omega: Union[np.ndarray, None] = None
        self.wavenumber: Union[np.ndarray, None] = None
        self.evp: Union[SLEPc.PEP, SLEPc.EPS, None] = None
        self.eigenvalues: list = []
        self.eigenvectors: list = []
        self.eigenforces: list = []
        self.energy_velocity: list = []
        self.traveling_direction: list = []
        self.pml_ratio: list = []
        
        # Print the number of degrees of freedom
        print(f'Total number of degrees of freedom: {self.M.size[0]}')

    def set_parameters(self, omega: Union[np.ndarray, None]=None, wavenumber:Union[np.ndarray, None]=None):
        """
        Set the parameter range (omega or wavenumber) as well as default parameters of the SLEPc eigensolver (evp)
        The user must specify the parameter omega or wavenumber, but not both
        This method generates the attributes omega (or wavenumber) and evp
        After this method call, different SLEPc parameters can be set by changing the attribute evp manually
        
        Parameters
        ----------
        omega or wavenumber : numpy.ndarray
            the parameter range specified by the user
        """
        if not (wavenumber is None) ^ (omega is None):
            raise NotImplementedError('Please specify omega or wavenumber (and not both)!')

        # The parameter is the frequency omega, the eigenvalue is the wavenumber k
        if wavenumber is None:
            self.problem_type = "omega"
            if isinstance(omega, (float, int, complex)): #single element special case
                omega = [omega]
            self.omega = omega
            # Setup the SLEPc solver for the quadratic eigenvalue problem
            self.evp = SLEPc.PEP()
            self.evp.create(comm=self.comm)
            self.evp.setProblemType(SLEPc.PEP.ProblemType.GENERAL) #note: for the undamped case, HERMITIAN is possible with QARNOLDI and TOAR but surprisingly not faster
            self.evp.setType(SLEPc.PEP.Type.LINEAR) #note: the computational speed of LINEAR, QARNOLDI and TOAR seems to be almost identical
            self.evp.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_IMAGINARY)

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
        
        # Setup common to EPS and PEP
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
        # Eigensolver setup
        self.evp.setDimensions(nev=nev)
        if isinstance(target, (float, int, complex)): #redefine target as a constant function if target is given as a number
            target_constant = target
            target = lambda parameter_value: target_constant
        
        # Loop over the parameter
        K2T = self.K2.copy().transpose() #K2^T is stored before loop (faster computations)
        parameters = {"omega": self.omega, "wavenumber": self.wavenumber}[self.problem_type]
        print(f'Waveguide parameter: {self.problem_type} ({len(parameters)} iterations)')
        for i, parameter_value in enumerate(parameters):
            start = time.perf_counter()
            self.evp.setTarget(target(parameter_value))
            if self.problem_type=="wavenumber":
                 self.evp.setOperators(self.K1 + PETSc.ScalarType(1j*parameter_value)*(self.K2-K2T) + PETSc.ScalarType(parameter_value**2)*self.K3, self.M)
            elif self.problem_type == "omega":
                self.evp.setOperators([self.K1-PETSc.ScalarType(parameter_value)**2*self.M, PETSc.ScalarType(1j)*(self.K2-K2T), self.K3])
            self.evp.solve()
            #self.evp.errorView()
            #self.evp.valuesView()
            self.eigenvalues.append(self._get_eigenvalues())
            self.eigenvectors.append(self._get_eigenvectors())
            print(f'Iteration {i}, elapsed time :{(time.perf_counter() - start):.2f}s')
            #self.evp.setInitialSpace(self.eigenvectors[-1]) #try to use current modal basis to compute next, but may be only the first eigenvector...
        #print('\n---- SLEPc setup (based on last iteration) ----\n')
        #self.evp.view()
        K2T.destroy()
        print('')

    def plot(self, direction=None, pml_threshold=None, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot dispersion curves Re(omega) vs. Re(wavenumber)
        
        Parameters
        ----------
        direction: +1 for positive-going modes, -1 for negative-going modes, None for plotting all modes
        pml_threshold: threshold to filter out PML modes (i.e. such that pml_ratio<pml_threshold)
        ax: the matplotlib axes on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot

        Returns
        -------
        ax: the matplotlib axes used for display
        """
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
        """ Post-process the eigenforces F=(K2^T+1j*k*K3)*U for every mode in the whole parameter range"""
        start = time.perf_counter()
        self.eigenforces = []
        K2T = self.K2.copy().transpose() #K2^T is stored before loop (faster computations)
        for i in range(len(self.eigenvectors)):
            wavenumber, _ = self._concatenate(i=i) #repeat parameter as many times as the number of eigenvalues
            self.eigenforces.append(K2T*self.eigenvectors[i]+PETSc.ScalarType(1j)*self.K3*self.eigenvectors[i]*self._diag(wavenumber))
        K2T.destroy()
        print(f'Computation of eigenforces, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_energy_velocity(self):
        """ Post-process the energy velocity ve for every mode in the whole parameter range"""
        # Compute the eigenforces if not yet computed
        if len(self.eigenforces)==0:
            self.compute_eigenforces()
        
        # Energy velocity, integration on the whole domain
        start = time.perf_counter()
        self.energy_velocity = []
        K2T = self.K2.copy().transpose() #K2^T is stored before loop (faster computations)
        for i in range(len(self.eigenvectors)):
            #repeat parameter as many times as the number of eigenvalues
            wavenumber, omega = self._concatenate(i=i)
            #time averaged kinetic energy
            E = 0.25*np.abs(omega**2)*np.real(self._dot_eigenvectors(i, self.M*self.eigenvectors[i])) 
            #add time averaged potential energy
            temp = (self.K1*self.eigenvectors[i] + 1j*self.K2*self.eigenvectors[i]*self._diag(wavenumber)
                    -1j*K2T*self.eigenvectors[i]*self._diag(wavenumber.conjugate()) + self.K3*self.eigenvectors[i]*self._diag(np.abs(wavenumber)**2))
            E = E + 0.25*np.real(self._dot_eigenvectors(i, temp))
            #time averaged complex Poynting vector (normal component)
            Pn = -1j*omega/2*self._dot_eigenvectors(i, self.eigenforces[i])
            #cross-section and time averaged energy velocity
            self.energy_velocity.append(np.real(Pn)/E)
        K2T.destroy()
        print(f'Computation of energy velocity, elapsed time : {(time.perf_counter() - start):.2f}s')
        
        # Warning for pml problems (integration restricted on the core is currently not possible)
        dofs_pml = np.nonzero(np.iscomplex(self.M.getDiagonal()[:]))[0]
        if dofs_pml.size!=0:
            print("Warning: the energy velocity is currently integrated on the whole domain including PML region")

    def compute_traveling_direction(self, delta=1e-2):
        """
        Post-process the traveling direction, +1 or -1, for every mode in the whole parameter range,
        using the sign of Im(k + 1j*delta/ve) where delta is the imaginary shift used for analytical
        continuation of k
        This criterion is based on the limiting absorption principle, although theoretically,
        vg should be used instead of ve
        """
        if len(self.energy_velocity)==0:  #compute the energy velocity if not yet computed
            self.compute_energy_velocity()
        start = time.perf_counter()
        self.traveling_direction = []
        for i in range(len(self.eigenvalues)):
            wavenumber, _ = self._concatenate(i=i)
            temp = delta/self.energy_velocity[i]
            temp[np.nonzero(np.abs(wavenumber.imag)+np.abs(temp)>np.abs(wavenumber.real))] = 0 #do not use the LAP if |Im(k)| + |delta/ve| is significant
            wavenumlap = wavenumber + 1j*temp
            self.traveling_direction.append(np.sign(wavenumlap.imag))
        print(f'Determination of traveling direction, elapsed time : {(time.perf_counter() - start):.2f}s')

    def compute_pml_ratio(self):
        """
        Post-process the pml ratio (useful to filter out PML mode), given by 1-Im(Ek)/|Ek| where Ek denotes
        the "complex" kinetic energy, for every mode in the whole parameter range        
        Note that in the absence of PML, the pml ratio is equal to 1
        """
        start = time.perf_counter()
        self.pml_ratio = []
        for i in range(len(self.eigenvectors)):
            _, omega = self._concatenate(i=i) #repeat parameter as many times as the number of eigenvalues
            Ek = 0.25*np.abs(omega**2)*self._dot_eigenvectors(i, self.M*self.eigenvectors[i]) #"complex" kinetic energy
            self.pml_ratio.append(1-np.imag(Ek)/np.abs(Ek))
        print(f'Computation of pml ratio, elapsed time : {(time.perf_counter() - start):.2f}s')

    def track_mode(self):
        """ Track a desired mode """
        #Future works
    
    def compute_response(self):
        """ Post-process the forced response """
        #Future works

    def _check_biorthogonality(self, i):
        """ Return and plot, for the ith parameter, the Modal Assurance Criterion (MAC) matrix based on the (bi)-orthogonality relation (for internal use)"""
        if len(self.eigenforces)==0: #the eigenforces has not yet been computed
            print('Eigenforces have not yet been computed')
            return
        if self.problem_type == "wavenumber":
            biorthogonality = self.eigenvectors[i].copy().hermitianTranspose()*self.M*self.eigenvectors[i] #hyp: K1, K2, K3, M and eigenvalues must be real here!
            # Warning for lossy problems
            dofs_complex = np.nonzero(np.iscomplex(self.K3.getDiagonal()[:]))[0]
            if dofs_complex.size!=0:
                print("Warning: the orthogonality relation implemented is currently valid for real matrices only (lossless problems)")
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

    def _get_eigenvectors(self):
        """ Return all converged eigenvectors of the current EVP object in a PETSc dense matrix (for internal use) """
        nconv = self.evp.getConverged()
        v = self.evp.getOperators()[0].createVecRight()
        eigenvectors = PETSc.Mat().create(comm=self.comm)
        eigenvectors.setType("dense")
        eigenvectors.setSizes([v.getSize(), nconv])
        eigenvectors.setFromOptions()
        eigenvectors.setUp()
        for i in range(nconv):
            self.evp.getEigenpair(i, v) #To get left eigenvectors: self.evp.getLeftEigenvector(i, v)
            eigenvectors.setValues(range(v.getSize()), i, v)
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
        if i is None:
            index = slice(None)
        else:
            index = slice(i, i+1)
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
        diag.setDiagonal(PETSc.Vec().createWithArray(vec))
        return diag

    def _dot_eigenvectors(self, i, eigenfield):
        """
        Return the dot product, mode by mode, between eigenvectors[i] (taking the conjugate) and a given eigenfield (for internal use)
        The matrix eigenfield must have the same size as eigenvectors[i]
        """
        res = []
        for mode in range(self.eigenvectors[i].getSize()[1]):
            res.append(eigenfield.getColumnVector(mode).dot(self.eigenvectors[i].getColumnVector(mode))) #or: np.vdot(self.eigenvectors[i][:,mode],eigenfield[:,mode]))
        res = np.array(res)
        return res