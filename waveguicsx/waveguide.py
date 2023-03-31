from typing import Union, List
from petsc4py import PETSc
from slepc4py import SLEPc

import matplotlib.pyplot as plt
import numpy as np
import time

# TO KEEP IN MIND:
# - eigenvectors are now stored as a PETSc matrix, which will allow fast matrix multiplications in the future (e.g. self.M * eigenvectors[ik])
# - memory usage (-> check with A is B, with import sys; sys.getsizeof(eigenvectors), or with memory profiler, etc.)

class Waveguide:
    """
    A class for solving waveguide problems (based on SLEPc eigensolver)
    
    The problem must be based on the so-called SAFE (Semi-Analytical Finite Element) formulation:
    (K1-omega**2*M + 1j*k*K2 + k**2*K3)*U=0
    The varying parameter can be the angular frequency omega or the wavenumber k.
    In the former case, the eigenvalue is k, while in the latter case, the eigenvalue is omega**2.
    
    Example:
    import waveguicsx
    param = np.arange(0.1, 2, 0.1)
    waveguide = waveguicsx.Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
    waveguide.set_parameters(wavenumber=param) #or: waveguide.setParameters(omega=param)
    waveguide.solve(nev)
    waveguide.plot_dispersion()
    plt.show()
    
    Attributes
    ----------
    comm : mpi4py.MPI.Intracomm
        MPI communicator (parallel processing)
    M, K1, K2, K3 : petsc4py.PETSc.Mat
        SAFE matrices
    pb_type : str
        pb_type is "omega" if the varying parameter is omega, "wavenumber" if this is k
    omega or wavenumber : numpy.ndarray
        the parameter range specified by the user (see method setParameters)
    evp : PEP or EPS instance (SLEPc object)
        eigensolver parameters (EPS if pb_type is "wavenumber", PEP otherwise)
    eigenvalues : list of numpy arrays
        list of wavenumbers or angular frequencies
        access to components with eigenvalues[ip][imode] (ip: parameter index, imode: mode index)
    eigenvectors : list of PETSc matrices
        list of mode shapes
        access to components with eigenvectors[ik][idof,imode] (ip: parameter index, imode: mode index, idof: dof index)
        or eigenvectors[ik].getColumnVector(imode)
    
    Methods
    -------
    set_parameters(omega=None, wavenumber=None):
        Set problem type (pb_type), the parameter range (omega or wavenumber) as well as default parameters of SLEPc eigensolver (evp)
    solve(nev=1, target=0):
        Solve the eigenvalue problem repeatedly for the parameter range, solutions are stored as attributes (eigenvalues, eigenvectors)
    plot_dispersion(axs=None, color="k",  marker="o", markersize=2, linestyle="", **kwargs):
        Plot dispersion curves (based on multiple subplot matplotlib)
    plot_spectrum(index=0, ax=None, color="k", marker="o", markersize=2, linestyle="", **kwargs):
        Plot the spectrum, Im(eigenvalues) vs. Re(eigenvalues), for the parameter index specified by the user
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
        self.pb_type: str = ""  # "wavenumber" or "omega"
        self.omega: Union[np.ndarray, None] = None
        self.wavenumber: Union[np.ndarray, None] = None
        self.evp: Union[SLEPc.PEP, SLEPc.EPS, None] = None
        self.eigenvalues: list = []
        self.eigenvectors: list = []
        
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
            self.pb_type = "omega"
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
            self.pb_type = "wavenumber"
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
        parameters = {"omega": self.omega, "wavenumber": self.wavenumber}[self.pb_type]
        print(f'Waveguide parameter: {self.pb_type} ({len(parameters)} iterations)')
        for i, parameter_value in enumerate(parameters):
            start = time.perf_counter()
            self.evp.setTarget(target(parameter_value))
            if self.pb_type=="wavenumber":
                 self.evp.setOperators(self.K1 + PETSc.ScalarType(1j*parameter_value)*self.K2 + PETSc.ScalarType(parameter_value**2)*self.K3, self.M)
            elif self.pb_type == "omega":
                self.evp.setOperators([self.K1-PETSc.ScalarType(parameter_value)**2*self.M, PETSc.ScalarType(1j)*self.K2, self.K3])
            self.evp.solve()
            #self.evp.errorView()
            #self.evp.valuesView()
            self.eigenvalues.append(self._get_eigenvalues())
            self.eigenvectors.append(self._get_eigenvectors())
            print(f'Iteration {i}, elapsed time :{(time.perf_counter() - start):.2f}s')
            #self.evp.setInitialSpace(self.eigenvectors[-1]) #try to use current modal basis to compute next, but may be only the first eigenvector...
        print('\n---- SLEPc setup (based on last iteration) ----\n')
        self.evp.view()
        print('')
        
    def plot_dispersion(self, axs=None, color="k",
                        marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot dispersion curves (based on multiple subplot matplotlib)
        
        Parameters
        ----------
        axs: the matplotlib axe (multiple subplot) on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot

        Returns
        -------
        axs: the multiple subplot axe used for display
        """
        if axs is None:
            fig, axs = plt.subplots(2, 2)
        
        if self.pb_type == "wavenumber":
            wavenumber = np.repeat(self.wavenumber, [len(egv) for egv in self.eigenvalues])
            omega = np.concatenate(self.eigenvalues)

        elif self.pb_type == "omega":
            omega = np.repeat(self.omega, [len(egv) for egv in self.eigenvalues])
            wavenumber = np.concatenate(self.eigenvalues)
        
        # Re(omega) vs. Re(k) 
        axs[0, 0].plot(wavenumber.real, omega.real, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        axs[0, 0].set_xlim(wavenumber.real.min(), wavenumber.real.max())
        axs[0, 0].set_ylim(omega.real.min(), omega.real.max())
        axs[0, 0].set_xlabel('Re(k)')
        axs[0, 0].set_ylabel('Re(omega)')
        
        # vp vs. Re(omega)
        vp = omega.real/wavenumber.real
        axs[0, 1].plot(omega.real, vp, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        axs[0, 1].set_xlim(omega.real.min(), omega.real.max())
        ylim = omega.real.max()/np.abs(wavenumber.real).mean()
        axs[0, 1].set_ylim(-ylim, ylim)
        axs[0, 1].set_xlabel('Re(omega)')
        axs[0, 1].set_ylabel('vp')
        
        # Im(k) vs Re(omega)
        axs[1, 0].plot(omega.real, wavenumber.imag, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        axs[1, 0].sharex(axs[0, 1])
        axs[1, 0].set_ylim(wavenumber.imag.min(), wavenumber.imag.max()+1e-6)
        axs[1, 0].set_xlabel('Re(omega)')
        axs[1, 0].set_ylabel('Im(k)')
        
        fig.tight_layout()
        # plt.show()  # let user decide whether he wants to interrupt the execution for display, or save to figure...
        return axs

    def plot_spectrum(self, index=0, ax=None, color="k",
                        marker="o", markersize=2, linestyle="", **kwargs):
        """
        Plot the spectrum, Im(k) vs. Re(k) computed for omega[index] (if the parameter is the frequency),
        or Im(omega) vs. Re(omega) for wavenumber[index] (if the parameter is the wavenumber)
        
        Parameters
        ----------
        index: 
        ax: the matplotlib axe on which to plot data (created if None)
        color: str, marker: str, markersize: int, linestyle: str, **kwargs are passed to ax.plot

        Returns
        -------
        ax: the plot axe used for display
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.eigenvalues[index].real, self.eigenvalues[index].imag, color=color, marker=marker, markersize=markersize, linestyle=linestyle, **kwargs)
        fig.tight_layout()
        return ax
    
    def compute_mode_properties(self):
        """ Post-process modal properties (ve, vg, traveling direction...) """
        #Future works
    
    def track_mode(self):
        """ Track a desired mode """
        #Future works
    
    def compute_response(self):
        """ Post-process the forced response """
        #Future works

    def _get_eigenvalues(self):
        """ Return all converged eigenvalues of the current EVP object (for internal use with SLEPc) """
        eigenvalues = np.array([self.evp.getEigenpair(i) for i in range(self.evp.getConverged())])
        if self.pb_type=="wavenumber":
            eigenvalues = np.sqrt(eigenvalues)
        return eigenvalues

    def _get_eigenvectors(self):
        """ Return all converged eigenvectors of the current EVP object in a PETSc dense matrix (for internal use with SLEPc) """
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
