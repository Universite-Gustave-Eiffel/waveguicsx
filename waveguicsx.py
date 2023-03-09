from typing import Union, List
from petsc4py import PETSc
from slepc4py import SLEPc

import matplotlib.pyplot as plt
import numpy as np
import time

# TODO:
#- parallelization: loop! (+SLEPc?)
#- test and debugging for: Dirichlet BC, multiple domains, PML

# TODISCUSS:
#- memory usage (-> play with A is B, play with import sys; sys.getsizeof(eigenvectors), memory profiler...
#      eigenvalues[i] as numpy array? (low gain seemingly), eigenvectors[i] as PETSC matrix? 

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
    eigenvalues : list
        list of wavenumbers or angular frequencies
        access to components with eigenvalues[ip][imode] (ip: parameter index, imode: mode index)
    eigenvectors : list
        list of mode shapes
        access to components with eigenvectors[ik][imode][idof] (ip: parameter index, imode: mode index, idof: dof index)
    
    Methods
    -------
    set_parameters(omega=None, wavenumber=None):
        Set problem type (pb_type), the parameter range (omega or wavenumber) as well as default parameters of SLEPc eigensolver (evp)
    solve(nev=1, target=0):
        Solve the eigenvalue problem repeatedly for the parameter range, solutions are stored as attributes (eigenvalues, eigenvectors)
    plot_dispersion(axs=None, color="k",  marker="o", markersize=2, linestyle="", **kwargs):
        Plot dispersion curves (based on multiple subplot matplotlib)
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
            self.omega = omega
            # Setup the SLEPc solver for the quadratic eigenvalue problem
            self.evp = SLEPc.PEP()
            self.evp.create(comm=self.comm)
            self.evp.setProblemType(SLEPc.PEP.ProblemType.GENERAL) #note: in the undamped case, HERMITIAN is currently not possible (matrices are not Hermitian due to round-off)
            self.evp.setType(SLEPc.PEP.Type.QARNOLDI) #note: the computational speed of LINEAR, QARNOLDI and TOAR seems to be almost identical
            self.evp.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_IMAGINARY)

        # The parameter is the frequency omega, the eigenvalue is the wavenumber k
        elif omega is None:
            self.pb_type = "wavenumber"
            self.wavenumber = wavenumber
            # Setup the SLEPc solver for the generalized eigenvalue problem
            self.evp = SLEPc.EPS()
            self.evp.create(comm=self.comm)
            self.evp.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #note: GHEP (generalized Hermitian) is slower, is it because Hermitian matrices are enforced internally?
            self.evp.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #note: ARNOLDI also works although slightly slower
            self.evp.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

        # Setup common to EPS and PEP
        self.evp.setTolerances(tol=1e-6, max_it=20)
        ST = self.evp.getST()
        ST.setType(SLEPc.ST.Type.SINVERT)
        ST.setShift(0)
        self.evp.setST(ST)
        self.evp.setFromOptions()

    def solve(self, nev=1, target=0):
        """
        Solve the dispersion problem, i.e. the eigenvalue problem repeatedly for the parameter range (omega or wavenumber)
        The solutions are stored in the attributes eigenvalues and eigenvectors
        
        Parameters
        ----------
        nev : int
            number of eigenpairs requested
        target : complex, optional (default: 0)
            target around which eigenpairs are looked for
        """
        # Eigensolver setup
        self.evp.setDimensions(nev=nev)
        self.evp.setTarget(target)
        
        # Print setup information
        parameters = {"omega": self.omega, "wavenumber": self.wavenumber}[self.pb_type]
        print(f'Waveguide parameter: {self.pb_type} ({len(parameters)} iterations)\n' \
               '\n---- SLEPc setup ----\n')
        self.evp.view()
        print('')
        
        # Loop over the parameter
        for i, parameter_value in enumerate(parameters):
            start = time.perf_counter()
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
        if axs is None: #!!!!!!!!!!Max. : est-ce pertinent d'avoir ax en input avec subplot????
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
        """ Return all converged eigenvalues of the current EVP object (for internal use, for SLEPc) """
        eigenvalues = [self.evp.getEigenpair(i) for i in range(self.evp.getConverged())]
        if self.pb_type=="wavenumber":
            eigenvalues = np.sqrt(eigenvalues)
        return eigenvalues

    def _get_eigenvectors(self):
        """ Return all converged eigenvectors of the current EVP object  in a list (for internal use, for SELPc) """
        eigenvectors = list()
        v = self.evp.getOperators()[0].createVecRight()
        for i in range(self.evp.getConverged()):
            self.evp.getEigenpair(i, v) #To get left eigenvectors: evp.getLeftEigenvector(i, vr.vector)
            eigenvectors.append(v.copy())
        return eigenvectors
