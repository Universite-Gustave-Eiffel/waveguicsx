# waveguicsx

A class for solving waveguide problems (based on SLEPc eigensolver)

The problem must be based on the so-called SAFE (Semi-Analytical Finite Element) formulation:
$(\textbf{K}_1-\omega^2\textbf{M}+\text{i}k(\textbf{K}_2+\textbf{K}_2^\text{T})+k^2\textbf{K}_3)\textbf{U}=\textbf{0}$

The varying parameter can be the angular frequency $\omega$ or the wavenumber $k$.
In the former case, the eigenvalue is $k$, while in the latter case, the eigenvalue is $\omega^2$.
    
Example:
...TODO...
import waveguicsx
param = np.arange(0.1, 2, 0.1)
waveguide = waveguicsx.Waveguide(MPI.COMM_WORLD, M, K1, K2, K3)
waveguide.set_parameters(wavenumber=param) #or: waveguide.setParameters(omega=param)
waveguide.solve(nev)
waveguide.plot_dispersion()
plt.show()


Installation:
...TODO...

References:
...TODO...
