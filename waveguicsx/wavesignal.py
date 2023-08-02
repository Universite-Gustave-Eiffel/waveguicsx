import matplotlib.pyplot as plt
import numpy as np

class Signal:
    """
    A class for handling signals in the time domain and in the frequency domain
    
    Reminder:
    - the sampling frequency fs must be at least twice the highest excited frequency (fs>=2fmax)
    - the time duration T must be large enough to capture the slowest wave at z, the source-receiver distance
    
    Fourier transform definition used: X(f) = 2/T * integral of x(t)*exp(+i*omega*t)*dt
    Two remarks:
    1. this is not the fft function convention, which is in exp(-i*omega*t)
    2. the true amplitude of the Fourier transform, when needed, has to be obtained by
       multiplying the output (spectrum) by the scalar T/2, where T is the duration of the time signal
       (with the above definition: the division by T simplifies dimensionless analyses,
       and the factor 2 is used because only the positive part of the spectrum is considered)
    
    Complex Fourier transform:
    A complex Fourier transform is applied if alpha is set to a nonzero value
    The frequency vector has then an imaginary part, constant and equal to alpha/(2*pi)
    Complex frequency computations can be useful for the analysis of long time duration signals (avoids aliasing)
    A good choice is alpha = log(50)/T
    Note that the first frequency component is kept in that case (the frequency has a zero real part
    but non-zero imaginary part)
    
    Example:
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
    alpha : decaying parameter to apply complex Fourier transform (useful for long time duration signal)
    
    Methods
    -------
    __init__(time=None, waveform=None, frequency=None, spectrum=None, alpha=0):
        Initialization of signal
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
        ...
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
        Compute Fourier transform (positive frequency part only, time waveform are assumed to be real)
        If the number of time steps is odd, one point is added
        The zero frequency, if any, is suppressed
        Results are stored as attributes (names: frequency, spectrum)
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
        Compute inverse Fourier transform (only the positive frequency part is needed, time waveform are assumed to be real)
        Zero padding is applied in the low-frequency range (if missing) and in the high-frequency range (if coeff is greater than 1)
        Zero padding in the high frequency range is applied up to the frequency coeff*max(frequency)
        Results are stored as attributes (names: time, waveform)
        waveform is an array of size number_of_signals*len(time)
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
        f_high = np.arange(frequency[-1]+df, coeff*frequency[-1]+df, df)  #high frequency
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
        - fs is rounded up so that fs/fc is an integer
        - T is adjusted so that the number of points is even
        """
        # Time
        fs = np.ceil(fs/fc)*fc  #redefine fs so that fs/fc is an integer
        dt = 1/fs  #time step
        T = np.floor(T/2/dt)*2*dt + dt  #redefine T so that the number of points is equal to an even integer
        self.time = np.arange(0, T+dt, dt)  #time vector
        #N = len(self.time)  # number of points (N=T/dt+1, even)
        
        # Ricker waveform
        t0 = 1/fc
        self.waveform = (1-2*(self.time-t0)**2*np.pi**2*fc**2)*np.exp(-(self.time-t0)**2*np.pi**2*fc**2)
        self.fft()

    def toneburst(self, fs, T, fc, n):
        """
        Generate a toneburst signal (fs: sampling frequency, T: time duration, fc: central frequency, n: number of cycles)
        This signal is a Hanning-modulated n cycles sinusoidal toneburst centred at fc Hz (with unit amplitude)
        For this kind of excitation, fmax can be considered as 2*fc roughly, hence one should choose fs>=4fc
        Note that for better accuracy:
        - fs is rounded up so that fs/fc is an integer
        - T is adjusted so that the number of points is even
        """
        # Time
        fs = np.ceil(fs/fc)*fc  #redefine fs so that fs/fc is an integer
        dt = 1/fs  #time step
        T = np.floor(T/2/dt)*2*dt + dt  #redefine T so that the number of points is equal to an even integer
        self.time = np.arange(0, T+dt, dt)  #time vector
        #N = len(self.time)  # number of points (N=T/dt+1, even)
        
        # Toneburst waveform
        t = np.arange(0, n/fc+dt, dt)  #n/fc yields an integer number of time steps because fs/fc is an integer
        x = np.sin(2*np.pi*fc*t)
        x *= np.hanning(len(x))  #hanning window
        self.waveform = np.zeros(len(self.time))
        self.waveform[:len(x)] = x  #time amplitude vector
        self.fft()

    def chirp(self, fs, T, f0, f1, chirp_duration):
        """
        Generate a chirp of unit amplitude (fs: sampling frequency, T: time duration, f0: first frequency, f1: last frequency, chirp_duration: time to sweep from f0 to f1)
        Note that for better accuracy, T is adjusted so that the number of points is even
        """
        # Time
        dt = 1/fs  #time step
        T = np.floor(T/2/dt)*2*dt + dt  #redefine T so that the number of points is equal to an even integer
        self.time = np.arange(0, T+dt, dt)  #time vector
        
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
        fig.tight_layout()
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