# %%
import math
from mne import parallel
import numpy as np
import matplotlib.pyplot as plt
#from numba import jit, cuda


def _rhu(n):
    '''
    Return integer rounded of the input. The half will round up.
    1.3 -> 1
    2.5 -> 3
    -0.5 -> 0
    -0.3 -> -1
    '''
    return int(math.floor(n + 0.5))


def _datawrap(x: np.ndarray, n: int) -> np.ndarray:
    '''
    The calculation of signal spectrum, such as periodogram, uses FFT internally, 
    where the length of FFT is denoted as NFFT. In theory, when using FFT, 
    the signal in both time domain and frequency domain are discrete and periodic, 
    where the period is given by NFFT. Hence, if you specify an NFFT that is less 
    than the signal length, it actually introduces the aliasing in the time domain 
    and make the signal (even if its length is N>NFFT) periodic with NFFT. 
    When you take FFT of this sequence, you are working with this aliased sequence. 
    This is what datawrap do for you. 

    For example: Sequence 1 2 3 4 5, period 5, it returns
        1 2 3 4 5
                  1 2 3 4 5
                            1 2 3 4 5
        --------------------------------
              ... 1 2 3 4 5 ...

    i.e., original series. assume a period of 3, then it looks like

        1 2 3 4 5
              1 2 3 4 5
                    1 2 3 4 5
        ------------------------
          ... 5 7 3 ...

    A sequence that is wrapped around and has only a length of 3.

    >>> _datawrap(range(1, 6),3)
    array([5, 7, 3])

    '''
    return np.array([sum(x[i::n]) for i in range(n)])


def _chwi_krn(D: np.ndarray, L: np.ndarray, A: int = None):
    '''

    CHWI_KRN Choi-Williams kernel function.

    https://en.wikipedia.org/wiki/Bilinear_time%E2%80%93frequency_distribution#Choi%E2%80%93Williams_distribution_function

    K = _chwi_krn(D, L, A) returns the values K of the Choi-Williams kernel function
    evaluated at the doppler-values in matrix D and the lag-values in matrix L.
    Matrices D and L must have the same size. The values in D should be in the range
    between -1 and +1 (with +1 being the Nyquist frequency). The parameter A is
    optional and controls the "diagonal bandwidth" of the kernel. Matrix K is of the
    same size as the matrices D and L. Parameter A defaults to 10 if omitted.

    Copyright (c) 1998 by Robert M. Nickel
    Revision: 1.1.1.1
    Date: 2001/03/05 09:09:36

    Written by: Mahdi Kiani, March 2021

    '''

    if A is None:
        A = 10
    K = np.exp((-1/(A*A)) * (D*D*L*L))

    return K

#@jit(parallel=True)
def rid_rihaczek(x: np.ndarray, fbins: int):
    '''
    This is python implementation of rid_rihaczek4 function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes reduced interference Rihaczek distribution

    Parameter:
        x: signal
        fbins=required frequency bins

    Returns:
        tfd = Generated reduced interference Rihaczek distribution

    Written by: Mahdi Kiani, March 2021
    '''

    tbins = x.shape[0]
    amb = np.zeros((tbins, tbins))
    for tau in range(tbins):
        amb[tau, :] = (np.conj(x) * np.concatenate((x[tau:], x[:tau])))

    ambTemp = np.concatenate(
        (amb[:, _rhu(tbins/2):], amb[:, :_rhu(tbins/2)]), axis=1)
    amb1 = np.concatenate(
        (ambTemp[_rhu(tbins/2):, :], ambTemp[:_rhu(tbins/2), :]), axis=0)

    D = np.outer(np.linspace(-1, 1, tbins), np.linspace(-1, 1, tbins), )
    K = _chwi_krn(D, D, 0.01)
    df = K[:amb1.shape[0], :amb1.shape[1]]
    ambf = amb1 * df

    A = np.zeros((fbins, tbins))
    tbins = tbins - 1

    if tbins != fbins:
        for tt in range(tbins):
            A[:, tt] = _datawrap(ambf[:, tt], fbins)
    else:
        A = ambf

    tfd = np.fft.fft(A, axis=0)

    return tfd


def _calc_MVL(phase: np.ndarray, amp: np.ndarray):
    assert phase.shape[0] == amp.shape[0]
    z1 = np.exp(1j * phase)

    z = amp * (z1)    # Generate complex valued signal
    MVL = abs(z.mean())

    return MVL


def tfMVL_tfd_2d(tfd, high_freq, low_freq):
    '''
    This is python implementation of MVL_lab function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        tfd          : input time frequency decomposition 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, March 2021
    '''

    # Amplitude and Phase calculation
    tf_canolty = np.zeros((high_freq[1] - high_freq[0] + 1, low_freq[1] - low_freq[0] + 1))
    for i, h_freq in enumerate(range(high_freq[0], high_freq[1] + 1)):
        for j, l_freq in enumerate(range(low_freq[0], low_freq[1] + 1)):
            Amp = abs(tfd[h_freq, :])
            tfd_low = tfd[l_freq, :]
            angle_low = np.angle(tfd_low)
            Phase = angle_low

            tf_canolty[i, j] = _calc_MVL(Phase, Amp)

    return tf_canolty


def tfMVL_tfd2_2d(tfdx, tfdy, high_freq, low_freq):
    '''
    This is python implementation of MVL_lab function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        tfd          : input time frequency decomposition 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, March 2021
    '''

    # Amplitude and Phase calculation
    tf_canolty = np.zeros((high_freq[1] - high_freq[0] + 1, low_freq[1] - low_freq[0] + 1))
    for i, h_freq in enumerate(range(high_freq[0], high_freq[1] + 1)):
        for j, l_freq in enumerate(range(low_freq[0], low_freq[1] + 1)):
            Amp = abs(tfdx[h_freq, :])
            tfd_low = tfdy[l_freq, :]
            angle_low = np.angle(tfd_low)
            Phase = angle_low

            tf_canolty[i, j] = _calc_MVL(Phase, Amp)

    return tf_canolty


def tfMVL_tfd2_2d_time(tfdx, tfdy, high_freq, low_freq, ind_start, ind_end):
    '''
    This is python implementation of MVL_lab function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        tfd          : input time frequency decomposition 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, August 2021
    '''

    # Amplitude and Phase calculation
    tfdx = tfdx[:, ind_start:ind_end]
    tfdy = tfdy[:, ind_start:ind_end]
    tf_canolty = np.zeros((high_freq[1] - high_freq[0] + 1, low_freq[1] - low_freq[0] + 1))
    for i, h_freq in enumerate(range(high_freq[0], high_freq[1] + 1)):
        for j, l_freq in enumerate(range(low_freq[0], low_freq[1] + 1)):
            Amp = abs(tfdx[h_freq, :])
            tfd_low = tfdy[l_freq, :]
            angle_low = np.angle(tfd_low)
            Phase = angle_low

            tf_canolty[i, j] = _calc_MVL(Phase, Amp)

    return tf_canolty

#@jit
def tfMVL_tfd(tfd, high_freq, low_freq):
    '''
    This is python implementation of MVL_lab function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        tfd          : input time frequency decomposition 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, March 2021
    '''

    # Amplitude and Phase calculation

    Amp = abs(sum(tfd[high_freq[0]:high_freq[1]+1, :]))
    tfd_low = sum(tfd[low_freq[0]:low_freq[1]+1, :])
    angle_low = np.angle(tfd_low)
    Phase = angle_low

    tf_canolty = _calc_MVL(Phase, Amp)

    return tf_canolty


def tfMVL_tfd2(tfdx, tfdy, high_freq, low_freq):
    '''
    This is python implementation of MVL_lab function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        tfd          : input time frequency decomposition 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, March 2021
    '''

    # Amplitude and Phase calculation

    Amp = abs(sum(tfdx[high_freq[0]:high_freq[1]+1, :]))
    tfd_low = sum(tfdy[low_freq[0]:low_freq[1]+1, :])
    angle_low = np.angle(tfd_low)
    Phase = angle_low

    tf_canolty = _calc_MVL(Phase, Amp)

    return tf_canolty

#@jit
def tfMVL(x, high_freq, low_freq, Fs):
    '''
    This is python implementation of tfMVL function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        x            : input signal 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, March 2021
    '''

    # Amplitude and Phase calculation
    tfd = rid_rihaczek(x, Fs)
    Amp = abs(sum(tfd[high_freq[0]:high_freq[1]+1, :]))
    tfd_low = sum(tfd[low_freq[0]:low_freq[1]+1, :])
    angle_low = np.angle(tfd_low)
    Phase = angle_low

    tf_canolty = _calc_MVL(Phase, Amp)

    return tf_canolty


def tfMVL2(x, high_freq, y, low_freq, Fs):
    '''
    This is python implementation of tfMVL function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes the phase amplitude coupling using TF-MVL method.

    Parameter:
        x            : input signal 
        high_freq    : Amplitude Frequency range 
        low_freq     : Phase Frequency range 
        Fs           : Sampling Frequency  

    Returns:
        tf_canolty   : Computed PAC using TF-MVL method

    Written by: Mahdi Kiani, March 2021
    '''

    # Amplitude and Phase calculation
    tfdx = rid_rihaczek(x, Fs)
    tfdy = rid_rihaczek(y, Fs)
    Amp = abs(sum(tfdx[high_freq[0]:high_freq[1]+1, :]))
    tfd_low = sum(tfdy[low_freq[0]:low_freq[1]+1, :])
    angle_low = np.angle(tfd_low)
    Phase = angle_low

    tf_canolty = _calc_MVL(Phase, Amp)

    return tf_canolty


# %%
# help(rid_rihaczek)

def main():

    high_freq = [39, 41]
    low_freq = [4, 8]
    t = np.linspace(-0.2, 1, 601)
    x = np.zeros(t.shape)

    for i in range(0, 11, 2):
        x += i * np.sin(2*np.pi*i*(t - i/10))*(t > i/10)

    tbins = x.shape[0]

    tfd = rid_rihaczek(x, 500)

    tfMVL(x, high_freq, low_freq, 500)

    plt.plot(x)


if __name__ == "__main__":
    main()
