import numpy as np
from scipy.fft import fft
from scipy.signal import convolve
import math
import matplotlib.pyplot as plt


def main():
    
    #emg sampled at 1024 Hz. f.npy has 1024 entries => 1 second of recording (?)
    EMG_signal = np.load("./data_files/f.npy")[0]
    EMG_signal_inter = np.load("./data_files/f.npy")[0]

    EMG_SAMPLES = len(EMG_signal)

    fig1 = plt.figure()
    fig1.suptitle("EMG signal and interference")
    axes1 = plt.axes()
    axes1.set_xlabel('Seconds [s]')
    axes1.set_ylabel('Arbitrary Unit [A.U]')
    axes1.plot(np.linspace(0, 1, EMG_SAMPLES), EMG_signal, 'c', label = "EMG")

    freq = 50 # Hz
    x_sin = np.linspace(0, (freq*2*np.pi), EMG_SAMPLES)
    y_sin_axis = np.sin(x_sin)/10         # peak-to-peak amplitude = 0.2 => amplitude [-0.1 ; 0.1]

    x_sin_axis = np.linspace(0, 1, EMG_SAMPLES)

    axes1.plot(x_sin_axis, y_sin_axis, 'r', label = "Interference of 50 Hz")
    axes1.legend()

    # This part is a little iffy. Does somebody know better values?
    # The current ones causes there to be no difference between the
    # frequency graph and the angular frequency graph.
    T = 1/1024    # 1024 samples in 1 second
    T0 = 1
    N0 = 1024     # T0 / T

    f2 = EMG_signal

    f2_inter = np.sum([EMG_signal_inter, y_sin_axis], axis = 0)

    F2 = fft(f2)
    amplitude = abs(F2)
    phase = np.arctan2(F2.imag, F2.real)

    F2_inter = fft(f2_inter)
    amp_inter = abs(F2_inter)
    phase_inter = np.arctan2(F2_inter.imag, F2_inter.real)

    k = np.arange(0, N0, step = 1)
    f_axis = k / T0     # axis in Hz (frequency axis)
    w_axis = 2*np.pi * k/T0     # axis in rad/s (angular speed axis)

    fig = plt.figure()
    #fig.suptitle("DFT of EMG signal")
    axes = plt.axes()
    axes.plot(f_axis[0:int(N0/2)], amp_inter[0:int(N0/2)], 'r', label = "Corrupted")
    axes.plot(f_axis[0:int(N0/2)], amplitude[0:int(N0/2)], 'c', label = "Pure")
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Arbitrary Unit [A.U]')
    axes.legend()
    #plt.tight_layout()

    # Temporarily removed since it's the same with current T and T0 values
    """
    fig1 = plt.figure()
    #fig1.suptitle("Angular frequency")
    axes1 = plt.axes()
    axes1.plot(w_axis[0:int(N0/2)], amp_inter[0:int(N0/2)], 'r', label = "Interference")
    axes1.plot(w_axis[0:int(N0/2)], amplitude[0:int(N0/2)], 'c', label = "Clean")
    axes1.set_xlabel('Angular frequency [rad/s]')
    axes1.set_ylabel('Arbitrary Unit [A.U]')
    axes1.legend()
    #plt.tight_layout()
    """
    
    plt.show()

main()

