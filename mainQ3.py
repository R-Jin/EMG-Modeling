import numpy as np
from scipy.fft import fft
from scipy.signal import convolve
import math
import matplotlib.pyplot as plt


def main():
    
    #emg sampled at 1024 Hz. f.npy has 1024 entries => 1 second of recording
    EMG_signal = np.load("./data_files/f.npy")[0]

    EMG_SAMPLES = len(EMG_signal)
    EMG_T0 = 1
    INTERFERANCE_FREQ = 50
    # peak-to-peak amplitude = 0.2 => amplitude [-0.1 ; 0.1]
    INTERFERANCE_AMP = 0.1  

    # plotting EMG signal and Interference signal together in time domain
    fig1 = plt.figure()
    fig1.suptitle("EMG signal and interference")
    axes1 = plt.axes()
    axes1.set_xlabel('Seconds [s]')
    axes1.set_ylabel('Arbitrary Unit [A.U]')
    axes1.plot(np.linspace(0, 1, EMG_SAMPLES), EMG_signal, 'c', label = "EMG")

    x_sin = np.linspace(0, (INTERFERANCE_FREQ*2*np.pi), EMG_SAMPLES)
    y_sin_axis = np.sin(x_sin) * INTERFERANCE_AMP       

    x_sin_axis = np.linspace(0, 1, EMG_SAMPLES)

    axes1.plot(x_sin_axis, y_sin_axis, 'r', label = "Interference of 50 Hz")
    axes1.legend()

    # plotting corrupted and pure EMG signal in frequency domain
    f2_inter = np.sum([EMG_signal, y_sin_axis], axis = 0)

    F2 = fft(EMG_signal)
    amp_pure = abs(F2)

    F2_inter = fft(f2_inter)
    amp_inter = abs(F2_inter)

    k = np.arange(0, EMG_SAMPLES, step = 1)
    f_axis = k / EMG_T0     # axis in Hz (frequency axis)

    fig = plt.figure()
    fig.suptitle("DFT of EMG signal")
    axes = plt.axes()
    axes.plot(f_axis[0:int(EMG_SAMPLES/2)], amp_inter[0:int(EMG_SAMPLES/2)], 'r', label = "Corrupted")
    axes.plot(f_axis[0:int(EMG_SAMPLES/2)], amp_pure[0:int(EMG_SAMPLES/2)], 'c', label = "Pure")
    axes.set_xlabel('Frequency [Hz]')
    axes.set_ylabel('Arbitrary Unit [A.U]')
    axes.legend()
    
    plt.show()

main()