import numpy as np
from scipy.fft import fft
from scipy.signal import convolve
import math
import matplotlib.pyplot as plt


def main():
    
    #emg sampled at 1024 Hz. f.npy has 1024 entries => 1 second of recording (?)
    EMG_signal = np.load("./data_files/f.npy")[0]

    EMG_SAMPLES = len(EMG_signal)

    fig1 = plt.figure()
    axes1 = plt.axes()
    axes1.set_xlabel('Seconds [s]')
    axes1.set_ylabel('Arbitrary Unit [A.U]')
    axes1.plot(np.linspace(0, 1, EMG_SAMPLES), EMG_signal)
    
    plt.show()

main()

