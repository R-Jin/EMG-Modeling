#!/usr/bin/env python3
import numpy as np
from scipy.fft import fft
from scipy.signal import convolve
import math
import matplotlib.pyplot as plt

def cart2pol(x, y):
    # converts cartesian coordinates to polar coordinates
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return [theta, rho]

def get_binary_vectors(firing_samples, n_samples):
    """
    Args:
        firing_samples:     N x L matrix that contains N cells (one for each motor unit) and L indexes
                            of the samples at which the discharges of action potentials occur
        n_samples:          Number of samples

    Returns:
        binary_vectors:     N x n_samples matrix that contains N binary vectors, each containing 
                            n_samples of elements
    """

    n_cells = len(firing_samples)

    binary_vectors = np.array([np.zeros(n_samples) for _ in range(n_cells)])

    for i in range(n_cells):
        for index in firing_samples[i]:
            index = index[0]
            binary_vectors[i][index] = 1

    return binary_vectors

"""
def firstExtFunc(array, arraySize):
    tmp = [0] * arraySize 
    result = [tmp] * len(array[0])

    for k in array:
        i = 0
        for n in k:
            for m in n:
                result[i][m[0]] = 1
            i = i + 1
    return result

"""



def get_trains(action_potentials, binary_vectors):
    """
    Args:
        action_potentials:  N x M matrix that contains M action potentials of N motor units
        binary_vectors:     N x L matrix that contains N rows which is a binary vector representing
                            the time of discharges of the action potentials 

    Returns:
        trains:             N x n_samples matrix that contains N action potential trains where each train has
                            n_samples of total samples
    """

    n_motor_units = len(action_potentials)

    trains = [convolve(binary_vectors[i], action_potentials[i])[:len(binary_vectors[i])] for i in range(n_motor_units)]

    return trains



def main():
    action_potentials = np.load("./data_files/action_potentials.npy")
    firing_samples = np.load("./data_files/firing_samples.npy", allow_pickle=True)[0]

    # Signal duration in seconds
    SIGNAL_DURATION = 20
    # Sampling frequency in Hz
    SAMPLE_FREQUENCY = 10000

    # Number of samples within 20 seconds
    TOTAL_SAMPLES = SIGNAL_DURATION * SAMPLE_FREQUENCY

    binary_vectors = get_binary_vectors(firing_samples, TOTAL_SAMPLES)
    action_potential_trains = get_trains(action_potentials, binary_vectors)

    fig, axs = plt.subplots(2)

    # 0 - 20 seconds plot
    axs[0].plot(np.linspace(0, SIGNAL_DURATION, TOTAL_SAMPLES), action_potential_trains[0])
    axs[0].set_title("0 - 20 seconds")
    axs[0].set_xlabel('Seconds [s]')
    axs[0].set_ylabel('Arbitrary Unit [A.U]')

    # 10 - 10.5 seconds plot
    SMALL_SAMPLE_START = 10
    SMALL_SAMPLE_END = 10.5
    num_samples = int((SMALL_SAMPLE_END - SMALL_SAMPLE_START) * SAMPLE_FREQUENCY)
    lower_sample_range, upper_sample_range = int(SMALL_SAMPLE_START * SAMPLE_FREQUENCY), int(SMALL_SAMPLE_END * SAMPLE_FREQUENCY)
    axs[1].plot(np.linspace(SMALL_SAMPLE_START, SMALL_SAMPLE_END, num_samples), action_potential_trains[0][lower_sample_range: upper_sample_range])
    axs[1].set_title("10 - 10.5 seconds")
    axs[1].set_xlabel('Seconds [s]')
    axs[1].set_ylabel('Arbitrary Unit [A.U]')
    plt.tight_layout()

    # Sum of potential trains plot
    sum_of_potential_trains = np.sum(action_potential_trains, axis=0)
    figure = plt.figure()
    ax = plt.axes()
    ax.plot(np.linspace(SMALL_SAMPLE_START, SMALL_SAMPLE_END, num_samples), sum_of_potential_trains[lower_sample_range : upper_sample_range])
    ax.set_title("10 - 10.5 seconds")
    ax.set_xlabel('Seconds [s]')
    ax.set_ylabel('Arbitrary Unit [A.U]')

    #Question 2
    bin_matrix = get_binary_vectors(firing_samples, TOTAL_SAMPLES)

    for m in range(0, len(bin_matrix)):
        bin_matrix[m] = convolve(bin_matrix[m], np.hanning(SAMPLE_FREQUENCY), mode='same')

    fig2 = plt.figure()
    axes = plt.axes()
    axes.set_xlabel('Seconds [s]')
    axes.set_ylabel('Arbitrary Unit [A.U]')
    for i in range(len(bin_matrix)):
        axes.plot(np.linspace(0, SIGNAL_DURATION, TOTAL_SAMPLES), bin_matrix[i], label = "Binary Vector " + str(i + 1))
    axes.legend(fontsize = 8, loc = 'best')

    plt.show()

main()



"""
From Tiffany

- The filter (Hanning window) is a time domain filter. (impulse response -> time domain ??)

- The action potential train should be produced from the two signals, the binary vector
  and action potential, using a signal method (not using our current For loops)
- Note: She did think our graphs look fine, but we used the wrong method to achieve that

"""