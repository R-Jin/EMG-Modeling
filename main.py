#!/usr/bin/env python3
import numpy as np
from scipy.fft import fft
import math
import matplotlib.pyplot as plt

def cart2pol(x, y):
    # converts cartesian coordinates to polar coordinates
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return [theta, rho]

def get_trains(action_potentials, firing_samples, n_samples):
    """
    Args:
        action_potentials:  N x M matrix that contains M action potentials of N motor units
        firing_samples:     N x L matrix that contains N cells (one for each motor unit) and L indexes
                            of the samples at which the discharges of action potentials occur

    Returns:
        trains:             N x n_samples matrix that contains N action potential trains where each train has
                            n_samples of total samples
    """

    n_cells = len(action_potentials)
    n_points = len(action_potentials[0])

    trains = np.array([np.zeros(n_samples) for _ in range(n_cells)])

    for row in range(n_cells):
        for index in firing_samples[row]:
            index = index[0]
            for i in range(n_points):
                trains[row][index + i] += action_potentials[row][i]

    return trains

def main():
    action_potentials = np.load("./data_files/action_potentials.npy")
    firing_samples = np.load("./data_files/firing_samples.npy", allow_pickle=True)[0]

    # Signal duration in seconds
    SIGNAL_DURATION = 20
    # Sampling frequency in hz
    SAMPLE_FREQUENCY = 10000

    # Number of samples within 20 seconds
    TOTAL_SAMPLES = SIGNAL_DURATION * SAMPLE_FREQUENCY

    action_potential_trains = get_trains(action_potentials, firing_samples, TOTAL_SAMPLES)

    fig, axs = plt.subplots(2)
    fig.suptitle('Action Potential Train')

    # 0 - 20 seconds plot
    axs[0].plot(np.linspace(0, 20, TOTAL_SAMPLES), action_potential_trains[0])
    axs[0].set_title("0 - 20 seconds")
    axs[0].set_xlabel('Seconds [s]')
    axs[0].set_ylabel('Arbitrary Unit [A.U]')

    # 10 - 10.5 seconds plot
    num_samples = int((10.5 - 10) * SAMPLE_FREQUENCY)
    lower_sample_range, upper_sample_range = int(10 * SAMPLE_FREQUENCY), int(10.5 * SAMPLE_FREQUENCY)
    axs[1].plot(np.linspace(10, 10.5, num_samples), action_potential_trains[0][lower_sample_range: upper_sample_range])
    axs[1].set_title("10 - 10.5 seconds")
    axs[1].set_xlabel('Seconds [s]')
    axs[1].set_ylabel('Arbitrary Unit [A.U]')

    plt.show()

main()
