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
        trains: N x n_samples matrix that contains N action potential trains where each train has
        n_samples of total samples
    """

    n_cells = len(action_potentials)

    trains = np.array([np.zeros(n_samples) for _ in range(n_cells)])

    return trains

# How many samples an action potential is.
def get_trains_action_samples(firing_samples): return np.fromiter([len(x) for x in firing_samples]  ,dtype=int)

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
    plt.plot(np.arange(0, 20, 1/SAMPLE_FREQUENCY), action_potential_trains[0])
    plt.show()

main()
