#!/usr/bin/env python3
"""Script testing the solver for the one dimensional time independent
schrodinger equation."""

import numpy as np


def energy_inf_square_well(EVmin, EVmax, xmin, xmax, mass):
    """Calculates the energies of the infinite square well for given number
    of eigenvalues.

    Args:
        EVmin: Set the lower bound for the eigenvalues to calculate.
        EVmax: Set the upper bound for the eigenvalues to calculate.
        xmin: Start of the box.
        xmax: End of the box.
        mass: Mass of the Particle.

    Returns:
        energy: Array containing the calculated eigenvalues."""
    energy = np.zeros((EVmax, ), dtype=float)
    for nn in range(EVmin, EVmax):
        energy[nn] = np.pi**2 * nn**2 / (2 * mass * (xmin - xmax)**2)
    return energy


def energy_harm_osc(EVmin, EVmax):
    """Calculates the energies of the harmonic oscillator for given number of
    eigenvalues.

    Args:
        EVmin: Set the lower bound for the eigenvalues to calculate.
        EVmax: Set the upper bound for the eigenvalues to calculate.

    Returns:
        energy: Array containing the calculated eigenvalues."""
    energy = np.zeros((EVmax, ), dtype=float)
    for nn in range(EVmin, EVmax):
        energy[nn] = nn + 0.5
    return energy
