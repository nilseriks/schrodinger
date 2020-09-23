#!/usr/bin/env python3
"""This script calculates the test data of problems which have analytic
   solutions and saves it:
        Energy of the infinite square well
        Energy of the harmonic oscillator
        Potential of the infinite square well
        Potential of the finite square well
        Potential of the harmonic oscillator.
    """

import numpy as np
import calculus


def _energy_inf_square_well():
    """Calculates the energies of the infinite square well for the first 20
    eigenvalues.

    Args:
        EVmax (int): Set the upper bound for the eigenvalues to calculate.

    Returns:
        energy (1darray): Array containing the calculated eigenvalues.
    """
    energy = np.zeros((20, ), dtype=float)
    for nn in range(0, 20):
        energy[nn] = np.pi**2 * (nn + 1)**2 / (2 * 2 * (-2 - 2)**2)
    calculus.file_io.write_result('../tests/test_energy',
                                  'E_inf_square_well.dat', energy)
    return energy


def _energy_harm_osc():
    """Calculates the energies of the harmonic oscillator for a given number of
    eigenvalues.

    Args:
        EVmin (int): Set the lower bound for the eigenvalues to calculate.
        EVmax (int): Set the upper bound for the eigenvalues to calculate.

    Returns:
        energy (1darray): Array containing the calculated eigenvalues.
    """
    energy = np.zeros((20, ), dtype=float)
    for nn in range(20):
        energy[nn] = 0.5 * nn + 0.25
    calculus.file_io.write_result('../tests/test_energy', 'E_harm_osc.dat',
                                  energy)
    return energy


def _pot_inf_square_well():
    """Calculates the potential of the ininite square well problem.
    """
    pot = np.zeros((1999, ), dtype=float)
    calculus.file_io.write_result('../tests/test_potential',
                                  'pot_inf_square_well.dat', pot)
    return pot


def _pot_fin_square_well():
    """Calculates the potential of the finite square well problem.
    """
    pot1 = np.zeros((750, ), dtype=float)
    pot2 = -10 * np.ones((499, ), dtype=float)
    pot3 = np.zeros((750, ), dtype=float)
    pot = np.concatenate((pot1, pot2, pot3), axis=0)
    calculus.file_io.write_result('../tests/test_potential',
                                  'pot_fin_square_well.dat', pot)
    return pot


def _pot_harm_osc():
    """Calculates the potential of the harmonic oscillator.
    """
    xplot = np.linspace(-5, 5, num=1999, endpoint=True)
    pot = 0.5 * xplot**2

    calculus.file_io.write_result('../tests/test_potential',
                                  'pot_harm_osc.dat', pot)
    return pot


if __name__ == '__main__':
    _energy_inf_square_well()
    _energy_harm_osc()
    _pot_inf_square_well()
    _pot_harm_osc()
    _pot_fin_square_well()
