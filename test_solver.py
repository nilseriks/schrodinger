#!/usr/bin/env python3
"""Script testing the solver for the one dimensional time independent
schrodinger equation."""

import numpy as np
import calculus


_DIRECTORY = 'files'
_FILE = 'schrodinger.inp'

inp_data = calculus.io.read_schrodinger(_DIRECTORY, _FILE)
_MASS = inp_data["_MASS"]
_XMIN = float(inp_data["plot_set"][0])
_XMAX = float(inp_data["plot_set"][1])
_NPOINT = int(inp_data["plot_set"][2])
_MIN_EV = int(inp_data["evalues"][0])
_MAX_EV = int(inp_data["evalues"][1])
_REG_TYPE = inp_data["regression"]
_INTERPOLATE_NR = int(inp_data["interpolate_nr"])
_POT = inp_data['pot']
_XPLOT = np.linspace(_XMIN, _XMAX, num=_NPOINT, endpoint=True)
_XPLOT2 = np.linspace(0, 4, num=_NPOINT, endpoint=True)

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


def wf_inf_square_well(xplot, xmin, xmax, npoint, wf_number):
    _LENGTH = abs(xmin - xmax)
    PSI = np.zeros((npoint, wf_number), dtype=float)
    for nn in range(wf_number):
        PSI[:, nn] = np.sqrt(2 / _LENGTH) * np.sin((nn + 1) * np.pi * xplot
                             / _LENGTH)
    return PSI


_POT2 = calculus.calc.pot_calc(_XPLOT, _POT, 'linear')
print(wf_inf_square_well(_XPLOT2, 0, 4, _NPOINT, 20))
print(calculus.calc._solve_seq(_XMIN, _XMAX, _NPOINT, 1, _POT2)[1][:, 0:20])


def test_inf_square_well():
    _POT2 = calculus.calc.pot_calc(_XPLOT, _POT, 'linear')
    AA = wf_inf_square_well(_XPLOT2, 0, 4, _NPOINT, 20)
    BB = calculus.calc._solve_seq(_XMIN, _XMAX, _NPOINT, 1, _POT2)[1][:, 0:20]
    for nn in range(20):
        BB[:, nn] = BB[:, nn] * 1 / np.sum(BB[:, nn])
    assert np.allclose(AA, BB, rtol=1, atol=1)
