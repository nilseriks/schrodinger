#!/usr/bin/env python3

import numpy as np


def energy_inf_square_well(EVmax, xmin, xmax, mass):
    energy = np.zeros((EVmax, ), dtype=float)
    for nn in range(EVmax):
        energy[nn] = np.pi**2 * nn**2 / (2 * mass * (xmin - xmax)**2)
    return energy
