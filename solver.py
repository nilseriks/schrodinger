#!/usr/bin/env python3
"""Main environment to solve the one dimensional time independent schrodinger
equation for diffrent potentials. It writes the energies into energies.dat,
the wavefunctions into wavefuncs.dat, the potential into potential.dat and
the expected values of the position into expvalues.dat."""

import calculus
import numpy as np


_DIRECTORY = 'files'

_FILE = 'schrodinger5.inp'


def main():
    """Main function to solve the one dimensiional time independent schrodinger
    equation."""
    # Read out the informations of the schrodinger.inp file.
    inp = calculus.io.read_schrodinger(_DIRECTORY, _FILE)

    xplot = np.linspace(inp['_XMIN'], inp['_XMAX'], num=inp['_NPOINT'],
                        endpoint=True)

    pot = calculus.calc.pot_calc(xplot, inp['_POT'], inp['_REG_TYPE'])

    energy, evec = calculus.calc._solve_seq(inp['_XMIN'], inp['_XMAX'],
                                            inp['_NPOINT'], inp['_MASS'], pot)

    energy = energy[inp['_MIN_EV'] - 1: inp['_MAX_EV']]

    xevec = calculus.calc.get_WF_array(xplot, inp['_MIN_EV'],
                                       inp['_MAX_EV'], evec)

    expectedx = calculus.calc.expected_values(xplot, evec, inp['_MIN_EV'],
                                              inp['_MAX_EV'])

    uncertainty = calculus.calc.uncertainty(xplot, evec, inp['_MIN_EV'],
                                            inp['_MAX_EV'])

    exp_values = calculus.calc.get_exp_unc(expectedx, uncertainty)

    x_pot = np.transpose(np.vstack((xplot, pot)))

    calculus.file_io.create_files(_DIRECTORY, energy, exp_values, x_pot, xevec)


if __name__ == '__main__':
    main()
