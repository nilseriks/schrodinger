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

    _XPLOT = np.linspace(inp['_XMIN'], inp['_XMAX'], num=inp['_NPOINT'],
                         endpoint=True)

    _POT = calculus.calc.pot_calc(_XPLOT, inp['_POT'], inp['_REG_TYPE'])

    _EVAL, _EVEC = calculus.calc._solve_seq(inp['_XMIN'], inp['_XMAX'],
                                            inp['_NPOINT'], inp['_MASS'], _POT)

    _EVAL = _EVAL[inp['_MIN_EV'] - 1: inp['_MAX_EV']]


    xevec = calculus.calc.get_WF_array(_XPLOT, inp['_MIN_EV'],
                                            inp['_MAX_EV'], _EVEC)

    _EXPECTEDX = calculus.calc.expected_values(_XPLOT, _EVEC, inp['_MIN_EV'],
                                               inp['_MAX_EV'])

    _UNCERTAINTY = calculus.calc.uncertainty(_XPLOT, _EVEC, inp['_MIN_EV'],
                                             inp['_MAX_EV'])

    _EXP_VALUES = calculus.calc.get_exp_unc(_EXPECTEDX, _UNCERTAINTY)

    _POTX = np.transpose(np.vstack((_XPLOT, _POT)))

    calculus.file_io.create_files(_DIRECTORY, _EVAL, _EXP_VALUES, _POTX, xevec)


if __name__ == '__main__':
    main()
