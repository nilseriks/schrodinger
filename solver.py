#!/usr/bin/env python3
"""Main environment to solve the one dimensional time independent schrodinger
equation for diffrent potentials. It writes the energies into energies.dat,
the wavefunctions into wavefuncs.dat, the potential into potential.dat and
the expected values of the position into expvalues.dat."""

import calculus
import numpy as np


_DIRECTORY = '../files'
_FILE = 'schrodinger3.inp'


def main():
    """Main function to solve the one dimensiional time independent schrodinger
    equation."""
    # Read out the informations of the schrodinger.inp file.
    inp_data = calculus.io.read_schrodinger(_DIRECTORY, _FILE)
    _MASS = inp_data["_MASS"]
    _XMIN = float(inp_data["plot_set"][0])
    _XMAX = float(inp_data["plot_set"][1])
    _NPOINT = int(inp_data["plot_set"][2])
    _MIN_EV = int(inp_data["evalues"][0])
    _MAX_EV = int(inp_data["evalues"][1])
    _REG_TYPE = inp_data["regression"]
    _INTERPOLATE_NR = int(inp_data["interpolate_nr"])
    _DISCRETE_POT = inp_data['pot']

    _XPLOT = np.linspace(_XMIN, _XMAX, num=_NPOINT, endpoint=True)

    _POT = calculus.calc.pot_calc(_XPLOT, _DISCRETE_POT, _REG_TYPE)

    _EVAL, _EVEC = calculus.calc._solve_seq(_XMIN, _XMAX, _NPOINT, _MASS, _POT)

    _WAVEFUNCS = calculus.calc.get_WF_array(_XPLOT, _MIN_EV, _MAX_EV, _EVEC)

    calculus.file_io.write_result(_DIRECTORY, 'wavefuncs.dat', _WAVEFUNCS)

    calculus.file_io.write_result(_DIRECTORY, 'energies.dat', _EVAL)

    _POT = np.transpose(np.vstack((_XPLOT, _POT)))

    calculus.file_io.write_result(_DIRECTORY, 'potential.dat', _POT)


if __name__ == '__main__':
    main()
