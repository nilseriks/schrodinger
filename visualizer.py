#!/usr/bin/env python3
"""This script visualizes the wavefunctions of a given potential. It also shows
the expected values of the position and the corresponding uncertainties."""


import numpy as np
import calculus


def main():
    """Main function to show the plot of the potential, the eigenvalues, the
    wavefunctions, the expected values of the position of the particle. It
    reads out the data which were calculated by the solver.
    """
    _DATA = calculus.file_io.read_files('files')
    _EVAL = _DATA[0]
    _EXPX = _DATA[1][:, 0]
    _UNC = _DATA[1][:, 1]
    _XPLOT = _DATA[2][:, 0]
    _POT = _DATA[2][:, 1:]
    _EVEC = _DATA[3][:, 1:]

    _XMIN = np.amin(_XPLOT)
    _XMAX = np.amax(_XPLOT)

    inp = calculus.file_io.read_schrodinger('files', 'schrodinger5.inp')
    _MIN_EV = inp['_MIN_EV']
    _MAX_EV = inp['_MAX_EV']

    _YDIFF = abs(_EVAL[_MAX_EV - 1] - np.amin(_POT))

    calculus.plot.pot_plot(_XMIN, _XMAX, _MIN_EV, _MAX_EV, _EVAL, _EVEC, _POT,
                           _XPLOT, _YDIFF, _EXPX, _UNC)


if __name__ == '__main__':
    main()
