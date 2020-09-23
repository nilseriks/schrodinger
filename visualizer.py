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
    data = calculus.file_io.read_files('files')
    energy = data[0]
    xplot = data[2][:, 0]
    pot = data[2][:, 1:]
    evec = data[3][:, 1:]

    xmin = np.amin(xplot)
    xmax = np.amax(xplot)

    inp = calculus.file_io.read_schrodinger('files', 'schrodinger5.inp')
    min_ev = inp['_MIN_EV']
    max_ev = inp['_MAX_EV']

    if min_ev < max_ev:
        expectedx = data[1][:, 0]
        uncertainty = data[1][:, 1]
    elif min_ev == max_ev:
        expectedx = data[1][0]
        uncertainty = data[1][1]

    if min_ev < max_ev:
        ydiff = abs(energy[max_ev - 1] - np.amin(pot))
    elif min_ev == max_ev:
        ydiff = abs(energy - np.amin(pot))

    if min_ev < max_ev:
        calculus.plot.pot_plot(xmin, xmax, min_ev, max_ev, energy, evec, pot,
                               xplot, ydiff, expectedx, uncertainty)
    else:
        calculus.plot.pot_plot2(xmin, xmax, min_ev, max_ev, energy, evec, pot,
                                xplot, ydiff, expectedx, uncertainty)


if __name__ == '__main__':
    main()
