#!/usr/bin/env python3
"""This script visualizes the wavefunctions of a given potential. It also shows
the expected values of the position and the corresponding uncertainties."""


import numpy as np
import matplotlib.pyplot as plt
import calculus


def _scale_plot(minEV, maxEV, EVAL, EVEC, indexEV, RTOL, ATOL):
    """Calculates the multiplication factor which scales the wavefunctions for
    a better visualization in the graphical plot.

    Args:
        minEV: Lower bound of the eigenvalues which should be visualized.
        maxEV: Upper bound of the eigenvalues which should be visualized.
        EVAL: Array of eigenvalues.
        EVEC: Array containing the wavefunctions as column vectors.
        indexEV: The indexEV'th wavefunction to calculate the multiplication
                 factor of.
        RTOL: Relative tolerence to compare diffrent eigenvalues.
        ATOL: Absolute tolerence to compare diffrent eigenvalues.

    Returns:
        _SCALE: Multiplication factor which scales the eigenvectors."""
    DIFF_LIST = []

    for kk in range(minEV - 1, maxEV - 1):
        if not np.allclose(EVAL[kk + 1], EVAL[kk], atol=ATOL,
                           rtol=RTOL):
            DIFF_LIST.append(abs(EVAL[kk + 1] - EVAL[kk]))
    _SCALE = 0.4 * min(DIFF_LIST) * 1 / np.amax(abs(EVEC[:, indexEV]))
    return _SCALE


def pot_plot(xmin, xmax, minEV, maxEV, EVAL, EVEC, pot, xplot, ydiff, expx, unc):
    """Creates a graphical plot. It shows the potential, the eigenvalues, the
    wavefunctions, the expected values of the position of the particle. And
    within a second plot it shows the uncertainty of the expected position.

    Args:
        xmin: Lower bound of the x values.
        xmax: Upper bound of the x values.
        minEV: Lower bound of the eigenvalues which should be visualized.
        maxEV: Upper bound of the eigenvalues which should be visualized.
        EVAL: Array of eigenvalues.
        EVEC: Array containing the wavefunctions as column vectors.
        pot: Interpolation of the potential at the xplot values.
        xplot: Values were the potential is defined.
        ydiff: Absolute difference between the lowest pot-value and the highest
               eigenvalue.
        expx: Expected values of the position.
        unc: Uncertainty of the position.
    """
    ATOL = 0.05 * ydiff
    RTOL = 0.05 * ydiff
    _YMIN = np.amin(pot) - 0.05 * ydiff
    _max_scale = _scale_plot(minEV, maxEV, EVAL, EVEC, maxEV - 1, RTOL, ATOL)
    _YMAX = EVAL[maxEV - 1] + np.amax(_max_scale * EVEC[:, maxEV - 1]) + 0.05 * ydiff

    plt.figure(figsize=(9, 6), dpi=80)

    plt.subplot(1, 2, 1)
    plt.xlim(xmin - 0.05 * abs(xmin), xmax + 0.05 * xmax)
    plt.ylim(_YMIN, _YMAX)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Potential, eigenstates, <x>', fontsize=16)
    plt.xlabel('x [Bohr]', fontsize=16)
    plt.ylabel('Energie [Hartree]', fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.xaxis.set_label_position('bottom')

    for ii in range(minEV - 1, maxEV):
        if ii % 2 == 0:
            _COLOR = 'blue'
        else:
            _COLOR = 'red'
        plt.hlines(EVAL[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        _SCALE = _scale_plot(minEV, maxEV, EVAL, EVEC, ii, RTOL, ATOL)
        plt.plot(expx[ii], EVAL[ii], 'x', color='green', markersize=12,
                 markeredgewidth=1.5, zorder=3)
        plt.plot(xplot, _SCALE * EVEC[:, ii] + EVAL[ii], color=_COLOR,
                 linewidth=2.5, zorder=2)
    plt.plot(xplot, pot, color='black', linewidth=2, zorder=0)

    plt.subplot(1, 2, 2)
    for ii in range(minEV - 1, maxEV):
        plt.hlines(EVAL[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        plt.plot(unc[ii], EVAL[ii], marker='+', color='magenta',
                 markersize=17, markeredgewidth=1.85, zorder=2)
    plt.xlim(0, np.amax(unc) + 0.1 * np.amax(unc))
    plt.ylim(_YMIN, _YMAX)
    plt.yticks([])
    plt.xticks(fontsize=14)
    plt.title('sigma x', fontsize=16)
    plt.xlabel('[Bohr]', fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.show()


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

    pot_plot(_XMIN, _XMAX, _MIN_EV, _MAX_EV, _EVAL, _EVEC, _POT, _XPLOT,
             _YDIFF, _EXPX, _UNC)


if __name__ == '__main__':
    main()
