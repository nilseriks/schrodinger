#!/usr/bin/env python3
"""This script visualizes the wavefunctions of a given potential."""


import numpy as np
import matplotlib.pyplot as plt
import calculus


#_YDIFF = abs(EVAL[_MAX_EV] - np.amin(pot_calc()))
ABSOLUTE_TOLERANCE = 0.5
RELATIVE_TOLERANCE = 0.5


def pot_plot(xmin, xmax, minEV, maxEV, pot, xplot, ydiff, EVAL, EVEC):
    _YMIN = np.amin(pot) - 0.05 * ydiff
    _YMAX = EVAL[maxEV - 1] + 0.1 * ydiff
    plt.figure(figsize=(4, 6), dpi=80)
    plt.xlim(xmin - 0.05 * abs(xmin), xmax + 0.05 * xmax)
    plt.ylim(_YMIN, _YMAX)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Potential, eigenstates, <x>', fontsize=16)
    plt.xlabel('x [Bohr]', fontsize=16)
    plt.ylabel('Energie [Hartree]', fontsize=16)
    ax.xaxis.set_label_position('bottom')
    for ii in range(minEV - 1, maxEV):
        if ii % 2 == 0:
            _COLOR = 'blue'
        else:
            _COLOR = 'red'
        plt.hlines(EVAL[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        DIFF_LIST = []
        '''
        for kk in range(minEV, maxEV - 1):
            if not np.allclose(EVAL[kk + 1], EVAL[kk], atol=ABSOLUTE_TOLERANCE,
                               rtol=RELATIVE_TOLERANCE):
                DIFF_LIST.append(abs(EVAL[kk + 1] - EVAL[kk]))
        # _SCALE = 0.2 * abs(_YMAX - _YMIN) / abs(_MAX_EV - _MIN_EV + 1) * 1 / np.amax(abs(EVEC[:, ii]))
        jj = ii
        while np.allclose(EVAL[jj + 1], EVAL[ii], atol=ABSOLUTE_TOLERANCE,
                          rtol=RELATIVE_TOLERANCE):
            jj += 1
        # _SCALE = 0.3 * abs(EVAL[jj + 1] - EVAL[ii]) * 1 / np.amax(abs(EVEC[:, ii]))
        # _SCALE = 0.6 * abs(EVAL[_MAX_EV] - EVAL[_MAX_EV - 1]) * 1 / np.amax(abs(EVEC[:, ii]))
        _SCALE = 0.3 * min(DIFF_LIST) * 1 / np.amax(abs(EVEC[:, ii]))
        '''
        _SCALE = 5
        plt.plot(xplot, _SCALE * EVEC[:, ii] + EVAL[ii], color=_COLOR,
                 linewidth=2.5, zorder=2)
    plt.plot(xplot, pot, color='black', linewidth=2, zorder=0)
    plt.show


def main():
    _DATA = calculus.file_io.read_files('files')
    _XPLOT = _DATA[2][:, 0]
    _XMIN = np.amin(_XPLOT)
    _XMAX = np.amax(_XPLOT)
    _EVAL = _DATA[0]
    #_MIN_EV = np.amin(_EVAL)
    _MIN_EV = 1
    #_MAX_EV = np.amax(_EVAL)
    _MAX_EV = 4
    _POT = _DATA[2][:, 1:]
    _EVEC = _DATA[3][:, 1:]

    pot_plot(_XMIN, _XMAX, _MIN_EV, _MAX_EV, _POT, _XPLOT, 10, _EVAL, _EVEC)


if __name__ == '__main__':
    main()
