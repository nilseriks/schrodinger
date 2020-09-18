#!/usr/bin/env python3
"""This script visualizes the wavefunctions of a given potential."""


import numpy as np
import matplotlib.pyplot as plt
import calculus


def _scale_plot(minEV, maxEV, EVAL, EVEC, indexEV, RTOL, ATOL):
    DIFF_LIST = []

    for kk in range(minEV - 1, maxEV - 1):
        if not np.allclose(EVAL[kk + 1], EVAL[kk], atol=ATOL,
                           rtol=RTOL):
            DIFF_LIST.append(abs(EVAL[kk + 1] - EVAL[kk]))
    _SCALE = 0.4 * min(DIFF_LIST) * 1 / np.amax(abs(EVEC[:, indexEV]))
    return _SCALE


def pot_plot(xmin, xmax, minEV, maxEV, pot, xplot, ydiff, EVAL, EVEC):
    ATOL = 0.05 * ydiff
    RTOL = 0.05 * ydiff
    _YMIN = np.amin(pot) - 0.05 * ydiff
    _max_scale = _scale_plot(minEV, maxEV, EVAL, EVEC, maxEV - 1, RTOL, ATOL)
    print("max", _max_scale)
    _YMAX = EVAL[maxEV - 1] + np.amax(_max_scale * EVEC[:, maxEV - 1]) + 0.05 * ydiff
    print(_YMAX)
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
        _SCALE = _scale_plot(minEV, maxEV, EVAL, EVEC, ii, RTOL, ATOL)
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
    inp = calculus.file_io.read_schrodinger('files', 'schrodinger.inp')
    _MIN_EV = inp['_MIN_EV']
    _MAX_EV = inp['_MAX_EV']
    print(_MIN_EV, _MAX_EV)
    _POT = _DATA[2][:, 1:]
    _EVEC = _DATA[3][:, 1:]
    _YDIFF = abs(_EVAL[_MAX_EV - 1] - np.amin(_POT))
    print(_YDIFF)

    pot_plot(_XMIN, _XMAX, _MIN_EV, _MAX_EV, _POT, _XPLOT, _YDIFF, _EVAL,
             _EVEC)


if __name__ == '__main__':
    main()
