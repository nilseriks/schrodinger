"""Containing routines to visualize the potential, the wavefunctions, the
expected positions and the corresponding uncertainties. The wavefunctions
were visuealized in a plot with the expected positions. The uncertainties were
visualized in a seperate plot."""


import numpy as np
import matplotlib.pyplot as plt


def _scale_plot(min_ev, max_ev, energy, evec, index_ev, rtol, atol):
    """Calculates the multiplication factor which scales the wavefunctions for
    a better visualization in the graphical plot.

    Args:
        min_ev: Lower bound of the eigenvalues which should be visualized.
        max_ev: Upper bound of the eigenvalues which should be visualized.
        energy: Array of eigenvalues.
        evec: Array containing the wavefunctions as column vectors.
        index_ev: The indexEV'th wavefunction to calculate the multiplication
                 factor of.
        rtol: Relative tolerence to compare diffrent eigenvalues.
        atol: Absolute tolerence to compare diffrent eigenvalues.

    Returns:
        scale: Multiplication factor which scales the eigenvectors."""
    diff_list = []

    for kk in range(min_ev - 1, max_ev - 1):
        if not np.allclose(energy[kk + 1], energy[kk], atol=atol,
                           rtol=rtol):
            diff_list.append(abs(energy[kk + 1] - energy[kk]))
    scale = 0.4 * min(diff_list) * 1 / np.amax(abs(evec[:, index_ev]))
    return scale


def _plot_set_wf(xmin, xmax, ymin, ymax):
    plt.xlim(xmin - 0.05 * abs(xmin), xmax + 0.05 * xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Potential, eigenstates, <x>', fontsize=16)
    plt.xlabel('x [Bohr]', fontsize=16)
    plt.ylabel('Energy [Hartree]', fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.xaxis.set_label_position('bottom')


def _plot_set_unc(ymin, ymax, unc):
    plt.xlim(0, np.amax(unc) + 0.1 * np.amax(unc))
    plt.ylim(ymin, ymax)
    plt.yticks([])
    plt.xticks(fontsize=14)
    plt.title('sigma x', fontsize=16)
    plt.xlabel('[Bohr]', fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)


def pot_plot(xmin, xmax, min_ev, max_ev, energy, evec, pot, xplot, ydiff, expx,
             unc):
    """Creates a graphical plot. It shows the potential, the eigenvalues, the
    wavefunctions, the expected values of the position of the particle. And
    within a second plot it shows the uncertainty of the expected position.

    Args:
        xmin: Lower bound of the x values.
        xmax: Upper bound of the x values.
        min_ev: Lower bound of the eigenvalues which should be visualized.
        max_ev: Upper bound of the eigenvalues which should be visualized.
        energy: Array of eigenvalues.
        evec: Array containing the wavefunctions as column vectors.
        pot: Interpolation of the potential at the xplot values.
        xplot: Values were the potential is defined.
        ydiff: Absolute difference between the lowest pot-value and the highest
               eigenvalue.
        expx: Expected values of the position.
        unc: Uncertainty of the position.
    """
    atol = 0.05 * ydiff
    rtol = 0.05 * ydiff
    ymin = np.amin(pot) - 0.05 * ydiff
    max_scale = _scale_plot(min_ev, max_ev, energy, evec, max_ev - 1, rtol,
                            atol)
    ymax = (energy[max_ev - 1] + np.amax(max_scale * evec[:, max_ev - 1]) +
            0.05 * ydiff)

    plt.figure(figsize=(9, 6), dpi=80)

    plt.subplot(1, 2, 1)
    _plot_set_wf(xmin, xmax, ymin, ymax)

    for ii in range(min_ev - 1, max_ev):
        if ii % 2 == 0:
            color = 'blue'
        else:
            color = 'red'
        plt.hlines(energy[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        scale = _scale_plot(min_ev, max_ev, energy, evec, ii, rtol, atol)
        plt.plot(expx[ii], energy[ii], 'x', color='green', markersize=12,
                 markeredgewidth=1.5, zorder=3)
        plt.plot(xplot, scale * evec[:, ii] + energy[ii], color=color,
                 linewidth=2.5, zorder=2)
    plt.plot(xplot, pot, color='black', linewidth=2, zorder=0)

    plt.subplot(1, 2, 2)
    _plot_set_unc(ymin, ymax, unc)

    for ii in range(min_ev - 1, max_ev):
        plt.hlines(energy[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        plt.plot(unc[ii], energy[ii], marker='+', color='magenta',
                 markersize=17, markeredgewidth=1.85, zorder=2)


    plt.show()


def pot_plot2(xmin, xmax, min_ev, max_ev, energy, evec, pot, xplot, ydiff,
              expx, unc):
    """Creates a graphical plot. It shows the potential, the eigenvalues, the
    wavefunctions, the expected values of the position of the particle. And
    within a second plot it shows the uncertainty of the expected position.

    Args:
        xmin: Lower bound of the x values.
        xmax: Upper bound of the x values.
        min_ev: Lower bound of the eigenvalues which should be visualized.
        max_ev: Upper bound of the eigenvalues which should be visualized.
        energy: Array of eigenvalues.
        evec: Array containing the wavefunctions as column vectors.
        pot: Interpolation of the potential at the xplot values.
        xplot: Values were the potential is defined.
        ydiff: Absolute difference between the lowest pot-value and the highest
               eigenvalue.
        expx: Expected values of the position.
        unc: Uncertainty of the position.
    """
    scale = 0.4 * abs(energy - np.amin(pot)) * 1 / np.amax(abs(evec[:, 0]))
    ymin = energy - np.amax(scale * evec[:, 0]) + 0.05 * ydiff
    ymax = energy + np.amax(scale * evec[:, 0]) + 0.05 * ydiff

    plt.figure(figsize=(9, 6), dpi=80)

    plt.subplot(1, 2, 1)
    _plot_set_wf(xmin, xmax, ymin, ymax)

    for ii in range(min_ev - 1, max_ev):
        if ii % 2 == 0:
            color = 'blue'
        else:
            color = 'red'
        plt.hlines(energy, xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        plt.plot(expx, energy, 'x', color='green', markersize=12,
                 markeredgewidth=1.5, zorder=3)
        plt.plot(xplot, scale * evec[:, ii] + energy, color=color,
                 linewidth=2.5, zorder=2)
    plt.plot(xplot, pot, color='black', linewidth=2, zorder=0)

    plt.subplot(1, 2, 2)
    for ii in range(min_ev - 1, max_ev):
        plt.hlines(energy, xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        plt.plot(unc, energy, marker='+', color='magenta',
                 markersize=17, markeredgewidth=1.85, zorder=2)

    _plot_set_unc(ymin, ymax, unc)

    plt.show()
