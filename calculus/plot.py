"""Containing routines to visualize the potential, the wavefunctions, the
expected positions and the corresponding uncertainties. The wavefunctions
are visualized in a plot with the expected positions. The uncertainties are
visualized in a seperate plot."""


import numpy as np
import matplotlib.pyplot as plt


def _scale_plot(min_ev, max_ev, energy, evec, index_ev, rtol, atol):
    """Calculates the multiplication factor which scales the wavefunctions for
    a better visualization in the graphical plot. The scaling factor normalizes
    the wavefunctions with respect to the highest absolute value. Then it
    scales the wavefunctions wit a factor 0.4 times the difference between the
    two nearest energies, excluded the degenderate eigenstates.

    Args:
        min_ev (int): Lower bound of the eigenvalues which should be visualized
        max_ev (int): Upper bound of the eigenvalues which should be visualized
        energy (1darray): Array of eigenvalues.
        evec (ndarray): Array containing the wavefunctions as column vectors
        index_ev (int): The indexEV'th wavefunction to calculate the
          multiplication factor of.
        rtol (float): Relative tolerance to compare different eigenvalues
        atol (float): Absolute tolerance to compare different eigenvalues

    Returns:
        int: Multiplication factor which scales the eigenvectors
    """
    diff_list = []

    # Saves the difference between the energy values in a list, if the energies
    # are not to close to each other.
    for kk in range(0, max_ev - min_ev):
        if not np.allclose(energy[kk + 1], energy[kk], atol=atol, rtol=rtol):
            diff_list.append(abs(energy[kk + 1] - energy[kk]))
    scale = 0.5 * min(diff_list) * 1 / np.amax(abs(evec[:, index_ev]))
    return scale


def _plot_set_wf(xmin, xmax, ymin, ymax):
    plt.xlim(xmin - 0.05 * abs(xmin), xmax + 0.05 * xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'Potential, eigenstates, $\langle x\rangle$', fontsize=16)
    plt.xlabel('$x$ [Bohr]', fontsize=16)
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
    plt.title(r'$\sigma_x$', fontsize=16)
    plt.xlabel('[Bohr]', fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)


def pot_plot_multi(xmin, xmax, min_ev, max_ev, energy, evec, pot, xplot, ydiff,
                   expx, unc, scale):
    """Creates a graphical plot. It shows the potential, the eigenvalues, the
    wavefunctions, the expected values of the position of the particle. Within
    a second plot it shows the uncertainty of the expected position.

    Args:
        xmin (int): Lower bound of the x values
        xmax (int): Upper bound of the x values
        min_ev (int): Lower bound of the eigenvalues which should be visualized
        max_ev (int): Upper bound of the eigenvalues which should be visualized
        energy (1darray): Array of eigenvalues
        evec (ndarray): Array containing the wavefunctions as column vectors
        pot (1darray): Interpolation of the potential at the xplot values
        xplot (1darray): Values where the potential is defined
        ydiff (int): Absolute difference between the lowest and the highest
            eigenvalue
        expx (1darray): Expected values of the position
        unc (1darray): Uncertainty of the position.
        scale (float): Scaling factor of the wavefunctions.
    """
    atol = 0.02 * ydiff
    rtol = 0.02 * ydiff
    if scale is None:
        max_scale = _scale_plot(min_ev, max_ev, energy, evec, -1, rtol,
                                atol)
        min_scale = _scale_plot(min_ev, max_ev, energy, evec, 0, rtol,
                                atol)
    else:
        max_scale = scale
        min_scale = scale
    ymin = np.amin(energy) - np.amax(min_scale * evec[:, 0]) - 0.05 * ydiff
    ymax = energy[-1] + np.amax(max_scale * evec[:, -1]) + 0.05 * ydiff

    plt.figure(figsize=(9, 6), dpi=80)

    plt.subplot(1, 2, 1)
    _plot_set_wf(xmin, xmax, ymin, ymax)

    for ii in range(0, max_ev - min_ev + 1):
        if ii % 2 == 0:
            color = 'blue'
        else:
            color = 'red'
        plt.hlines(energy[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        if scale is None:
            scale = _scale_plot(min_ev, max_ev, energy, evec, ii, rtol, atol)
        plt.plot(expx[ii], energy[ii], 'x', color='green', markersize=12,
                 markeredgewidth=1.5, zorder=3)
        plt.plot(xplot, scale * evec[:, ii] + energy[ii], color=color,
                 linewidth=2.5, zorder=2)
    plt.plot(xplot, pot, color='black', linewidth=2, zorder=0)

    plt.subplot(1, 2, 2)
    _plot_set_unc(ymin, ymax, unc)

    for ii in range(0, max_ev - min_ev + 1):
        plt.hlines(energy[ii], xmin, xmax, color='lightgray', linewidth=2.5,
                   zorder=1)
        plt.plot(unc[ii], energy[ii], marker='+', color='magenta',
                 markersize=17, markeredgewidth=1.85, zorder=2)

    plt.show()


def pot_plot_one(xmin, xmax, energy, evec, pot, xplot, ydiff, expx, unc,
                 scale):
    """Creates a graphical plot. It shows the potential, the eigenvalues, the
    wavefunctions, the expected values of the position of the particle. And
    within a second plot it shows the uncertainty of the expected position.

    Args:
        xmin (int): Lower bound of the x values
        xmax (int): Upper bound of the x values
        energy (1darray): Array of eigenvalues
        evec (ndarray): Array containing the wavefunctions as column vectors
        pot (1darray): Interpolation of the potential at the xplot values
        xplot (1darray): Values where the potential is defined
        ydiff (int): Absolute difference between the lowest and the highest
            eigenvalue
        expx (1darray): Expected values of the position
        unc (1darray): Uncertainty of the position.
        scale (float): Scaling factor of the wavefunctions.
    """
    if scale is None:
        scale = 0.4 * abs(energy - np.amin(pot)) * 1 / np.amax(abs(evec[:, 0]))
    ymin = energy - np.amax(scale * evec[:, 0]) - 0.05 * ydiff
    ymax = energy + np.amax(scale * evec[:, 0]) + 0.05 * ydiff

    plt.figure(figsize=(9, 6), dpi=80)

    plt.subplot(1, 2, 1)
    _plot_set_wf(xmin, xmax, ymin, ymax)

    plt.hlines(energy, xmin, xmax, color='lightgray', linewidth=2.5,
               zorder=1)
    plt.plot(expx, energy, 'x', color='green', markersize=12,
             markeredgewidth=1.5, zorder=3)
    plt.plot(xplot, scale * evec + energy, color='blue',
             linewidth=2.5, zorder=2)
    plt.plot(xplot, pot, color='black', linewidth=2, zorder=0)

    plt.subplot(1, 2, 2)
    plt.hlines(energy, xmin, xmax, color='lightgray', linewidth=2.5,
               zorder=1)
    plt.plot(unc, energy, marker='+', color='magenta',
             markersize=17, markeredgewidth=1.85, zorder=2)

    _plot_set_unc(ymin, ymax, unc)

    plt.show()
