*****************
API Documentation
*****************


Calc
====

.. automodule:: calc
   :members:

Following code is a snippet of the calc module. It shows the function
interpolating the potential based on the user input.

    .. code-block:: python

       import numpy as np
       import scipy as sp
       import scipy.interpolate
       import calculus


       def pot_calc(xplot, discrete_pot, interpoltype):
           """Interpolates the Potential for given data points.

           Args:
               xplot: Array of the x values.
               pot: Array with the data points of the potential.
               interpoltype: Type of the interpolation.

           Returns:
               VV: Array with values of the potential at the points of xplot.
           """
           xx = discrete_pot[:, 0]
           yy = discrete_pot[:, 1]
           if interpoltype == 'linear':
               VV = scipy.interpolate.interp1d(xx, yy, 'linear')
               return VV(xplot)
           elif interpoltype == 'polynomial':
               VV = sp.interpolate.barycentric_interpolate(xx, yy, xplot)
               return VV
           elif interpoltype == 'cspline':
               VV = sp.interpolate.CubicSpline(xx, yy, bc_type='natural')
               return VV(xplot)

Plot
====

.. automodule:: plot
   :members:

Following code is a snippet of the plot module. It shows the function
that creates the graphical interpretation of the plot.

    .. code-block:: python

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
           ax = plt.gca()
           ax.spines['top'].set_linewidth(1.2)
           ax.spines['right'].set_linewidth(1.2)
           ax.spines['bottom'].set_linewidth(1.2)
           ax.spines['left'].set_linewidth(1.2)
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
