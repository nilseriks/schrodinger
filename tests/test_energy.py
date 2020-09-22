#!/usr/bin/env python3
"""Script testing the solver for the one dimensional time independent
schrodinger equation."""

import numpy as np
import pytest
import calculus


_DIRECTORYFILE = 'tests'
_DIRECTORYTEST = 'tests/test_energy'


_LIST = [('inf_square_well.inp', 'E_inf_square_well.dat'),
         ('fin_square_well.inp', 'E_fin_square_well.dat'),
         ('harm_osc.inp', 'E_harm_osc.dat'),
         ('double_lin.inp', 'E_double_lin.dat'),
         ('double_spline.inp', 'E_double_spline.dat'),
         ('morse.inp', 'E_morse.dat')]


@pytest.mark.parametrize('problem', _LIST)
def test_energy(problem):
    """Testing environment for the energies of the following physical problems,
    with specific tolarence:
        infinite square well (rtol=1e-02, atol=1e-12)
        finite square well (rtol=1e-15, atol=1e-15)
        harmonic oscillator (rtol=1e-03, atol=1e-12)
        double oscillator (linear interpolation) (rtol=1e-15, atol=1e-15)
        double oscillator (spline interpolation) (rtol=1e-15, atol=1e-15)
        morse potential (rtol=1e-15, atol=1e-15).
    """
    expectede = calculus.file_io.read_data(_DIRECTORYTEST, problem[1])

    inp = calculus.io.read_schrodinger(_DIRECTORYFILE, problem[0])
    xplot = np.linspace(inp['_XMIN'], inp['_XMAX'], num=inp['_NPOINT'],
                        endpoint=True)
    pot = calculus.calc.pot_calc(xplot, inp['_POT'], inp['_REG_TYPE'])

    calculatede = calculus.calc._solve_seq(inp['_XMIN'], inp['_XMAX'],
                                           inp['_NPOINT'], inp['_MASS'],
                                           pot)[0][0:20]
    if problem[0] == 'harm_osc.inp':
        assert np.allclose(expectede, calculatede, rtol=1e-03, atol=1e-12)
    elif problem[0] == 'inf_square_well.inp':
        assert np.allclose(expectede, calculatede, rtol=1e-02, atol=1e-12)
    else:
        assert np.allclose(expectede, calculatede, rtol=1e-15, atol=1e-15)
