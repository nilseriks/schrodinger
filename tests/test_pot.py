#!/usr/bin/env python3
"""Script testing the interpolation of the potentials."""


import numpy as np
import calculus
import pytest


_DIRECTORYFILE = 'tests'
_DIRECTORYTEST = 'tests/test_potential'


_LIST = [('inf_square_well.inp', 'pot_inf_square_well.dat'),
         ('fin_square_well.inp', 'pot_fin_square_well.dat'),
         ('harm_osc.inp', 'pot_harm_osc.dat'),
         ('double_lin.inp', 'pot_double_lin.dat'),
         ('double_spline.inp', 'pot_double_spline.dat'),
         ('morse.inp', 'pot_morse.dat')]


@pytest.mark.parametrize('problem', _LIST)
def test_pot(problem):
    """Test for the interpolation of the potentials. Following potentials will
    be tested with rtol=1e-14, atol=1e-14:
        infinite square well
        finite square well
        harmonic oscillator
        double oscillator (linear interpolation)
        double oscillator (spline interpolation)
        morse potential.
    """
    expected_pot = calculus.file_io.read_data(_DIRECTORYTEST, problem[1])

    inp_data = calculus.io.read_schrodinger(_DIRECTORYFILE, problem[0])
    _XMIN = float(inp_data["plot_set"][0])
    _XMAX = float(inp_data["plot_set"][1])
    _NPOINT = int(inp_data["plot_set"][2])
    _REG_TYPE = inp_data["regression"]
    _XPLOT = np.linspace(_XMIN, _XMAX, num=_NPOINT, endpoint=True)

    calculated_pot = calculus.calc.pot_calc(_XPLOT, inp_data['pot'], _REG_TYPE)

    assert np.allclose(expected_pot, calculated_pot, rtol=1e-14, atol=1e-14)
