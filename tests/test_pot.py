#!/usr/bin/env python3
"""Script testing the interpolation of the potentials."""

import sys
import numpy as np
import pytest
sys.path.append('../..')
import calculus


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
    expected_pot = calculus.io.read_data(_DIRECTORYTEST, problem[1])

    inp = calculus.io.read_schrodinger(_DIRECTORYFILE, problem[0])
    xplot = np.linspace(inp['_XMIN'], inp['_XMAX'], num=inp['_NPOINT'],
                        endpoint=True)

    calculated_pot = calculus.calc.pot_calc(xplot, inp['_POT'],
                                            inp['_REG_TYPE'])

    assert np.allclose(expected_pot, calculated_pot, rtol=1e-14, atol=1e-14)
