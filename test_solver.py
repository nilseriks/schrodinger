#!/usr/bin/env python3
"""Script testing the solver for the one dimensional time independent
schrodinger equation."""

import numpy as np
import calculus
import pytest


_DIRECTORYFILE = 'tests'
_DIRECTORYTEST = 'tests/test_energy'


_LIST = [('inf_square_well.inp', 'E_inf_square_well.dat'),
         ('harm_osc.inp', 'E_harm_osc.dat')]


@pytest.mark.parametrize('problem', _LIST)
def test_energy3(problem):
    expectedE = calculus.file_io.read_data(_DIRECTORYTEST, problem[1])
    inp_data = calculus.io.read_schrodinger(_DIRECTORYFILE, problem[0])
    _MASS = inp_data["_MASS"]
    _XMIN = float(inp_data["plot_set"][0])
    _XMAX = float(inp_data["plot_set"][1])
    _NPOINT = int(inp_data["plot_set"][2])
    _REG_TYPE = inp_data["regression"]
    _XPLOT = np.linspace(_XMIN, _XMAX, num=_NPOINT, endpoint=True)
    _POT = calculus.calc.pot_calc(_XPLOT, inp_data['pot'], _REG_TYPE)
    calculatedE = calculus.calc._solve_seq(_XMIN, _XMAX, _NPOINT, _MASS,
                                           _POT)[0][0:100]
    if problem[0] == 'schrodinger.inp':
        assert np.allclose(expectedE, calculatedE, rtol=1e-02, atol=1e-12)
    else:
        assert np.allclose(expectedE, calculatedE, rtol=1, atol=1e-12)




'''
def wf_inf_square_well(xplot, xmin, xmax, npoint, wf_number):
    _LENGTH = abs(xmin - xmax)
    PSI = np.zeros((npoint, wf_number), dtype=float)
    for nn in range(wf_number):
        PSI[:, nn] = np.sqrt(2 / _LENGTH) * np.sin((nn + 1) * np.pi * xplot
                             / _LENGTH)
    return PSI

#_POT2 = calculus.calc.pot_calc(_XPLOT, _POT, 'linear')
print(wf_inf_square_well(_XPLOT2, 0, 4, _NPOINT, 20))
print(calculus.calc._solve_seq(_XMIN, _XMAX, _NPOINT, 1, _POT2)[1][:, 0:20])


def test_inf_square_well():
    _POT2 = calculus.calc.pot_calc(_XPLOT, _POT, 'linear')
    AA = wf_inf_square_well(_XPLOT2, 0, 4, _NPOINT, 20)
    BB = calculus.calc._solve_seq(_XMIN, _XMAX, _NPOINT, 1, _POT2)[1][:, 0:20]
    for nn in range(20):
        BB[:, nn] = BB[:, nn] * 1 / np.sum(BB[:, nn])
    assert np.allclose(AA, BB, rtol=1, atol=1)
    '''
