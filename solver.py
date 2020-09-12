#!/usr/bin/env python3

import calculus


_DIRECTORY = 'files'
_FILE = 'schrodinger.inp'


def main():
    """Main function to solve the one dimensiional time independent schrodinger
    equation."""
    # Read out the informations of the schrodinger.inp file.
    inp_data = calculus.io.read_schrodinger(_DIRECTORY, _FILE)
    _MASS = inp_data["_MASS"]
    _XMIN = float(inp_data["plot_set"][0])
    _XMAX = float(inp_data["plot_set"][1])
    _NPOINT = int(inp_data["plot_set"][2])
    _MIN_EV = int(inp_data["evalues"][0])
    _MAX_EV = int(inp_data["evalues"][1])
    _REG_TYPE = inp_data["regression"]
    _INTERPOLATE_NR = int(inp_data["interpolate_nr"])
    _POT = inp_data['pot']


if __name__ == '__main__':
    main()
