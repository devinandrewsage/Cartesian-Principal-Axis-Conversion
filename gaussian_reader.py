"""
gaussian_reader.py

Parse Cartesian coordinates out of a Gaussian .com/.gjf input file.
"""

import re
import numpy as np


def read_gaussian_com(path):
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]

    charge_mult_pattern = re.compile(r"^\s*-?\d+\s+\d+\s*$")
    start_idx = None
    for i, line in enumerate(lines):
        if charge_mult_pattern.match(line):
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Could not find charge/multiplicity line in {path}")

    symbols, coords = [], []
    atom_pattern = re.compile(
        r"^\s*([A-Za-z]{1,2})\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*$"
    )
    for line in lines[start_idx:]:
        if line.strip() == "":
            if symbols:
                break
            else:
                continue
        m = atom_pattern.match(line)
        if not m:
            break
        sym, x, y, z = m.groups()
        symbols.append(sym)
        coords.append([float(x), float(y), float(z)])

    return symbols, np.array(coords)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "GAMA-peptide/Gly12.com"
    symbols, coords = read_gaussian_com(path)
    print(f"Parsed {len(symbols)} atoms from {path}")
    print(f"Elements present: {sorted(set(symbols))}")
