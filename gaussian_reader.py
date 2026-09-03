"""
gaussian_reader.py

Parse Cartesian coordinates out of a Gaussian .com/.gjf input file.
"""

import re
import os
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


def write_gaussian_com(path, symbols, coords, title="principal-axis converted structure",
                        charge=0, multiplicity=1, route="#p hf/6-31g(d)",
                        chk=None, mem="4gb", nproc=8):
    """
    Write a Gaussian .com input file. Charge, multiplicity, and the route
    line (level of theory/basis set) are NOT known from a bare Cartesian
    structure -- the defaults here are placeholders and MUST be reviewed/
    edited before submitting a real Gaussian job.
    """
    if chk is None:
        chk = os.path.splitext(os.path.basename(path))[0] + ".chk"

    with open(path, "w") as f:
        f.write(f"%chk={chk}\n")
        f.write(f"%mem={mem}\n")
        f.write(f"%nproc={nproc}\n")
        f.write(f"{route}\n")
        f.write("\n")
        f.write(f"{title}\n")
        f.write("\n")
        f.write(f"{charge} {multiplicity}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.8f} {y:.8f} {z:.8f}\n")
        f.write("\n\n")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "GAMA-peptide/Gly12.com"
    symbols, coords = read_gaussian_com(path)
    print(f"Parsed {len(symbols)} atoms from {path}")
    print(f"Elements present: {sorted(set(symbols))}")
