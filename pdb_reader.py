"""
pdb_reader.py

Read/write PDB format structures, with correct element-symbol
disambiguation. This is the file where the "CA" problem lives: in a PDB
ATOM record, "CA" almost always means the alpha-carbon (element Carbon),
NOT the element calcium -- but a HETATM record named "CA" (a bound
calcium ion) genuinely IS calcium.

Strategy (in priority order):
  1. Use the explicit element column (columns 77-78) when present.
  2. If missing (legacy files), fall back to a heuristic that only
     attempts a two-letter element match for HETATM records.
"""

import numpy as np

TWO_LETTER_HETERO_ELEMENTS = {
    "CL", "BR", "FE", "ZN", "MG", "CA", "NA", "MN", "CU", "NI",
    "CO", "SE", "LI", "AL", "SI", "AS", "AG", "AU", "PT", "HG",
}


def element_from_pdb_fields(atom_name, element_col, record_type):
    if element_col and element_col.strip():
        return element_col.strip().capitalize(), False

    name = atom_name.strip()
    if not name:
        raise ValueError("Empty atom name and no element column -- cannot determine element.")

    if name[0].isdigit():
        name = name[1:]

    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        raise ValueError(f"Could not extract any element letters from atom name '{atom_name}'.")

    if record_type == "HETATM" and len(letters) >= 2:
        two_letter = letters[:2].capitalize()
        if two_letter.upper() in TWO_LETTER_HETERO_ELEMENTS:
            return two_letter, True

    return letters[0].upper(), True


def read_pdb(path):
    symbols, coords = [], []
    n_ambiguous = 0

    with open(path) as f:
        for line in f:
            record = line[0:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            if len(line) < 54:
                continue

            atom_name = line[12:16]
            element_col = line[76:78] if len(line) >= 78 else ""

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            element, ambiguous = element_from_pdb_fields(atom_name, element_col, record)
            if ambiguous:
                n_ambiguous += 1

            symbols.append(element)
            coords.append([x, y, z])

    return symbols, np.array(coords), n_ambiguous


def write_pdb(path, symbols, coords, res_name="LIG", chain="A"):
    with open(path, "w") as f:
        for i, (s, (x, y, z)) in enumerate(zip(symbols, coords), start=1):
            atom_name = s.ljust(3) if len(s) <= 3 else s[:4]
            f.write(
                f"ATOM  {i:>5} {atom_name:<4}{res_name:<3} {chain}{1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}"
                f"{'':>10}{s.rjust(2)}\n"
            )
        f.write("END\n")
