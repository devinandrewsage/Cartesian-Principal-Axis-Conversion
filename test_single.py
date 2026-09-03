"""Run the principal-axis pipeline on a single .xyz file and print the
manual-style step-by-step output for verification."""

import sys
import numpy as np
from principal_axes import to_principal_axes, ATOMIC_MASSES

np.set_printoptions(precision=6, suppress=True)


def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, np.array(coords)


def bond_length(a, b):
    return np.linalg.norm(a - b)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_molecules/formaldehyde.xyz"
    symbols, coords = read_xyz(path)
    masses = np.array([ATOMIC_MASSES[s] for s in symbols])

    print(f"Loaded {path}: {len(symbols)} atoms -> {symbols}\n")

    new_coords, info = to_principal_axes(coords, masses)

    print("Center of mass:", info["com"])
    print("\nPrincipal moments of inertia:", info["principal_moments"])
    print("\nOriginal coordinates:")
    for s, c in zip(symbols, coords):
        print(f"  {s}: {c}")
    print("\nPrincipal-axis frame coordinates:")
    for s, c in zip(symbols, new_coords):
        print(f"  {s}: {c}")

    print("\nBounding box area (XY): before={:.4f}  after={:.4f}  shrinkage={:.1f}%".format(
        info["bbox_area_before"], info["bbox_area_after"],
        100 * (1 - info["bbox_area_after"] / info["bbox_area_before"])
    ))

    idx = {s: i for i, s in enumerate(symbols)}
    h_idx = [i for i, s in enumerate(symbols) if s == "H"][0]
    print("\n--- Verification: bond length invariance ---")
    for frame_name, c in [("original", coords), ("principal-axis", new_coords)]:
        d_CO = bond_length(c[idx["C"]], c[idx["O"]])
        d_CH = bond_length(c[idx["C"]], c[h_idx])
        print(f"{frame_name:>15}: C=O = {d_CO:.4f} A,  C-H = {d_CH:.4f} A")
