"""
verify_setup.py

Consolidated regression test. Run this any time after changing the code
or setting up a new environment, BEFORE trusting output from a real batch
run.
"""

import numpy as np
from principal_axes import to_principal_axes, ATOMIC_MASSES
from batch_principal_axes import read_structure

np.set_printoptions(precision=6, suppress=True)


def rotation_matrix(angle_deg, axis="z"):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z": return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def bond_length(a, b):
    return np.linalg.norm(a - b)


def check_water_reference():
    coords = np.array([
        [0.046, -0.046, 0.0],
        [0.167, 0.904, 0.0],
        [-0.904, -0.167, 0.0],
    ])
    masses = np.array([15.999, 1.008, 1.008])
    _, info = to_principal_axes(coords, masses)
    area = info["bbox_area_after"]
    passed = abs(area - 0.888) < 0.01
    print(f"[{'PASS' if passed else 'FAIL'}] Water reference case: "
          f"bounding box = {area:.4f} A^2 (expect ~0.887-0.888)")
    return passed


def check_invariance(path, label):
    symbols, coords_orig = read_structure(path)
    masses = np.array([ATOMIC_MASSES[s] for s in symbols])

    new_A, info_A = to_principal_axes(coords_orig, masses)

    R = rotation_matrix(51, "x") @ rotation_matrix(84, "y") @ rotation_matrix(17, "z")
    translation = np.array([22.5, -8.1, 41.3])
    coords_rot = coords_orig @ R.T + translation
    new_B, info_B = to_principal_axes(coords_rot, masses)

    max_diff = np.abs(new_A - new_B).max()
    passed = max_diff < 1e-6

    d_orig = bond_length(coords_orig[0], coords_orig[1])
    d_new = bond_length(new_A[0], new_A[1])
    bond_ok = abs(d_orig - d_new) < 1e-6

    print(f"[{'PASS' if passed else 'FAIL'}] {label} ({len(symbols)} atoms): "
          f"invariance max diff = {max_diff:.2e}   "
          f"bond check: {'OK' if bond_ok else 'MISMATCH'}")
    return passed and bond_ok


def check_sign_ambiguity_detection():
    """Confirms the sign_ambiguous flag correctly identifies known-unstable
    cases (formaldehyde, CO2) and does NOT false-positive on a known-stable
    asymmetric case (formic acid)."""
    cases = [
        (["C", "O", "H", "H"], np.array([
            [0, 0, 0], [0, -1.208, 0], [0.9495, 0.5865, 0], [-0.9495, 0.5865, 0]
        ]), "Formaldehyde", True),
        (["O", "C", "O"], np.array([
            [-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]
        ]), "CO2", True),
        (["C", "O", "O", "H", "H"], np.array([
            [0, 0, 0], [1.202, 0, 0], [-0.7684, 1.1015, 0],
            [-0.615, -0.9084, 0], [-0.1593, 1.859, 0]
        ]), "Formic acid", False),
    ]
    all_ok = True
    for symbols, coords, label, expected in cases:
        masses = np.array([ATOMIC_MASSES[s] for s in symbols])
        _, info = to_principal_axes(coords, masses)
        got = info["sign_ambiguous"]
        ok = got == expected
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] Sign-ambiguity detection ({label}): "
              f"expected {expected}, got {got}")
    return all_ok


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFICATION SUITE")
    print("=" * 70)

    results = []
    results.append(check_water_reference())

    import os
    peptide_dir = "GAMA-peptide"
    if os.path.isdir(peptide_dir):
        results.append(check_invariance(
            os.path.join(peptide_dir, "alpha_helix_ala_18.com"), "Alpha-helix Ala18"))
        results.append(check_invariance(
            os.path.join(peptide_dir, "Segglu_gama.com"), "Segglu"))
    else:
        print("[SKIPPED] Real peptide test data not found at", peptide_dir)

    check_sign_ambiguity_detection()

    print("=" * 70)
    if all(results):
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- do not trust batch results until fixed")
    print("=" * 70)
