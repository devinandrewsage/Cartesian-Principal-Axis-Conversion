"""
principal_axes.py

Convert Cartesian atomic coordinates into a mass-weighted principal-axis
frame: center of mass at origin, axes aligned to the eigenvectors of the
moment-of-inertia tensor, sign/reflection convention enforced.
"""

import numpy as np
from scipy.linalg import eigh

ATOMIC_MASSES = {
    "H": 1.008, "He": 4.003,
    "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078,
    "Fe": 55.845, "Zn": 65.38,
    "Br": 79.904, "I": 126.904,
}


def center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()


def inertia_tensor(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Ixz = -np.sum(masses * x * z)
    Iyz = -np.sum(masses * y * z)
    return np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz],
    ])


def fix_rotation_convention(V: np.ndarray, coords: np.ndarray):
    V = V.copy()
    projections = coords @ V

    sign_ambiguous = False
    for k in range(3):
        abs_proj = np.abs(projections[:, k])
        sorted_proj = np.sort(abs_proj)[::-1]
        # Flag when the top two atoms are within 0.1% of each other in
        # magnitude -- this is the exact mechanism that broke formaldehyde
        # and CO2: the sign convention's choice becomes essentially a coin
        # flip decided by floating-point rounding under rotation.
        if len(sorted_proj) >= 2 and sorted_proj[0] > 0:
            if (sorted_proj[0] - sorted_proj[1]) / sorted_proj[0] < 1e-3:
                sign_ambiguous = True

        idx = np.argmax(abs_proj)
        if projections[idx, k] < 0:
            V[:, k] *= -1

    reflection_corrected = np.linalg.det(V) < 0
    if reflection_corrected:
        V[:, 2] *= -1
    return V, reflection_corrected, sign_ambiguous


def to_principal_axes(coords: np.ndarray, masses: np.ndarray):
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)

    com = center_of_mass(coords, masses)
    centered = coords - com

    I = inertia_tensor(centered, masses)
    eigvals, V = eigh(I)
    V, reflection_corrected, sign_ambiguous = fix_rotation_convention(V, centered)

    coords_new = centered @ V

    def bbox_area_xy(c):
        dx = c[:, 0].max() - c[:, 0].min()
        dy = c[:, 1].max() - c[:, 1].min()
        return dx * dy

    info = {
        "com": com,
        "rotation_matrix": V,
        "principal_moments": eigvals,
        "inertia_tensor": I,
        "reflection_corrected": reflection_corrected,
        "sign_ambiguous": sign_ambiguous,
        "bbox_area_before": bbox_area_xy(centered),
        "bbox_area_after": bbox_area_xy(coords_new),
    }
    return coords_new, info
