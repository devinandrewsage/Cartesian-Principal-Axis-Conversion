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

NEAR_DEGENERATE_THRESHOLD = 0.02   # moments within 2% of each other
SIGN_AMBIGUOUS_THRESHOLD = 1e-3    # top-2 projections within 0.1% of each other


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
        if len(sorted_proj) >= 2 and sorted_proj[0] > 0:
            if (sorted_proj[0] - sorted_proj[1]) / sorted_proj[0] < SIGN_AMBIGUOUS_THRESHOLD:
                sign_ambiguous = True

        idx = np.argmax(abs_proj)
        if projections[idx, k] < 0:
            V[:, k] *= -1

    reflection_corrected = np.linalg.det(V) < 0
    if reflection_corrected:
        V[:, 2] *= -1
    return V, reflection_corrected, sign_ambiguous


def validate_input(coords: np.ndarray, masses: np.ndarray):
    """
    Check for common real-data problems before doing any computation, and
    raise a clear, specific error rather than letting a cryptic failure
    happen several function calls deep inside the linear algebra.
    """
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if coords.shape[0] == 0:
        raise ValueError("Structure has 0 atoms -- cannot compute an orientation.")

    if coords.shape[0] != masses.shape[0]:
        raise ValueError(
            f"Mismatched atom count: {coords.shape[0]} coordinate rows but "
            f"{masses.shape[0]} masses."
        )

    if coords.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape (N, 3), got {coords.shape}.")

    nan_rows = np.where(np.isnan(coords).any(axis=1))[0]
    if len(nan_rows) > 0:
        raise ValueError(f"NaN coordinate(s) found at atom index(es): {list(nan_rows)}.")

    inf_rows = np.where(np.isinf(coords).any(axis=1))[0]
    if len(inf_rows) > 0:
        raise ValueError(f"Infinite coordinate(s) found at atom index(es): {list(inf_rows)}.")

    if np.any(masses <= 0):
        bad_idx = np.where(masses <= 0)[0]
        raise ValueError(
            f"Non-positive mass(es) found at atom index(es): {list(bad_idx)}. "
            f"All atomic masses must be > 0."
        )

    if coords.shape[0] == 1:
        import warnings
        warnings.warn(
            "Structure has only 1 atom -- orientation is undefined (all "
            "moments of inertia are zero); returning the atom at the origin."
        )


def to_principal_axes(coords: np.ndarray, masses: np.ndarray):
    validate_input(coords, masses)
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
