"""
generate_test_molecules.py

Builds a 50-molecule benchmark test suite spanning 2 to ~150 atoms,
covering polar/nonpolar, covalent/non-covalent, chiral pairs, and
molecules deliberately chosen to stress the eigenvalue/sign-convention
edge cases discovered during development (exact degeneracy, near
degeneracy, sign-ambiguity).

Hand-built molecules use literature-standard bond lengths/angles for
exact, controlled geometry. Larger/typical organic molecules are
generated via RDKit (SMILES -> 3D embedding -> MMFF optimization).

Run this once to populate test_molecules/ with all 50 structures.
"""

import os
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
    print("WARNING: RDKit not installed -- run `pip install rdkit` first. "
          "Hand-built molecules will still be generated.")

OUT_DIR = "test_molecules"
os.makedirs(OUT_DIR, exist_ok=True)

manifest = []


def write_xyz(name, symbols, coords, comment=""):
    path = os.path.join(OUT_DIR, f"{name}.xyz")
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
    return path


def from_rdkit(name, smiles, category, note="", seed=42):
    if not HAVE_RDKIT:
        print(f"SKIPPED {name}: RDKit not available")
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"FAILED to parse SMILES for {name}: {smiles}")
        return
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        print(f"FAILED to embed {name}")
        return
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = conf.GetPositions()
    write_xyz(name, symbols, coords, comment=f"{name} ({smiles})")
    manifest.append((name, category, f"{len(symbols)} atoms; {note}"))


# CATEGORY A: Diatomics / small linear -- extreme degeneracy cases

def diatomic(name, sym1, sym2, bond_length, category, note):
    coords = np.array([[0, 0, 0], [bond_length, 0, 0]])
    write_xyz(name, [sym1, sym2], coords, comment=name)
    manifest.append((name, category, note))

diatomic("N2", "N", "N", 1.098, "A: diatomic", "nonpolar, homonuclear -- both moments exactly 0/degenerate")
diatomic("O2", "O", "O", 1.208, "A: diatomic", "nonpolar, homonuclear")
diatomic("CO", "C", "O", 1.128, "A: diatomic", "polar, heteronuclear")
diatomic("HF", "H", "F", 0.917, "A: diatomic", "strongly polar")
diatomic("HCl", "H", "Cl", 1.275, "A: diatomic", "polar")

coords = np.array([[0.046, -0.046, 0.0], [0.167, 0.904, 0.0], [-0.904, -0.167, 0.0]])
write_xyz("H2O", ["O", "H", "H"], coords, comment="water, reference case")
manifest.append(("H2O", "A: small polar", "3 atoms; reproduces paper's worked example"))

coords = np.array([[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
write_xyz("CO2", ["O", "C", "O"], coords, comment="carbon dioxide, linear")
manifest.append(("CO2", "A: linear, EDGE CASE", "3 atoms; EXACT eigenvalue degeneracy + sign-ambiguity (validated unstable)"))

coords = np.array([[0, 0, 0], [1.066, 0, 0], [2.632, 0, 0]])
write_xyz("HCN", ["H", "C", "N"], coords, comment="hydrogen cyanide, linear")
manifest.append(("HCN", "A: linear polar", "3 atoms; linear -- perpendicular moments exactly degenerate (all linear molecules), but no atom-projection tie (heteronuclear)"))


# CATEGORY B: Small polar/nonpolar covalent, more edge cases

coords = np.array([[0, 0, 0], [0, -1.208, 0], [0.9495, 0.5865, 0], [-0.9495, 0.5865, 0]])
write_xyz("formaldehyde", ["C", "O", "H", "H"], coords, comment="formaldehyde")
manifest.append(("formaldehyde", "B: small polar, EDGE CASE", "4 atoms; C2v symmetry, sign-ambiguity (validated unstable)"))

r, theta = 1.012, np.radians(106.7)
h_positions = [np.array([r*np.sin(theta)*np.cos(a), r*np.sin(theta)*np.sin(a), r*np.cos(theta)])
               for a in [0, 2*np.pi/3, 4*np.pi/3]]
coords = np.array([[0, 0, 0]] + h_positions)
write_xyz("ammonia", ["N", "H", "H", "H"], coords, comment="ammonia, C3v")
manifest.append(("ammonia", "B: small polar, EDGE CASE", "4 atoms; C3v symmetric top -- 2 of 3 moments EXACTLY degenerate"))

d = 1.087
scale = d / np.sqrt(3)
h_dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * scale
coords = np.vstack([[0, 0, 0], h_dirs])
write_xyz("methane", ["C", "H", "H", "H", "H"], coords, comment="methane, Td spherical top")
manifest.append(("methane", "B: small nonpolar, EXTREME EDGE CASE", "5 atoms; Td symmetry -- ALL 3 moments exactly equal (spherical top)"))

coords = np.array([[0, 0, 0], [1.061, 0, 0], [1.203, 0, 0], [2.264, 0, 0]])
write_xyz("acetylene", ["H", "C", "C", "H"], coords, comment="acetylene, linear D-infinity-h")
manifest.append(("acetylene", "B: linear nonpolar, EDGE CASE", "4 atoms; linear, homonuclear-symmetric -- exact tie risk"))

if HAVE_RDKIT:
    from_rdkit("H2S", "S", "B: small polar", "bent, heavier S")
    from_rdkit("ozone", "[O-][O+]=O", "B: small bent", "bent, resonance-delocalized")
    from_rdkit("ethylene", "C=C", "B: small nonpolar", "planar")
    from_rdkit("hydrogen_peroxide", "OO", "B: small polar", "nonplanar, C2 symmetry")


# Formic acid -- established asymmetric baseline
C = np.array([0.0, 0.0, 0.0])
O1 = C + 1.202 * np.array([1.0, 0.0, 0.0])
ang = np.radians(124.9)
O2 = C + 1.343 * np.array([np.cos(ang), np.sin(ang), 0.0])
ang2 = np.radians(-124.1)
H1 = C + 1.097 * np.array([np.cos(ang2), np.sin(ang2), 0.0])
vec = C - O2
ang_base = np.arctan2(vec[1], vec[0])
ang3 = ang_base + np.radians(106.3)
H2 = O2 + 0.972 * np.array([np.cos(ang3), np.sin(ang3), 0.0])
write_xyz("formic_acid", ["C", "O", "O", "H", "H"], np.array([C, O1, O2, H1, H2]),
          comment="formic acid, asymmetric baseline")
manifest.append(("formic_acid", "B: small polar baseline", "5 atoms; no symmetry -- validated stable under rotation"))


# CATEGORY C: Chirality pairs -- confirm mirror images stay distinct
if HAVE_RDKIT:
    from_rdkit("R_bromochlorofluoromethane", "F[C@H](Cl)Br", "C: chiral pair", "simplest chiral molecule, R enantiomer")
    from_rdkit("S_bromochlorofluoromethane", "F[C@@H](Cl)Br", "C: chiral pair", "simplest chiral molecule, S enantiomer")
    from_rdkit("L_alanine", "N[C@@H](C)C(=O)O", "C: chiral pair", "natural amino acid form")
    from_rdkit("D_alanine", "N[C@H](C)C(=O)O", "C: chiral pair", "mirror image of natural form")


# CATEGORY D: High-symmetry "extreme" cases -- degenerate moments
if HAVE_RDKIT:
    from_rdkit("benzene", "c1ccccc1", "D: aromatic, EXTREME", "D6h -- 2 of 3 moments exactly degenerate (planar symmetric ring)")
    from_rdkit("cyclohexane", "C1CCCCC1", "D: nonpolar ring", "chair conformation")
    from_rdkit("naphthalene", "c1ccc2ccccc2c1", "D: aromatic", "fused bicyclic, planar symmetric")
    from_rdkit("adamantane", "C1C2CC3CC1CC(C2)C3", "D: cage, EXTREME", "Td cage -- near-spherical-top, highly degenerate")
    from_rdkit("cubane", "C12C3C4C1C5C4C3C25", "D: cage, EXTREME", "Oh cage -- near-perfect spherical top")


# CATEGORY E: Common polar organics -- realistic asymmetric cases
if HAVE_RDKIT:
    from_rdkit("methanol", "CO", "E: polar organic", "")
    from_rdkit("ethanol", "CCO", "E: polar organic", "")
    from_rdkit("acetone", "CC(=O)C", "E: polar organic", "")
    from_rdkit("acetic_acid", "CC(=O)O", "E: polar organic", "")
    from_rdkit("pyridine", "c1ccncc1", "E: aromatic polar", "")
    from_rdkit("phenol", "c1ccc(cc1)O", "E: aromatic polar", "")
    from_rdkit("aniline", "c1ccc(cc1)N", "E: aromatic polar", "")
    from_rdkit("toluene", "Cc1ccccc1", "E: aromatic nonpolar", "")


# CATEGORY G: Symmetric ions
if HAVE_RDKIT:
    from_rdkit("ammonium", "[NH4+]", "G: ion, EXTREME", "Td, fully degenerate like methane")
    from_rdkit("hydronium", "[OH3+]", "G: ion", "pyramidal cation")
    from_rdkit("acetate", "CC(=O)[O-]", "G: ion", "")


# CATEGORY F: Non-covalent clusters

def build_water_monomer(offset=np.zeros(3), rot=None):
    coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    if rot is not None:
        coords = coords @ rot.T
    return ["O", "H", "H"], coords + offset

def rotmat(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == "y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    if axis == "z": return np.array([[c,-s,0],[s,c,0],[0,0,1]])

s1, c1 = build_water_monomer(offset=np.array([0, 0, 0]))
s2, c2 = build_water_monomer(offset=np.array([2.98, -1.5, 0]), rot=rotmat("z", 180))
write_xyz("water_dimer", s1 + s2, np.vstack([c1, c2]), comment="water dimer, non-covalent H-bonded")
manifest.append(("water_dimer", "F: non-covalent cluster", "6 atoms; hydrogen-bonded pair, approximate geometry"))

s1, c1 = build_water_monomer(offset=np.array([0, 0, 0]))
s2, c2 = build_water_monomer(offset=np.array([2.9, 0, 0]), rot=rotmat("z", 120))
s3, c3 = build_water_monomer(offset=np.array([1.45, 2.5, 0]), rot=rotmat("z", 240))
write_xyz("water_trimer", s1 + s2 + s3, np.vstack([c1, c2, c3]), comment="water trimer, cyclic non-covalent")
manifest.append(("water_trimer", "F: non-covalent cluster", "9 atoms; cyclic H-bonded arrangement, approximate geometry"))

sw, cw = build_water_monomer(offset=np.array([0, 0, 0]))
methane_syms = ["C", "H", "H", "H", "H"]
d = 1.087 / np.sqrt(3)
methane_coords = np.array([[0,0,0],[d,d,d],[d,-d,-d],[-d,d,-d],[-d,-d,d]]) + np.array([4.0, 0, 0])
write_xyz("methane_water", sw + methane_syms, np.vstack([cw, methane_coords]),
          comment="methane-water complex, weak non-covalent")
manifest.append(("methane_water", "F: non-covalent cluster", "8 atoms; weak vdW complex, approximate geometry"))

if HAVE_RDKIT:
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    bsym = [a.GetSymbol() for a in mol.GetAtoms()]
    bc1 = conf.GetPositions()
    bc2 = bc1 + np.array([0, 0, 3.8])
    write_xyz("benzene_dimer", bsym + bsym, np.vstack([bc1, bc2]),
              comment="benzene dimer, pi-stacked non-covalent")
    manifest.append(("benzene_dimer", "F: non-covalent cluster", "24 atoms; pi-stacked, approximate geometry"))


# CATEGORY H: Larger biomolecule-like
if HAVE_RDKIT:
    from_rdkit("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "H: biomolecule", "chiral, polar")
    from_rdkit("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "H: biomolecule", "multi-ring, polar")
    from_rdkit("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "H: biomolecule", "polar")


# CATEGORY I: Growing peptide chain (2 to ~150 atoms)
if HAVE_RDKIT:
    from_rdkit("glycine", "NCC(=O)O", "I: peptide chain", "single amino acid")
    from_rdkit("alanine", "NC(C)C(=O)O", "I: peptide chain", "single amino acid")
    from_rdkit("gly_gly", "NCC(=O)NCC(=O)O", "I: peptide chain", "dipeptide")
    from_rdkit("gly_ala_gly", "NCC(=O)NC(C)C(=O)NCC(=O)O", "I: peptide chain", "tripeptide")

    def poly_ala_smiles(n):
        return "N" + "C(C)C(=O)N" * (n - 1) + "C(C)C(=O)O"

    from_rdkit("poly_ala_10", poly_ala_smiles(10), "I: peptide chain", "10-residue chain, approaching real peptide scale")
    from_rdkit("poly_ala_15", poly_ala_smiles(15), "I: peptide chain", "15-residue chain, ~150 atoms")


with open(os.path.join(OUT_DIR, "MANIFEST.md"), "w") as f:
    f.write("# Test Molecule Suite Manifest\n\n")
    f.write(f"Total molecules: {len(manifest)}\n\n")
    f.write("| Name | Category | Note |\n|---|---|---|\n")
    for name, cat, note in manifest:
        f.write(f"| {name} | {cat} | {note} |\n")

print(f"\nGenerated {len(manifest)} molecules in {OUT_DIR}/")
print(f"Manifest written to {OUT_DIR}/MANIFEST.md")
