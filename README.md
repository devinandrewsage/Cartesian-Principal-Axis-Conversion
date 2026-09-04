# Principal-Axis Conversion Pipeline

Converts molecular Cartesian coordinates into a mass-weighted
principal-axis frame: center of mass at the origin, axes aligned to the
eigenvectors of the moment-of-inertia tensor, with sign and chirality
conventions enforced. 

This produces coordinates that are **invariant to
the arbitrary translation/rotation of the input file** — two
representations of the same physical structure, however differently
oriented, converge to identical output.

This is a preprocessing step intended to feed into orientation-dependent
downstream analysis where the answer to
"which atoms fall in this spatial region" must not depend on how a
structure happened to be written to a file.

## Status

Core pipeline validated against:
- The reference paper's own worked example (water molecule)
- 9 real peptide structures from the paper's own benchmark set
  (Segglu, Segtrp, Tuftsin, Gly12, 1YJP, and three (Ala)18 conformers),
  sizes 64–189 atoms
- A real experimental PDB structure (1UBQ, ubiquitin, 660 heavy atoms
  from X-ray crystallography)
- An independent third-party implementation (ASE) — moment-of-inertia
  eigenvalues agree to machine precision (~1e-16 relative)
- A curated 50-molecule benchmark suite spanning 2–153 atoms, covering
  polar/nonpolar, covalent/non-covalent, chiral pairs, and deliberately
  symmetric edge cases (`generate_test_molecules.py`)
- A 16-check automated regression suite (`verify_setup.py`)

## Installation

```bash
git clone <this repo>
cd principle_conversion_project
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

For cross-validation against ASE, consolidated HDF5 output, and the
test-molecule generator (all optional):
```bash
pip install -e ".[validation,hdf5]"
pip install rdkit   # only needed to regenerate the test-molecule suite
```

## Quick start

Verify the installation before trusting any results:
```bash
python3 verify_setup.py
```
This should print `ALL CHECKS PASSED`. Do not proceed if it does not.

Convert a batch of structures:
```bash
principal-axes-batch <input_dir> <output_dir>
```
or equivalently:
```bash
python3 batch_principal_axes.py <input_dir> <output_dir>
```

## Supported input formats

- `.xyz` — plain XYZ coordinate files
- `.com` / `.gjf` — Gaussian input decks (coordinates extracted; route
  line, charge, multiplicity ignored on read)
- `.pdb` — Protein Data Bank format, with correct element-symbol
  disambiguation (see below). Validated against a real RCSB structure
  (1UBQ).

## Supported output formats

```bash
--output-format xyz   # default, universally readable
--output-format com   # Gaussian input file (placeholder route/charge/
                       # multiplicity -- MUST be reviewed before submitting
                       # a real Gaussian job)
--output-format pdb
```

## Options

