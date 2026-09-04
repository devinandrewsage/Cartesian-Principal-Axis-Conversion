# Test Molecule Suite Manifest

Total molecules: 50

| Name | Category | Note |
|---|---|---|
| N2 | A: diatomic | nonpolar, homonuclear -- both moments exactly 0/degenerate |
| O2 | A: diatomic | nonpolar, homonuclear |
| CO | A: diatomic | polar, heteronuclear |
| HF | A: diatomic | strongly polar |
| HCl | A: diatomic | polar |
| H2O | A: small polar | 3 atoms; reproduces paper's worked example |
| CO2 | A: linear, EDGE CASE | 3 atoms; EXACT eigenvalue degeneracy + sign-ambiguity (validated unstable) |
| HCN | A: linear polar | 3 atoms; linear -- perpendicular moments exactly degenerate (all linear molecules), but no atom-projection tie (heteronuclear) |
| formaldehyde | B: small polar, EDGE CASE | 4 atoms; C2v symmetry, sign-ambiguity (validated unstable) |
| ammonia | B: small polar, EDGE CASE | 4 atoms; C3v symmetric top -- 2 of 3 moments EXACTLY degenerate |
| methane | B: small nonpolar, EXTREME EDGE CASE | 5 atoms; Td symmetry -- ALL 3 moments exactly equal (spherical top) |
| acetylene | B: linear nonpolar, EDGE CASE | 4 atoms; linear, homonuclear-symmetric -- exact tie risk |
| H2S | B: small polar | 3 atoms; bent, heavier S |
| ozone | B: small bent | 3 atoms; bent, resonance-delocalized |
| ethylene | B: small nonpolar | 6 atoms; planar |
| hydrogen_peroxide | B: small polar | 4 atoms; nonplanar, C2 symmetry |
| formic_acid | B: small polar baseline | 5 atoms; no symmetry -- validated stable under rotation |
| R_bromochlorofluoromethane | C: chiral pair | 5 atoms; simplest chiral molecule, R enantiomer |
| S_bromochlorofluoromethane | C: chiral pair | 5 atoms; simplest chiral molecule, S enantiomer |
| L_alanine | C: chiral pair | 13 atoms; natural amino acid form |
| D_alanine | C: chiral pair | 13 atoms; mirror image of natural form |
| benzene | D: aromatic, EXTREME | 12 atoms; D6h -- 2 of 3 moments exactly degenerate (planar symmetric ring) |
| cyclohexane | D: nonpolar ring | 18 atoms; chair conformation |
| naphthalene | D: aromatic | 18 atoms; fused bicyclic, planar symmetric |
| adamantane | D: cage, EXTREME | 26 atoms; Td cage -- near-spherical-top, highly degenerate |
| cubane | D: cage, EXTREME | 16 atoms; Oh cage -- near-perfect spherical top |
| methanol | E: polar organic | 6 atoms;  |
| ethanol | E: polar organic | 9 atoms;  |
| acetone | E: polar organic | 10 atoms;  |
| acetic_acid | E: polar organic | 8 atoms;  |
| pyridine | E: aromatic polar | 11 atoms;  |
| phenol | E: aromatic polar | 13 atoms;  |
| aniline | E: aromatic polar | 14 atoms;  |
| toluene | E: aromatic nonpolar | 15 atoms;  |
| ammonium | G: ion, EXTREME | 5 atoms; Td, fully degenerate like methane |
| hydronium | G: ion | 4 atoms; pyramidal cation |
| acetate | G: ion | 7 atoms;  |
| water_dimer | F: non-covalent cluster | 6 atoms; hydrogen-bonded pair, approximate geometry |
| water_trimer | F: non-covalent cluster | 9 atoms; cyclic H-bonded arrangement, approximate geometry |
| methane_water | F: non-covalent cluster | 8 atoms; weak vdW complex, approximate geometry |
| benzene_dimer | F: non-covalent cluster | 24 atoms; pi-stacked, approximate geometry |
| glucose | H: biomolecule | 24 atoms; chiral, polar |
| caffeine | H: biomolecule | 24 atoms; multi-ring, polar |
| aspirin | H: biomolecule | 21 atoms; polar |
| glycine | I: peptide chain | 10 atoms; single amino acid |
| alanine | I: peptide chain | 13 atoms; single amino acid |
| gly_gly | I: peptide chain | 17 atoms; dipeptide |
| gly_ala_gly | I: peptide chain | 27 atoms; tripeptide |
| poly_ala_10 | I: peptide chain | 103 atoms; 10-residue chain, approaching real peptide scale |
| poly_ala_15 | I: peptide chain | 153 atoms; 15-residue chain, ~150 atoms |
