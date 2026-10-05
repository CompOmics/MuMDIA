"""The modification catalogue for the library helpers: UniMod PSI-MS name, accession and
monoisotopic delta of every modification MuMDIA knows by name.

A copy of `rust/mumdia/crates/mumdia-core/src/modifications.json`, the table the engine's
mass model and the desktop Search screen read, kept here as Python so the helpers run from
any directory they are copied to (the desktop bundle ships `scripts/*.py` alone).
`tests/python/test_unimod_catalogue.py` fails when the two disagree; regenerate this list
from the JSON rather than editing it by hand.
"""

# (name, UniMod accession, monoisotopic delta), in catalogue order.
MODIFICATIONS = (
    ('Carbamidomethyl', 4, 57.021463735),
    ('Propionamide', 24, 71.037114),
    ('Carboxymethyl', 6, 58.005479),
    ('Nethylmaleimide', 108, 125.047679),
    ('Methylthio', 39, 45.987721),
    ('Oxidation', 35, 15.99491462),
    ('Dioxidation', 425, 31.989829),
    ('Trioxidation', 345, 47.984744),
    ('Kynurenine', 351, 3.994915),
    ('Deamidated', 7, 0.984016106),
    ('Carbamyl', 5, 43.005813726),
    ('Nitro', 354, 44.985078),
    ('Cysteinyl', 312, 119.004099),
    ('Glutathione', 55, 305.068156),
    ('Phospho', 21, 79.96633109),
    ('Sulfo', 40, 79.956815),
    ('Acetyl', 1, 42.010564684),
    ('Formyl', 122, 27.994915),
    ('Propionyl', 58, 56.026215),
    ('Butyryl', 1289, 70.041865),
    ('Crotonyl', 1363, 68.026215),
    ('Malonyl', 747, 86.000394),
    ('Succinyl', 64, 100.016044),
    ('Methyl', 34, 14.015650064),
    ('Dimethyl', 36, 28.031300128),
    ('Trimethyl', 37, 42.04695),
    ('GlyGly', 121, 114.042927),
    ('HexNAc', 43, 203.079373),
    ('Hex', 41, 162.052824),
    ('Palmitoyl', 47, 238.229666),
    ('Farnesyl', 44, 204.1878011),
    ('GeranylGeranyl', 48, 272.2504012),
    ('Hydroxyfarnesyl', 376, 220.1827157),
    ('Biotin', 3, 226.077598),
)

MASS_BY_NAME = {name: mass for name, _, mass in MODIFICATIONS}
NAME_BY_UNIMOD = {unimod: name for name, unimod, _ in MODIFICATIONS}
