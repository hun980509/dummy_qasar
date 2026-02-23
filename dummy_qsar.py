"""
Dummy QSAR example for GitHub
- Input: SMILES + target (toy data included)
- Feature: Morgan fingerprint (RDKit)
- Model: RandomForestRegressor (scikit-learn)
- Output: MAE + sample predictions

Install:
  pip install rdkit-pypi scikit-learn numpy
Run:
  python dummy_qsar.py
"""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main() -> None:
    # Toy dataset (SMILES, target). Targets are dummy numbers for demonstration.
    data = [
        ("CCO", 0.12),                 # ethanol
        ("CC(=O)O", 0.25),             # acetic acid
        ("c1ccccc1", 0.55),            # benzene
        ("CCN(CC)CC", 0.40),           # triethylamine
        ("O=C(O)C(O)(CO)CO", 0.30),    # gluconic acid (approx SMILES)
        ("CCOC(=O)C", 0.22),           # ethyl acetate
        ("CC(C)O", 0.15),              # isopropanol
        ("CC(C)C(=O)O", 0.28),         # isobutyric acid
        ("C1CCCCC1", 0.50),            # cyclohexane
        ("COC", 0.10),                 # dimethyl ether
    ]

    smiles = [s for s, _ in data]
    y = np.array([t for _, t in data], dtype=float)

    X = np.vstack([morgan_fp(s) for s in smiles])

    X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
        X, y, smiles, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    print(f"Test MAE: {mae:.4f}\n")
    print("Sample predictions:")
    for s, yt, yp in zip(smi_test, y_test, pred):
        print(f"  {s:20s}  true={yt:.3f}  pred={yp:.3f}")


if __name__ == "__main__":
    main()
