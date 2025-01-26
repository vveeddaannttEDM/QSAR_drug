from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def morgan_fingerprint(smiles, radius=2, n_bits=512):
    """Generate Morgan fingerprints using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

# Placeholder for ImageMol (requires Docker setup; use a pretrained model in practice)
def imagemol_embedding(smiles):
    """Placeholder for ImageMol embeddings."""
    return np.random.rand(512)  # Simulate 512-dim embedding
