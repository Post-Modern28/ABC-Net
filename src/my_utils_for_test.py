import numpy as np
import torch
from rdkit import Chem

from generate_smiles import sdf2smiles
from utils import atom_vocab


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)



def extract_graph(atom_peaks, atom_types_pred,
                  bond_peaks, bond_types_pred,
                  bond_rhos_pred, bond_omega_pred):

    atom_positions = atom_peaks.nonzero(as_tuple=False)
    atoms = []

    for pos in atom_positions:
        x, y = pos[1].item(), pos[2].item()
        atom_type = atom_types_pred[:, x, y].argmax().item()
        atoms.append((x, y, atom_type))

    bonds = []

    for pos in bond_peaks.nonzero(as_tuple=False):
        x, y = pos[1].item(), pos[2].item()

        omega_idx = bond_omega_pred[:, x, y].argmax().item()
        rho = bond_rhos_pred[omega_idx, x, y].item()

        angle = omega_idx * (np.pi / 30) - np.pi/2

        dx = rho * np.cos(angle)
        dy = rho * np.sin(angle)

        bond_type = bond_types_pred[:, omega_idx, x, y].argmax().item()

        bonds.append((x, y, dx, dy, bond_type))

    return atoms, bonds


def match_bonds_to_atoms(atoms, bonds):

    atom_coords = np.array([[a[0], a[1]] for a in atoms])
    edges = []

    for (x, y, dx, dy, bond_type) in bonds:

        p1 = np.array([x + dx, y + dy])
        p2 = np.array([x - dx, y - dy])

        d1 = ((atom_coords - p1)**2).sum(axis=1)
        d2 = ((atom_coords - p2)**2).sum(axis=1)

        i = d1.argmin()
        j = d2.argmin()

        if i != j:
            edges.append((i, j, bond_type))

    return edges



atom_type_devocab = {v: k for k, v in atom_vocab.items()}

def predict_smiles(atoms, edges):
    atom_types = [a[2] for a in atoms]
    atom_labels = [atom_type_devocab.get(t, 'C') for t in atom_types]

    try:
        smiles = sdf2smiles(
            atom_labels,
            [(e[0]+1, e[1]+1) for e in edges],
            [0]*len(atom_labels),
            [e[2]+1 for e in edges],
            [[a[0], a[1]] for a in atoms],
            []
        )
    except:
        return None

    return smiles



import cv2


class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, df, mode="train", amount=0.0):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_path):
        # ⚠️ ВАЖНО: оставил cv2 (обычно совпадает с train)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["path"]
        if not img_path.startswith("/"):
            img_path = "../" + img_path

        img = self._load_image(img_path)

        smiles = row["Smiles"] if "Smiles" in row else None

        # ===== INFERENCE MODE =====
        if self.mode == "infer":
            return img, smiles, img_path

        # ===== TRAIN MODE =====
        return (
            img,
            row["atoms_string"],
            row["atom_types"],
            row["atom_charges"],
            row["atom_hs"],
            row["bond_targets"],
            row["bond_types"],
            row["bond_rhos"],
            row["bond_omega_types"]
        )
