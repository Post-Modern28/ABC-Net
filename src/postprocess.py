import numpy as np
from copy import deepcopy
from generate_smiles import sdf2smiles
from utils import atom_vocab, charge_vocab

# =======================
# DEVOCAB
# =======================
atom_type_devocab = {v: k for k, v in atom_vocab.items()}
atom_type_devocab[0] = 'C'

atom_charge_devocab = {v: k for k, v in charge_vocab.items()}

bond_type_devocab = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

# =======================
# VALENCE TABLE
# =======================
atom_max_valence = {
    '<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1, 'P': 5, 'Br': 1,
    'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3, 'Al': 3, 'Zn': 2,
    'Ca': 2, 'Ag': 1
}

# =======================
# UTILS
# =======================
def leaky_relu(x):
    return np.maximum(x, 0.5 * x)


# =======================
# 1. EXTRACT ATOMS
# =======================
def extract_atoms(atom_peaks, atom_types, atom_charges, atom_hs):

    positions = atom_peaks.nonzero(as_tuple=False)
    atoms_pos = []
    atoms_type = []
    atoms_charge = []
    atoms_h = []

    for pos in positions:
        x, y = pos[1].item(), pos[2].item()

        # remove duplicates (distance < 2px)
        if len(atoms_pos) > 0:
            d = np.array(atoms_pos) - np.array([[x, y]])
            if np.min(np.sum(d**2, axis=1)) < 4:
                continue

        atoms_pos.append([x, y])

        t = atom_types[:, x, y].argmax().item()
        c = atom_charges[:, x, y].argmax().item()
        h = atom_hs[:, x, y].argmax().item()

        atoms_type.append(atom_type_devocab.get(t, 'C'))
        atoms_charge.append(atom_charge_devocab.get(c, 0))
        atoms_h.append(h)

    return atoms_pos, atoms_type, atoms_charge, atoms_h


# =======================
# 2. EXTRACT BONDS
# =======================
def extract_bonds(bond_peaks, bond_types, bond_rhos, bond_omega):

    bonds_pos = []
    bonds_type = []
    bonds_delta = []

    for pos in bond_peaks.nonzero(as_tuple=False):
        x, y = pos[1].item(), pos[2].item()

        # --- MULTI-OMEGA PEAKS ---
        for omega_idx in (bond_omega[:, x, y] > 0.25).nonzero(as_tuple=False):

            omega_idx = omega_idx.item()

            # Complex NMS logic from img2smiles2.py
            if omega_idx <= 28:
                if bond_omega[omega_idx, x, y] < bond_omega[(omega_idx+29):(omega_idx+31), x, y].max():
                    continue
            elif omega_idx == 29:
                if bond_omega[omega_idx, x, y] < bond_omega[(omega_idx+29):(omega_idx+30), x, y].max() or \
                        bond_omega[omega_idx, x, y] < bond_omega[0, x, y]:
                    continue
            elif omega_idx == 30:
                if bond_omega[omega_idx, x, y] <= bond_omega[(omega_idx-30):(omega_idx-29), x, y].max() or \
                    bond_omega[omega_idx, x, y] <= bond_omega[59, x, y]:
                    continue
            elif omega_idx >= 31:
                if bond_omega[omega_idx, x, y] <= bond_omega[(omega_idx-31):(omega_idx-29), x, y].max():
                    continue

            omega = omega_idx * (np.pi / 30) + np.pi / 60 - np.pi / 2
            rho = bond_rhos[omega_idx, x, y].item()

            dx = rho * np.cos(omega)
            dy = rho * np.sin(omega)

            btype = bond_types[:, omega_idx, x, y].argmax().item()

            bonds_pos.append([x, y])
            bonds_type.append(btype)
            bonds_delta.append([dx, dy])

    return bonds_pos, bonds_type, bonds_delta


# =======================
# 3. MATCH BONDS → ATOMS
# =======================
def match_bonds_to_atoms(atoms_pos, bonds_pos, bonds_delta, bonds_type):

    atoms = np.array(atoms_pos)
    edges = []
    edge_types = []

    # Calculate unit vectors for bond direction
    bonds_delta_arr = np.array(bonds_delta)
    bond_lengths = np.sqrt((bonds_delta_arr ** 2).sum(-1, keepdims=True))
    e1 = bonds_delta_arr / bond_lengths
    e2 = np.flip(e1.copy(), 1)
    e2[:, 0] = -e2[:, 0]

    for i in range(len(bonds_pos)):
        x, y = bonds_pos[i]
        dx, dy = bonds_delta[i]

        p1 = np.array([x + dx, y + dy])
        p2 = np.array([x - dx, y - dy])

        # Sophisticated distance calculation with leaky_relu from img2smiles2.py
        distance1 = np.abs(leaky_relu(((p1 - atoms) * e1[i]).sum(-1))) + \
                    np.abs((2 * (p1 - atoms) * e2[i]).sum(-1))
        distance2 = np.abs(leaky_relu(-((p2 - atoms) * e1[i]).sum(-1))) + \
                    np.abs((2 * (p2 - atoms) * e2[i]).sum(-1))

        i1 = distance2.argmin()
        i2 = distance1.argmin()

        if i1 == i2:
            continue

        edge = sorted([i1, i2])

        if edge in edges:
            continue

        edges.append(edge)
        edge_types.append(bond_type_devocab[bonds_type[i]])

    return edges, edge_types


# =======================
# 4. VALENCE CORRECTION
# =======================
def fix_valence(atoms_type, edges, edge_types, atoms_charge):

    valence = [-c for c in atoms_charge]

    for (i, j), t in zip(edges, edge_types):
        v = t if t < 4 else 1
        valence[i] += v
        valence[j] += v

    for i in range(len(atoms_type)):
        atom = atoms_type[i]
        if atom not in atom_max_valence:
            continue

        if valence[i] > atom_max_valence[atom]:
            # heuristic correction based on actual valence count
            if valence[i] == 2:
                atoms_type[i] = 'O'
            elif valence[i] == 3:
                atoms_type[i] = 'N'
            elif valence[i] == 4:
                atoms_type[i] = 'C'
            elif valence[i] == 5:
                atoms_type[i] = 'P'
            elif valence[i] == 6:
                atoms_type[i] = 'S'
            elif valence[i] == 7:
                atoms_type[i] = 'Cl'

    return atoms_type


# =======================
# 5. REMOVE UNUSED ATOMS
# =======================
def filter_atoms(atoms_pos, atoms_type, atoms_charge, atoms_h, edges):

    used = set()
    for i, j in edges:
        used.add(i)
        used.add(j)

    mapping = {}
    new_pos = []
    new_type = []
    new_charge = []
    new_h = []

    idx = 1
    for i in range(len(atoms_pos)):
        if i in used:
            mapping[i] = idx
            new_pos.append(atoms_pos[i])
            new_type.append(atoms_type[i])
            new_charge.append(atoms_charge[i])
            new_h.append(atoms_h[i])
            idx += 1

    new_edges = []
    for i, j in edges:
        if i in mapping and j in mapping:
            new_edges.append([mapping[i], mapping[j]])

    return new_pos, new_type, new_charge, new_h, new_edges


# =======================
# 6. MAIN PIPELINE
# =======================
def predict_smiles_full(
    atom_peaks,
    atom_types,
    atom_charges,
    atom_hs,
    bond_peaks,
    bond_types,
    bond_rhos,
    bond_omega
):

    atoms_pos, atoms_type, atoms_charge, atoms_h = extract_atoms(
        atom_peaks, atom_types, atom_charges, atom_hs
    )

    if len(atoms_pos) == 0:
        return None

    bonds_pos, bonds_type, bonds_delta = extract_bonds(
        bond_peaks, bond_types, bond_rhos, bond_omega
    )

    edges, edge_types = match_bonds_to_atoms(
        atoms_pos, bonds_pos, bonds_delta, bonds_type
    )

    if len(edges) == 0:
        return None

    atoms_type = fix_valence(atoms_type, edges, edge_types, atoms_charge)

    atoms_pos, atoms_type, atoms_charge, atoms_h, edges = filter_atoms(
        atoms_pos, atoms_type, atoms_charge, atoms_h, edges
    )

    try:
        smiles = sdf2smiles(
            atoms_type,
            edges,
            atoms_charge,
            edge_types,
            deepcopy(atoms_pos),
            []
        )
    except:
        return None

    return smiles