"""
Postprocessing functions for ABC-Net inference.
Refactored from img2smiles_clean.py for better code organization.
"""

from copy import deepcopy

import numpy as np

from generate_smiles import sdf2smiles
from utils import atom_vocab, charge_vocab

# =======================
# DEVOCAB
# =======================
atom_type_devocab = {j: i for i, j in atom_vocab.items()}
atom_type_devocab[0] = 'C'

atom_charge_devocab = {j: i for i, j in charge_vocab.items()}

bond_type_devocab = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

# =======================
# VALENCE TABLE
# =======================
atom_max_valence = {
    '<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1, 'P': 5, 'Br': 1,
    'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3, 'Al': 3, 'Zn': 2,
    'Ca': 2, 'Ag': 1
}

# Simplified valence correction mapping
VALENCE_CORRECTION = {
    2: 'O',
    3: 'N',
    4: 'C',
    5: 'P',
    6: 'S',
    7: 'Cl'
}


# =======================
# UTILS
# =======================
def leaky_relu(x):
    """Leaky ReLU activation function."""
    return np.maximum(x, 0.5 * x)


def correct_atom_valence(atom_type, atom_count):
    """
    Correct atom type based on valence count.
    
    Args:
        atom_type: Current atom type
        atom_count: Number of bonds connected to atom
    
    Returns:
        Corrected atom type
    """
    if atom_type in atom_max_valence and atom_max_valence[atom_type] < atom_count:
        return VALENCE_CORRECTION.get(atom_count, atom_type)
    return atom_type


# =======================
# 1. EXTRACT ATOMS
# =======================
def extract_atoms(atom_target_img, atom_type_img, atom_charge_img, atom_hs_img):
    """
    Extract atoms from predicted heatmaps.
    
    Args:
        atom_target_img: Atom presence heatmap [1, H, W]
        atom_type_img: Atom type predictions [14, H, W]
        atom_charge_img: Atom charge predictions [3, H, W]
        atom_hs_img: Atom H-count predictions [2, H, W]
    
    Returns:
        atoms_pos: List of [x, y] positions
        atoms_type: List of atom types
        atoms_charge: List of atom charges
        atoms_h: List of H-counts
    """
    atoms_pos = []
    atoms_type = []
    atoms_charge = []
    atoms_h = []

    for position in atom_target_img.nonzero(as_tuple=False):
        x, y = position[0].item(), position[1].item()

        # Remove duplicates (distance < 2px)
        if len(atoms_pos) > 0:
            temp1 = np.array(atoms_pos)
            temp2 = np.array([[x, y]])
            if np.sum(np.square(temp1 - temp2), axis=-1).min() < 4:
                continue

        atoms_pos.append([x, y])

        # Extract properties at position (direct indexing, not argmax)
        t = atom_type_img[x, y].item()
        c = atom_charge_img[x, y].item()
        h = atom_hs_img[x, y].item()

        atoms_type.append(atom_type_devocab.get(t, 'C'))
        atoms_charge.append(atom_charge_devocab.get(c, 0))
        atoms_h.append(h)

    return atoms_pos, atoms_type, atoms_charge, atoms_h


# =======================
# 2. EXTRACT BONDS
# =======================
def extract_bonds(bond_target_img, bond_type_img, bond_rhos_img, bond_omega_img):
    """
    Extract bonds from predicted heatmaps.
    
    Args:
        bond_target_img: Bond presence heatmap [1, H, W]
        bond_type_img: Bond type predictions [6, 60, H, W]
        bond_rhos_img: Bond length predictions [60, H, W]
        bond_omega_img: Bond omega predictions [60, H, W]
    
    Returns:
        bonds_pos: List of [x, y] positions
        bonds_type: List of bond types
        bonds_delta: List of [dx, dy] vectors
    """
    bonds_pos = []
    bonds_type = []
    bonds_delta = []

    for position in bond_target_img.nonzero(as_tuple=False):
        x, y = position[0].item(), position[1].item()

        # --- MULTI-OMEGA PEAKS ---
        for omega_idx in bond_omega_img[:, x, y].nonzero(as_tuple=False):
            omega_idx = omega_idx.item()

            # Complex NMS logic
            if omega_idx <= 28:
                if bond_omega_img[omega_idx, x, y] < bond_omega_img[(omega_idx+29):(omega_idx+31), x, y].max():
                    continue
            elif omega_idx == 29:
                if bond_omega_img[omega_idx, x, y] < bond_omega_img[(omega_idx+29):(omega_idx+30), x, y].max() or \
                        bond_omega_img[omega_idx, x, y] < bond_omega_img[0, x, y]:
                    continue
            elif omega_idx == 30:
                if bond_omega_img[omega_idx, x, y] <= bond_omega_img[(omega_idx-30):(omega_idx-29), x, y].max() or \
                    bond_omega_img[omega_idx, x, y] <= bond_omega_img[59, x, y]:
                    continue
            elif omega_idx >= 31:
                if bond_omega_img[omega_idx, x, y] <= bond_omega_img[(omega_idx-31):(omega_idx-29), x, y].max():
                    continue

            omega = omega_idx * (np.pi / 30) + np.pi / 60 - np.pi / 2
            rho = bond_rhos_img[omega_idx, x, y].item()

            dx = rho * np.cos(omega)
            dy = rho * np.sin(omega)

            btype = bond_type_img[omega_idx, x, y].item()

            bonds_pos.append([x, y])
            bonds_type.append(btype)
            bonds_delta.append([dx, dy])

    return bonds_pos, bonds_type, bonds_delta


# =======================
# 3. MATCH BONDS → ATOMS
# =======================
def match_bonds_to_atoms(atoms_pos, bonds_pos, bonds_delta, bonds_type):
    """
    Match bonds to atoms using sophisticated distance calculation.
    
    Args:
        atoms_pos: List of atom positions [[x, y], ...]
        bonds_pos: List of bond center positions [[x, y], ...]
        bonds_delta: List of bond direction vectors [[dx, dy], ...]
        bonds_type: List of bond types
    
    Returns:
        edges: List of [atom_idx1, atom_idx2]
        edge_types: List of bond types
        bonds_pos_filtered: Filtered bond positions (matching edges)
        bonds_delta_filtered: Filtered bond direction vectors (matching edges)
    """
    atoms = np.array(atoms_pos)
    edges = []
    edge_types = []
    bonds_pos_filtered = []
    bonds_delta_filtered = []

    # Calculate unit vectors for bond direction
    bonds_delta_arr = np.array(bonds_delta)
    bond_lengths = np.sqrt((bonds_delta_arr ** 2).sum(-1, keepdims=True))
    e1 = bonds_delta_arr / bond_lengths
    e2 = np.flip(e1.copy(), 1)
    e2[:, 0] = -e2[:, 0]

    e1 = np.expand_dims(e1, 1)
    e2 = np.expand_dims(e2, 1)

    # Predicted atom positions from bonds
    atom_pred_position1 = np.expand_dims(np.array(bonds_pos) + np.array(bonds_delta), 1)
    atom_pred_position2 = np.expand_dims(np.array(bonds_pos) - np.array(bonds_delta), 1)
    atoms_position = np.expand_dims(np.array(atoms_pos), 0)

    # Sophisticated distance calculation with leaky_relu (broadcasted)
    distance1 = np.abs(leaky_relu(((atom_pred_position1 - atoms_position) * e1).sum(-1))) + \
                np.abs((2 * (atom_pred_position1 - atoms_position) * e2).sum(-1))
    distance2 = np.abs(leaky_relu(-((atom_pred_position2 - atoms_position) * e1).sum(-1))) + \
                np.abs((2 * (atom_pred_position2 - atoms_position) * e2).sum(-1))

    atom_index1 = distance2.argmin(-1)
    atom_index2 = distance1.argmin(-1)

    for i in range(len(bonds_pos)):
        index1 = atom_index1[i]
        index2 = atom_index2[i]

        if index1 == index2:
            continue

        # Check for duplicate bonds
        if [index1, index2] in edges or [index2, index1] in edges:
            continue

        edges.append([index1, index2])
        edge_types.append(bond_type_devocab[bonds_type[i]])
        bonds_pos_filtered.append(bonds_pos[i])
        bonds_delta_filtered.append(bonds_delta[i])

    return edges, edge_types, bonds_pos_filtered, bonds_delta_filtered


# =======================
# 4. VALENCE CORRECTION
# =======================
def apply_valence_correction(atoms_type, atoms_charge, edges, edge_types):
    """
    Apply valence correction to atoms based on bond counts.
    
    Args:
        atoms_type: List of atom types
        atoms_charge: List of atom charges
        edges: List of [atom_idx1, atom_idx2]
        edge_types: List of bond types
    
    Returns:
        atoms_type: Corrected atom types
    """
    # Calculate valence for each atom (starting from negative charge)
    atom_counts = [-c for c in atoms_charge]

    for temp, atom_index in enumerate(edges):
        x, y = atom_index
        bond_nums = edge_types[temp]

        # Aromatic bonds (4, 5, 6) count as single bonds
        if bond_nums == 4 or bond_nums == 5 or bond_nums == 6:
            bond_nums = 1

        atom_counts[x] += bond_nums
        atom_counts[y] += bond_nums

    # Apply correction
    for serial, atom_count in enumerate(atom_counts):
        atom_type = atoms_type[serial]
        if atom_type in atom_max_valence and atom_max_valence[atom_type] < atom_count:
            atoms_type[serial] = correct_atom_valence(atom_type, atom_count)

    return atoms_type


# =======================
# 5. FILTER ATOMS
# =======================
def filter_atoms(atoms_pos, atoms_type, atoms_charge, atoms_h, edges):
    """
    Filter atoms to only those connected by bonds.
    
    Args:
        atoms_pos: List of atom positions
        atoms_type: List of atom types
        atoms_charge: List of atom charges
        atoms_h: List of H-counts
        edges: List of [atom_idx1, atom_idx2]
    
    Returns:
        atoms_pos_final: Filtered atom positions
        atoms_type_final: Filtered atom types
        atoms_charge_final: Filtered atom charges
        atoms_h_final: Filtered H-counts
        atom_mapping: Mapping from old to new indices
    """
    # Find atoms that are connected by bonds
    atom_showed_list = []
    for atom_index in edges:
        atom_showed_list += atom_index

    # Create mask for atoms that are connected
    atom_mask = []
    for i in range(len(atoms_pos)):
        if i in atom_showed_list:
            atom_mask.append(True)
        else:
            atom_mask.append(False)

    # Create mapping (1-based indexing for sdf2smiles)
    corresponding_index = []
    new_pos = []
    new_type = []
    new_charge = []
    new_h = []

    k = 1
    for i in range(len(atoms_pos)):
        if atom_mask[i]:
            corresponding_index.append(k)
            new_pos.append(atoms_pos[i])
            new_type.append(atoms_type[i])
            new_charge.append(atoms_charge[i])
            new_h.append(atoms_h[i])
            k += 1
        else:
            corresponding_index.append(k)

    return new_pos, new_type, new_charge, new_h, corresponding_index


# =======================
# 6. REMAP BONDS
# =======================
def remap_bonds(edges, atom_mapping):
    """
    Remap bond indices after atom filtering.
    
    Args:
        edges: List of [atom_idx1, atom_idx2]
        atom_mapping: Mapping from old to new indices (1-based)
    
    Returns:
        edges_final: Remapped edges
    """
    edges_final = []
    for atom_index in edges:
        x, y = atom_index
        x = atom_mapping[x]
        y = atom_mapping[y]
        edges_final.append([x, y])

    return edges_final


# =======================
# 7. IMPLICIT HYDROGEN HANDLING
# =======================
def find_implicit_hydrogens(atoms_type, edges, edge_types, atoms_h):
    """
    Find atoms that need implicit hydrogens.
    
    Args:
        atoms_type: List of atom types
        edges: List of [atom_idx1, atom_idx2]
        edge_types: List of bond types
        atoms_h: List of H-counts
    
    Returns:
        implicit_h_list: List of atom indices (1-based) needing implicit H
    """
    implicit_h_list = []

    for temp, atom_index in enumerate(edges):
        x, y = atom_index
        bond_nums = edge_types[temp]

        # Aromatic bonds (type 4) don't count for valence
        if bond_nums == 4:
            if atoms_type[x - 1] != 'C':
                if atoms_h[x - 1] != 0:
                    if x not in implicit_h_list:
                        implicit_h_list.append(x)
            if atoms_type[y - 1] != 'C':
                if atoms_h[y - 1] != 0:
                    if y not in implicit_h_list:
                        implicit_h_list.append(y)

    return implicit_h_list


# =======================
# 8. MAIN PIPELINE
# =======================
def predict_smiles_full(
    atom_peaks,
    atom_types,
    atom_charges,
    atom_hs,
    bond_peaks,
    bond_types,
    bond_rhos,
    bond_omega,
    return_intermediates=False
):
    """
    Full SMILES prediction pipeline.
    
    Args:
        atom_peaks: Atom presence heatmap [1, H, W]
        atom_types: Atom type predictions [14, H, W]
        atom_charges: Atom charge predictions [3, H, W]
        atom_hs: Atom H-count predictions [2, H, W]
        bond_peaks: Bond presence heatmap [1, H, W]
        bond_types: Bond type predictions [6, 60, H, W]
        bond_rhos: Bond length predictions [60, H, W]
        bond_omega: Bond omega predictions [60, H, W]
        return_intermediates: If True, return intermediate results for plotting
    
    Returns:
        smiles: Predicted SMILES string or None
        intermediates: Dictionary containing intermediate results (only if return_intermediates=True)
    """
    # Extract atoms
    atoms_pos, atoms_type, atoms_charge, atoms_h = extract_atoms(
        atom_peaks, atom_types, atom_charges, atom_hs
    )

    if len(atoms_pos) == 0:
        if return_intermediates:
            return None, None
        return None

    # Extract bonds
    bonds_pos, bonds_type, bonds_delta = extract_bonds(
        bond_peaks, bond_types, bond_rhos, bond_omega
    )

    if len(bonds_pos) == 0:
        if return_intermediates:
            return None, None
        return None

    # Match bonds to atoms
    edges, edge_types, bonds_pos_filtered, bonds_delta_filtered = match_bonds_to_atoms(
        atoms_pos, bonds_pos, bonds_delta, bonds_type
    )

    if len(edges) == 0:
        if return_intermediates:
            return None, None
        return None

    # Apply valence correction (pass atoms_charge)
    atoms_type = apply_valence_correction(atoms_type, atoms_charge, edges, edge_types)

    # Filter atoms (keep only those connected by bonds)
    atoms_pos, atoms_type, atoms_charge, atoms_h, atom_mapping = filter_atoms(
        atoms_pos, atoms_type, atoms_charge, atoms_h, edges
    )

    # Remap bonds
    edges = remap_bonds(edges, atom_mapping)

    # Find implicit hydrogens
    implicit_h_list = find_implicit_hydrogens(atoms_type, edges, edge_types, atoms_h)

    # Generate SMILES
    try:
        smiles = sdf2smiles(
            atoms_type,
            edges,
            atoms_charge,
            edge_types,
            deepcopy(atoms_pos),
            implicit_h_list
        )
    except:
        if return_intermediates:
            return None, None
        return None

    # Return intermediate results if requested
    if return_intermediates:
        intermediates = {
            'atoms_position_list_final': atoms_pos,
            'atoms_type_list_final': atoms_type,
            'bonds_position_list_final': bonds_pos_filtered,
            'bonds_property_list_final': edge_types,
            'bonds_delta_list_final': bonds_delta_filtered
        }
        return smiles, intermediates

    return smiles
