import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.unet import UNet
from ..utils.utils import MolecularImageDataset, atom_vocab, charge_vocab

# --------------------- Настройки ---------------------
os.makedirs('results', exist_ok=True)
plt.switch_backend('agg')
from .generate_smiles import sdf2smiles


def leaky_relu(x):
    return np.maximum(x, 0.5 * x)

# Словари
atom_type_devocab = {j: i for i, j in atom_vocab.items()}
atom_type_devocab[0] = 'C'
atom_charge_devocab = {j: i for i, j in charge_vocab.items()}
bond_type_devocab = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

atom_max_valence = {'<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1, 'P': 5, 'Br': 1,
                    'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3, 'Al': 3, 'Zn': 2, 'Ca': 2, 'Ag': 1}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------- СПЕЦИАЛЬНЫЙ ДАТАСЕТ ДЛЯ ИНФЕРЕНСА (без аугментаций) ---------------------
class InferenceDataset(MolecularImageDataset):
    def __init__(self, df):
        super().__init__(df, amount=0.0)  # убираем noise

def inference_collate_fn(batch):
    """Простая collate функция для инференса"""
    imgs = np.concatenate([np.expand_dims(img, 0) for img in batch], axis=0)
    return torch.from_numpy(imgs)

# --------------------- Загрузка модели ---------------------
print("Loading data...")
df = pd.read_csv('../../train_data/processed_chembl.csv')
df = df[:500].copy().reset_index(drop=True)

dataset = InferenceDataset(df)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=inference_collate_fn, num_workers=0)

print("Loading model...")
model = UNet(in_channels=1, heads=[1, 14, 3, 2, 1, 360, 60, 60])
model = nn.DataParallel(model)

checkpoint_path = 'weights/unet_model_weights29.pkl'
if not os.path.exists(checkpoint_path):
    checkpoint_path = 'weights0.2/unet_model_weights29.pkl'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# --------------------- Инференс ---------------------
torch.cuda.empty_cache()
analysis_data = []
total_nums = 0

print("Starting inference...")
with torch.no_grad():
    for batch_num, imgs in enumerate(dataloader):
        imgs = imgs.to(device)

        # Предсказание
        atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred, \
        bond_targets_pred, bond_types_pred, bond_rhos_pred, bond_omega_types_pred = model(imgs)

        # Постобработка
        atom_targets_pred = torch.sigmoid(atom_targets_pred)
        temp = F.max_pool2d(atom_targets_pred, kernel_size=3, stride=1, padding=1)
        atom_targets_pred = ((temp == atom_targets_pred) * (atom_targets_pred > 0.25)).float()

        bond_targets_pred = torch.sigmoid(bond_targets_pred)
        temp = F.max_pool2d(bond_targets_pred, kernel_size=3, stride=1, padding=1)
        bond_targets_pred = ((temp == bond_targets_pred) * (bond_targets_pred > 0.25)).float()

        bond_rhos_pred = torch.abs(bond_rhos_pred)
        bond_types_pred = torch.softmax(bond_types_pred.view(-1, 6, 60, 128, 128), dim=1)
        bond_omega_types_pred = torch.sigmoid(bond_omega_types_pred)

        for j in range(imgs.shape[0]):
            stats = {
                'img_id': total_nums,
                'true_smiles': df.loc[total_nums, 'Smiles'],
                'pred_smiles': None,
                'status': 'UNKNOWN',
                'num_atoms_detected': 0,
                'num_bonds_detected': 0,
                'max_atom_conf': 0.0,
                'max_bond_conf': 0.0
            }

            current_img = imgs[j, 0].cpu().numpy()
            stats['img_min'] = current_img.min()
            stats['img_max'] = current_img.max()
            stats['img_mean'] = current_img.mean()

            if total_nums == 0:
                plt.imsave('results/sample_input_image.png', current_img, cmap='gray')
                print(f"Sample image: min={current_img.min():.3f}, max={current_img.max():.3f}, mean={current_img.mean():.3f}")

            atom_target_img = atom_targets_pred[j, 0].cpu()
            atom_type_img = atom_types_pred[j].argmax(0).cpu()
            atom_charge_img = atom_charges_pred[j].argmax(0).cpu()
            atom_hs_img = atom_hs_pred[j].argmax(0).cpu()

            bond_target_img = bond_targets_pred[j, 0].cpu()
            bond_type_img = bond_types_pred[j].argmax(0).cpu()
            bond_rhos_img = bond_rhos_pred[j].cpu()
            bond_omega_img = bond_omega_types_pred[j].cpu()

            atom_coords = atom_target_img.nonzero(as_tuple=False)
            stats['num_atoms_detected'] = len(atom_coords)
            if len(atom_coords) > 0:
                stats['max_atom_conf'] = atom_target_img[atom_coords[:, 0], atom_coords[:, 1]].max().item()

            bond_coords = bond_target_img.nonzero(as_tuple=False)
            stats['num_bonds_detected'] = len(bond_coords)
            if len(bond_coords) > 0:
                stats['max_bond_conf'] = bond_target_img[bond_coords[:, 0], bond_coords[:, 1]].max().item()

            if stats['num_atoms_detected'] == 0:
                stats['status'] = 'NO_ATOMS'
            elif stats['num_bonds_detected'] == 0:
                stats['status'] = 'NO_BONDS'
            else:
                try:
                    bonds_position_list = []
                    bonds_property_list = []
                    bonds_delta_list = []

                    for pos in bond_target_img.nonzero(as_tuple=False):
                        x, y = pos[0].item(), pos[1].item()

                        for omega_idx in bond_omega_img[:, x, y].nonzero(as_tuple=False):
                            omega_idx = omega_idx[0].item()

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

                            delta_x = rho * np.cos(omega)
                            delta_y = rho * np.sin(omega)
                            bond_type = bond_type_img[omega_idx, x, y].item()

                            bonds_position_list.append([x, y])
                            bonds_property_list.append(bond_type)
                            bonds_delta_list.append([delta_x, delta_y])

                    atoms_position_list = []
                    atoms_type_list = []
                    atoms_charge_list = []
                    atoms_hs_list = []

                    for pos in atom_target_img.nonzero(as_tuple=False):
                        x, y = pos[0].item(), pos[1].item()
                        atom_type = atom_type_img[x, y].item()
                        atom_charge = atom_charge_img[x, y].item()
                        atom_h = atom_hs_img[x, y].item()

                        if len(atoms_position_list) > 0:
                            temp1 = np.array(atoms_position_list)
                            temp2 = np.array([[x, y]])
                            if np.sum(np.square(temp1 - temp2), axis=-1).min() < 4:
                                continue

                        atoms_position_list.append([x, y])
                        atoms_type_list.append(atom_type_devocab.get(atom_type, 'C'))
                        atoms_charge_list.append(atom_charge_devocab.get(atom_charge, 0))
                        atoms_hs_list.append(atom_h)

                    if len(atoms_position_list) == 0 or len(bonds_position_list) == 0:
                        stats['status'] = 'EMPTY_AFTER_FILTER'
                    else:
                        atom_pred_pos1 = np.expand_dims(np.array(bonds_position_list) + np.array(bonds_delta_list), 1)
                        atom_pred_pos2 = np.expand_dims(np.array(bonds_position_list) - np.array(bonds_delta_list), 1)
                        atoms_pos = np.expand_dims(np.array(atoms_position_list), 0)

                        e1 = np.array(bonds_delta_list) / (np.sqrt((np.array(bonds_delta_list) ** 2).sum(-1, keepdims=True)) + 1e-6)
                        e2 = np.flip(e1.copy(), 1)
                        e2[:, 0] = -e2[:, 0]
                        e1 = np.expand_dims(e1, 1)
                        e2 = np.expand_dims(e2, 1)

                        dist1 = np.abs(leaky_relu(((atom_pred_pos1 - atoms_pos) * e1).sum(-1))) + \
                                np.abs((2 * (atom_pred_pos1 - atoms_pos) * e2).sum(-1))
                        dist2 = np.abs(leaky_relu(-((atom_pred_pos2 - atoms_pos) * e1).sum(-1))) + \
                                np.abs((2 * (atom_pred_pos2 - atoms_pos) * e2).sum(-1))

                        atom_idx1 = dist2.argmin(-1)
                        atom_idx2 = dist1.argmin(-1)

                        bond2atom = []
                        bonds_prop_final = []

                        for i in range(len(bonds_position_list)):
                            idx1 = atom_idx1[i]
                            idx2 = atom_idx2[i]

                            if idx1 == idx2:
                                continue
                            if [idx1, idx2] in bond2atom or [idx2, idx1] in bond2atom:
                                continue

                            bond2atom.append([idx1, idx2])
                            bonds_prop_final.append(bond_type_devocab.get(bonds_property_list[i], 1))

                        if len(bond2atom) == 0:
                            stats['status'] = 'NO_VALID_BONDS'
                        else:
                            atom_counts = [-c for c in atoms_charge_list]
                            for temp_idx, (x, y) in enumerate(bond2atom):
                                bond_nums = bonds_prop_final[temp_idx]
                                if bond_nums >= 4:
                                    bond_nums = 1
                                atom_counts[x] += bond_nums
                                atom_counts[y] += bond_nums

                            for serial, atom_count in enumerate(atom_counts):
                                atom_type = atoms_type_list[serial]
                                if atom_max_valence.get(atom_type, 4) < atom_count:
                                    if atom_count == 2: atoms_type_list[serial] = 'O'
                                    elif atom_count == 3: atoms_type_list[serial] = 'N'
                                    elif atom_count == 4: atoms_type_list[serial] = 'C'
                                    elif atom_count == 5: atoms_type_list[serial] = 'P'
                                    elif atom_count == 6: atoms_type_list[serial] = 'S'
                                    elif atom_count == 7: atoms_type_list[serial] = 'Cl'

                            atom_showed = []
                            for idx_pair in bond2atom:
                                atom_showed.extend(idx_pair)

                            atom_mask = [i in atom_showed for i in range(len(atoms_position_list))]

                            corr_idx = []
                            atoms_type_final = []
                            atoms_charge_final = []
                            atoms_pos_final = []
                            atoms_hs_final = []

                            k = 1
                            for i in range(len(atoms_position_list)):
                                if atom_mask[i]:
                                    corr_idx.append(k)
                                    atoms_type_final.append(atoms_type_list[i])
                                    atoms_charge_final.append(atoms_charge_list[i])
                                    atoms_pos_final.append(atoms_position_list[i])
                                    atoms_hs_final.append(atoms_hs_list[i])
                                    k += 1
                                else:
                                    corr_idx.append(k)

                            bond2atom_final = []
                            for x, y in bond2atom:
                                bond2atom_final.append([corr_idx[x], corr_idx[y]])

                            atom_implicit_hs = []
                            for temp_idx, (x, y) in enumerate(bond2atom_final):
                                bond_nums = bonds_prop_final[temp_idx]
                                if bond_nums == 4:
                                    if atoms_type_final[x - 1] != 'C' and atoms_hs_final[x - 1] != 0:
                                        if x not in atom_implicit_hs:
                                            atom_implicit_hs.append(x)
                                    if atoms_type_final[y - 1] != 'C' and atoms_hs_final[y - 1] != 0:
                                        if y not in atom_implicit_hs:
                                            atom_implicit_hs.append(y)

                            smiles_pred = sdf2smiles(
                                atoms_type_final, bond2atom_final, atoms_charge_final,
                                bonds_prop_final, deepcopy(atoms_pos_final), atom_implicit_hs
                            )

                            stats['pred_smiles'] = smiles_pred
                            stats['status'] = 'SUCCESS'

                except Exception as e:
                    stats['status'] = f'ERROR: {str(e)[:50]}'

            analysis_data.append(stats)
            total_nums += 1

            if total_nums % 10 == 0:
                print(f"Processed {total_nums} images...")
                pd.DataFrame(analysis_data).to_csv('results/detailed_analysis.csv', index=False)

# --------------------- Сохранение ---------------------
analysis_df = pd.DataFrame(analysis_data)
analysis_df.to_csv('results/detailed_analysis.csv', index=False)

print("\n" + "="*50)
print("СВОДКА ПО ИНФЕРЕНСУ")
print("="*50)
print(f"Всего обработано: {len(analysis_df)}")
print("\nСтатусы:")
print(analysis_df['status'].value_counts())
print(f"\nСреднее атомов: {analysis_df['num_atoms_detected'].mean():.1f}")
print(f"Среднее связей: {analysis_df['num_bonds_detected'].mean():.1f}")
print(f"Средняя уверенность атомов: {analysis_df['max_atom_conf'].mean():.3f}")
print(f"Средняя уверенность связей: {analysis_df['max_bond_conf'].mean():.3f}")
        # Успешные предсказания
success_df = analysis_df[analysis_df['status'] == 'SUCCESS']
if len(success_df) > 0:
    print(f"\nУспешных предсказаний: {len(success_df)}")
    print("\nПримеры успешных предсказаний:")
    for _, row in success_df.head(5).iterrows():
        true_smiles = row['true_smiles']
        pred_smiles = row['pred_smiles']
        if pred_smiles is not None:
            print(f"  True:  {true_smiles[:50] if true_smiles else 'N/A'}...")
            print(f"  Pred:  {pred_smiles[:50]}...")
        else:
            print(f"  True:  {true_smiles[:50] if true_smiles else 'N/A'}...")
            print("  Pred:  None")
        print()
