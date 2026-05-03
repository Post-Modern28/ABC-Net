from utils_for_test import MolecularImageDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt
import os
from generate_smiles import sdf2smiles
from copy import deepcopy
from utils import atom_vocab, charge_vocab
import rdkit
from rdkit import Chem

plt.switch_backend('agg')

def leaky_relu(x):
    return np.maximum(x, 0.5 * x)

# Словари
atom_type_devocab = {j: i for i, j in atom_vocab.items()}
atom_type_devocab[0] = 'C'
atom_charge_devocab = {j: i for i, j in charge_vocab.items()}
bond_type_devocab = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

atom_max_valence = {'<unkonw>': 4, 'O': 2, 'C': 4, 'N': 3, 'F': 1, 'H': 1, 'S': 6, 'Cl': 1,
                    'P': 5, 'Br': 1, 'B': 3, 'I': 1, 'Si': 4, 'Se': 6, 'Te': 6, 'As': 3,
                    'Al': 3, 'Zn': 2, 'Ca': 2, 'Ag': 1}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'results/viz'
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Загрузка данных
# ---------------------------
df = pd.read_csv('../data2/UOB/uob.csv')

# Приводим колонку со SMILES к единому имени
if 'Smiles' in df.columns:
    df.rename(columns={'Smiles': 'smiles'}, inplace=True)
elif 'SMILES' in df.columns:
    df.rename(columns={'SMILES': 'smiles'}, inplace=True)

dataset = MolecularImageDataset(df)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)

# ---------------------------
# Модель
# ---------------------------
model = UNet(in_channels=1, heads=[1, 14, 3, 2, 1, 360, 60, 60])
model = nn.DataParallel(model)
model.load_state_dict(torch.load('weights/unet_model_weights29.pkl', map_location=device))
model = model.to(device)
model.eval()

torch.cuda.empty_cache()

total_nums = 0
results = []

with torch.no_grad():
    for batch_num, imgs in enumerate(dataloader):
        imgs = imgs.to(device)

        outputs = model(imgs)
        (atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred,
         bond_targets_pred, bond_types_pred, bond_rhos_pred, bond_omega_types_pred) = outputs

        # Non-Maximum Suppression
        temp = torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3, stride=1, padding=1)
        atom_targets_pred = ((temp == atom_targets_pred) & (atom_targets_pred > 0.25)).float()

        temp = torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3, stride=1, padding=1)
        bond_targets_pred = ((temp == bond_targets_pred) & (bond_targets_pred > 0.25)).float()

        bond_rhos_pred = torch.abs(bond_rhos_pred)
        bond_types_pred = bond_types_pred.view(-1, 6, 60, 128, 128)
        bond_omega_types_pred = torch.sigmoid(bond_omega_types_pred)

        # Визуализация для первых 4 изображений в батче
        for j in range(min(4, imgs.shape[0])):
            idx_in_batch = batch_num * dataloader.batch_size + j
            if idx_in_batch >= len(df):
                break

            img = imgs[j, 0].cpu().numpy()
            atom_pred = atom_targets_pred[j, 0].cpu().numpy()
            bond_pred = bond_targets_pred[j, 0].cpu().numpy()
            atom_peak = atom_targets_pred[j, 0].cpu().numpy()  # уже после NMS
            bond_peak = bond_targets_pred[j, 0].cpu().numpy()

            # Получаем имя файла из колонки 'path'
            path = df.loc[idx_in_batch, 'path']
            filename = os.path.basename(path)

            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f"Input: {filename}", fontsize=8)

            axes[1].imshow(atom_pred)
            axes[1].set_title("Atom heatmap")

            axes[2].imshow(atom_peak)
            axes[2].set_title("Atom peaks")

            axes[3].imshow(bond_pred)
            axes[3].set_title("Bond heatmap")

            axes[4].imshow(bond_peak)
            axes[4].set_title("Bond peaks")

            plt.tight_layout()
            safe_filename = filename.replace('/', '_').replace('\\', '_')
            plt.savefig(f"{SAVE_DIR}/batch{batch_num}_img{j}_{safe_filename}.png", dpi=150)
            plt.close()

        # Обработка всех изображений батча для предсказания SMILES
        for j in range(imgs.shape[0]):
            idx_in_batch = batch_num * dataloader.batch_size + j
            if idx_in_batch >= len(df):
                break

            smiles_true = df.loc[idx_in_batch, 'smiles']
            mol = Chem.MolFromSmiles(smiles_true)
            if mol is not None:
                smiles_true = Chem.MolToSmiles(mol, canonical=True)

            total_nums += 1
            if total_nums % 100 == 0:
                print(f"Processed {total_nums} images")

            atom_target_img = atom_targets_pred[j, 0].cpu()
            atom_type_img = atom_types_pred[j].argmax(0).cpu()
            atom_charge_img = atom_charges_pred[j].argmax(0).cpu()
            atom_hs_img = atom_hs_pred[j].argmax(0).cpu()

            bond_target_img = bond_targets_pred[j, 0].cpu()
            bond_type_img = bond_types_pred[j].argmax(0).cpu()
            bond_rhos_img = bond_rhos_pred[j].cpu()
            bond_omega_img = bond_omega_types_pred[j].cpu()

            if atom_target_img.sum() == 0 or bond_target_img.sum() == 0:
                print(f"  Warning: no atoms or bonds for image {idx_in_batch}")
                results.append(None)
                continue

            # ----- Извлечение связей -----
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

            # ----- Извлечение атомов -----
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
                results.append(None)
                continue

            # ----- Сопоставление атомов и связей -----
            atom_pred_pos1 = np.expand_dims(np.array(bonds_position_list) + np.array(bonds_delta_list), 1)
            atom_pred_pos2 = np.expand_dims(np.array(bonds_position_list) - np.array(bonds_delta_list), 1)
            atoms_pos = np.expand_dims(np.array(atoms_position_list), 0)

            e1 = np.array(bonds_delta_list) / (np.sqrt((np.array(bonds_delta_list)**2).sum(-1, keepdims=True)) + 1e-6)
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

            bond2atom_index = []
            bonds_property_final = []
            for i in range(len(bonds_position_list)):
                idx1 = atom_idx1[i]
                idx2 = atom_idx2[i]
                if idx1 == idx2:
                    continue
                if [idx1, idx2] in bond2atom_index or [idx2, idx1] in bond2atom_index:
                    continue
                bond2atom_index.append([idx1, idx2])
                bonds_property_final.append(bond_type_devocab.get(bonds_property_list[i], 1))

            if len(bond2atom_index) == 0:
                results.append(None)
                continue

            # ----- Коррекция валентности -----
            atom_counts = [-c for c in atoms_charge_list]
            for temp_idx, (x, y) in enumerate(bond2atom_index):
                bond_nums = bonds_property_final[temp_idx]
                if bond_nums >= 4:
                    bond_nums = 1
                atom_counts[x] += bond_nums
                atom_counts[y] += bond_nums

            for serial, atom_count in enumerate(atom_counts):
                atom_type = atoms_type_list[serial]
                if atom_max_valence.get(atom_type, 4) < atom_count:
                    if atom_count == 2:
                        atoms_type_list[serial] = 'O'
                    elif atom_count == 3:
                        atoms_type_list[serial] = 'N'
                    elif atom_count == 4:
                        atoms_type_list[serial] = 'C'
                    elif atom_count == 5:
                        atoms_type_list[serial] = 'P'
                    elif atom_count == 6:
                        atoms_type_list[serial] = 'S'
                    elif atom_count == 7:
                        atoms_type_list[serial] = 'Cl'

            # ----- Перенумерация -----
            atom_showed = []
            for idx_pair in bond2atom_index:
                atom_showed.extend(idx_pair)
            atom_mask = [i in atom_showed for i in range(len(atoms_position_list))]

            corresponding_index = []
            atoms_type_final = []
            atoms_charge_final = []
            atoms_pos_final = []
            atoms_hs_final = []

            k = 1
            for i in range(len(atoms_position_list)):
                if atom_mask[i]:
                    corresponding_index.append(k)
                    atoms_type_final.append(atoms_type_list[i])
                    atoms_charge_final.append(atoms_charge_list[i])
                    atoms_pos_final.append(atoms_position_list[i])
                    atoms_hs_final.append(atoms_hs_list[i])
                    k += 1
                else:
                    corresponding_index.append(k)

            bond2atom_final = []
            for x, y in bond2atom_index:
                bond2atom_final.append([corresponding_index[x], corresponding_index[y]])

            # ----- Неявные водороды -----
            atom_implicit_hs = []
            for temp_idx, (x, y) in enumerate(bond2atom_final):
                bond_nums = bonds_property_final[temp_idx]
                if bond_nums == 4:
                    if atoms_type_final[x-1] != 'C' and atoms_hs_final[x-1] != 0:
                        if x not in atom_implicit_hs:
                            atom_implicit_hs.append(x)
                    if atoms_type_final[y-1] != 'C' and atoms_hs_final[y-1] != 0:
                        if y not in atom_implicit_hs:
                            atom_implicit_hs.append(y)

            # ----- Генерация SMILES -----
            try:
                smiles_pred = sdf2smiles(
                    atoms_type_final,
                    bond2atom_final,
                    atoms_charge_final,
                    bonds_property_final,
                    deepcopy(atoms_pos_final),
                    atom_implicit_hs
                )
                results.append(smiles_pred)
            except Exception as e:
                print(f"Error generating SMILES for image {idx_in_batch}: {e}")
                results.append(None)

# Сохранение итогов
df['smiles_pred'] = results
output_df = df[['smiles', 'smiles_pred']]
output_df.to_csv('results/results.csv', index=False)
print("Done! Results saved to results/results.csv")
print(f"Visualizations saved to {SAVE_DIR}")