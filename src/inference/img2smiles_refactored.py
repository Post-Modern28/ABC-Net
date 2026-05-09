"""
Refactored version of img2smiles.py using postprocessing2.py functions.
Cleaner code organization with modular functions.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.unet import UNet
from ..utils.utils import MolecularImageDataset, collate_fn

plt.switch_backend('agg')
from rdkit import Chem
from pathlib import Path

from .plotting_utils import plot_inference_results
from .postprocessing import predict_smiles_full

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =======================
# CONFIG
# =======================


names = ['test_chembl', 'uob', 'uspto']
DATASET_NAME = names[0]


current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # inference -> src -> ABC-Net
weights_dir = current_file.parent.parent / 'weights'  # inference -> src -> weights
results_dir = current_file.parent.parent / 'results' # inference -> src -> results

CSV_PATH = project_root / 'csv_with_path' / f'{DATASET_NAME}.csv'
WEIGHTS_PATH = weights_dir / 'unet_model_weights29.pkl'
BATCH_SIZE = 32
PLOT_IMAGES = True  # Set to True to save visualization images
PLOT_DPI = 1000
PLOT_DIR = results_dir / DATASET_NAME / 'imgs'
RESULTS_DIR = results_dir / DATASET_NAME

# =======================
# DATA
# =======================

df = pd.read_csv(CSV_PATH).copy().reset_index(drop=True)[:30]

df['atoms_string'] = ''
df['bonds_string'] = ''

dataset = MolecularImageDataset(df, amount=0.1)

dataloader = DataLoader(dataset, BATCH_SIZE, collate_fn=collate_fn)

# =======================
# MODEL
# =======================
model = UNet(in_channels=1, heads=[1,14,3,2,1,360,60,60])
model = nn.DataParallel(model)
model.load_state_dict(torch.load(WEIGHTS_PATH))

model = model.to(device)

torch.cuda.empty_cache()
model.eval()

# =======================
# MAIN INFERENCE
# =======================
total_nums = 0
results = []

with torch.no_grad():
    for batch_num, (imgs, atom_targets, atom_types, atom_charges, atom_hs,
                    bond_targets, bond_types, bond_rhos, bond_omega_types) in enumerate(dataloader):

        imgs = imgs.to(device)
        atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred, bond_targets_pred, \
        bond_types_pred, bond_rhos_pred, bond_omega_types_pred = model(
            imgs)

        # ===== PEAK DETECTION =====
        temp = torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3,
                                              stride=1, padding=1)
        atom_targets_pred = (temp == atom_targets_pred) * (atom_targets_pred > -1).float()

        temp = torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3,
                                              stride=1, padding=1)
        bond_targets_pred = (temp == bond_targets_pred) * (bond_targets_pred > -1).float()

        bond_rhos_pred = torch.abs(bond_rhos_pred)

        bond_types_pred = bond_types_pred.view(-1, 6, 60, 128, 128)

        # ===== BOND OMEGA PREPROCESSING (NMS) =====
        bond_omega_types_pred2 = torch.cat(
            [bond_omega_types_pred[:, 59:], bond_omega_types_pred, bond_omega_types_pred[:, :1]], dim=1
        ).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)

        bond_omega_types_pred2 = ((torch.nn.functional.max_pool1d(bond_omega_types_pred2, stride=1, kernel_size=3,
                padding=0).reshape(-1, 128, 128, 60).permute(0, 3, 1, 2) == bond_omega_types_pred) * \
                (bond_omega_types_pred > -1)).float()

        # ===== PROCESS EACH SAMPLE =====
        for j in range(atom_targets_pred.shape[0]):
            smiles = df.loc[total_nums, 'Smiles']
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, canonical=True)

            total_nums += 1
            if total_nums % 100 == 0:
                print(f"Processed: {total_nums}")

            # ===== EXTRACT PREDICTIONS =====
            img = imgs[j].detach().cpu().numpy()
            atom_target_img = atom_targets_pred[j, 0]
            atom_type_img = atom_types_pred[j].argmax(0)
            atom_charge_img = atom_charges_pred[j].argmax(0)
            atom_hs_img = atom_hs_pred[j].argmax(0)

            bond_target_img = bond_targets_pred[j, 0]
            bond_type_img = bond_types_pred[j].argmax(0)
            bond_rhos_img = bond_rhos_pred[j]
            bond_omega_img = bond_omega_types_pred[j]
            bond_omega_img2 = bond_omega_types_pred2[j]

            # ===== CHECK FOR EMPTY PREDICTIONS =====
            if (atom_target_img.sum()) == 0 or (bond_target_img.sum() == 0):
                print(f"Sample {total_nums}: No atoms/bonds detected")
                results.append(None)
                continue

            # ===== FULL POSTPROCESSING PIPELINE =====
            if PLOT_IMAGES:
                pred_smiles, intermediates = predict_smiles_full(
                    atom_target_img,
                    atom_type_img,
                    atom_charge_img,
                    atom_hs_img,
                    bond_target_img,
                    bond_type_img,
                    bond_rhos_img,
                    bond_omega_img2,
                    return_intermediates=True
                )

                # Plot results if intermediates are available
                if intermediates is not None:
                    plot_inference_results(
                        img=img[0],
                        atom_target_img=atom_target_img,
                        atoms_position_list_final=intermediates['atoms_position_list_final'],
                        atoms_type_list_final=intermediates['atoms_type_list_final'],
                        bond_target_img=bond_target_img,
                        bonds_position_list_final=intermediates['bonds_position_list_final'],
                        bonds_property_list_final=intermediates['bonds_property_list_final'],
                        bonds_delta_list_final=intermediates['bonds_delta_list_final'],
                        save_path=f'{PLOT_DIR}/{total_nums}.png',
                        dpi=PLOT_DPI
                    )
            else:
                pred_smiles = predict_smiles_full(
                    atom_target_img,
                    atom_type_img,
                    atom_charge_img,
                    atom_hs_img,
                    bond_target_img,
                    bond_type_img,
                    bond_rhos_img,
                    bond_omega_img2
                )

            results.append(pred_smiles)

# =======================
# RESULTS
# =======================
df['smiles_pred'] = results

# Calculate exact match accuracy
valid_results = [r for r in results if r is not None]
total_valid = len(valid_results)
exact_matches = sum(1 for gt, pred in zip(df['Smiles'], results)
                    if gt is not None and pred is not None and gt == pred)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total processed: {len(results)}")
print(f"Valid predictions: {total_valid}")
print(f"Exact matches: {exact_matches}")
print(f"Accuracy: {100 * exact_matches / total_valid:.2f}%")
print("="*60 + "\n")

# Save results
os.makedirs(RESULTS_DIR, exist_ok=True)
dff = df[['Smiles', 'smiles_pred']].reset_index(drop=True)
dff['path'] = df['path'].values
dff['match'] = dff['Smiles'] == dff['smiles_pred']
dff.to_csv(f'{RESULTS_DIR}/results_refactored.csv', index=False)
