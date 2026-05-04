"""
Инференс на тестовом датасете с корректным подсчётом метрик
(исправлены баги с accuracy > 1 и перезаписью тензоров)
"""

from generate_smiles import sdf2smiles
from rdkit import Chem
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os
from postprocess import predict_smiles_full
from utils import MolecularImageDataset, collate_fn
from unet import UNet
from my_utils_for_test import (
    canonical,
    extract_graph,
    match_bonds_to_atoms,
    predict_smiles
)

# ========= Meter =========
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========= CONFIG =========
CSV_PATH = '../train_data/test_chembl.csv'
WEIGHTS_PATH = 'weights/unet_model_weights29.pkl'
SAVE_DIR = 'debug_outputs/final_test'
BATCH_SIZE = 16
NUM_SAMPLES = 10_000

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= DATA =========
df = pd.read_csv(CSV_PATH)
df = df[:NUM_SAMPLES].reset_index(drop=True)

dataset = MolecularImageDataset(df, amount=0.0)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        collate_fn=collate_fn, shuffle=False)

# ========= MODEL =========
model = UNet(in_channels=1, heads=[1,14,3,2,1,360,60,60])
model = nn.DataParallel(model)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# ========= Метрики =========
atom_precision = AverageMeter()
atom_recall = AverageMeter()
atom_precision3 = AverageMeter()
atom_recall3 = AverageMeter()

atom_types_acc = AverageMeter()
atom_charges_acc = AverageMeter()
atom_hs_acc = AverageMeter()

bond_precision = AverageMeter()
bond_recall = AverageMeter()
bond_precision3 = AverageMeter()
bond_recall3 = AverageMeter()

bond_types_acc = AverageMeter()
bond_rhos_mae = AverageMeter()

exact_match_count = 0
total_processed = 0


with torch.no_grad():
    for batch_idx, (imgs, atom_targets, atom_types, atom_charges, atom_hs,
                    bond_targets, bond_types, bond_rhos, bond_omega_types) in enumerate(dataloader):

        imgs = imgs.to(device)
       

        
        atom_targets = atom_targets.to(device)
        atom_types = atom_types.to(device)
        atom_charges = atom_charges.to(device)
        atom_hs = atom_hs.to(device)

        bond_targets = bond_targets.to(device)
        bond_types = bond_types.to(device)
        bond_rhos = bond_rhos.to(device)

        # ===== forward =====
        outputs = model(imgs)

        (atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred,
         bond_targets_pred, bond_types_pred, bond_rhos_pred, bond_omega_types_pred) = outputs

        # ===== normalization =====
        atom_targets_pred = torch.sigmoid(atom_targets_pred)
        bond_targets_pred = torch.sigmoid(bond_targets_pred)

        atom_types_pred = torch.softmax(atom_types_pred, dim=1)
        atom_charges_pred = torch.softmax(atom_charges_pred, dim=1)
        atom_hs_pred = torch.softmax(atom_hs_pred, dim=1)

        bond_types_pred = torch.softmax(
            bond_types_pred.view(-1, 6, 60, 128, 128), dim=1
        )

        bond_rhos_pred = torch.abs(bond_rhos_pred)

        # Normalize bond_omega with sigmoid
        bond_omega_types_pred = torch.sigmoid(bond_omega_types_pred)

        # ===== peaks =====
        temp = torch.nn.functional.max_pool2d(atom_targets_pred, 3, 1, 1)
        atom_peaks_pred = ((temp == atom_targets_pred) & (atom_targets_pred > 0.25)).float()

        temp = torch.nn.functional.max_pool2d(bond_targets_pred, 3, 1, 1)
        bond_peaks_pred = ((temp == bond_targets_pred) & (bond_targets_pred > 0.25)).float()

        atom_peaks_gt = (atom_targets == 1).float()
        bond_peaks_gt = (bond_targets == 1).float()

        # ===================== АТОМЫ =====================
        tp = (atom_peaks_pred * atom_peaks_gt).sum().item()
        pred_sum = atom_peaks_pred.sum().item()
        gt_sum = atom_peaks_gt.sum().item()

        if pred_sum > 0:
            atom_precision.update(tp / pred_sum, pred_sum)
        if gt_sum > 0:
            atom_recall.update(tp / gt_sum, gt_sum)

        # @3
        atom_gt_pooled = torch.nn.functional.max_pool2d(atom_peaks_gt, 3, 1, 1)
        tp3 = (atom_peaks_pred * atom_gt_pooled).sum().item()
        if pred_sum > 0:
            atom_precision3.update(tp3 / pred_sum, pred_sum)

        atom_pred_pooled = torch.nn.functional.max_pool2d(atom_peaks_pred, 3, 1, 1)
        rec3 = (atom_peaks_gt * atom_pred_pooled).sum().item()
        if gt_sum > 0:
            atom_recall3.update(rec3 / gt_sum, gt_sum)

        # ===== типы / заряды / H =====
        total_atom_types = atom_types.sum().item()
        total_atom_charges = atom_charges.sum().item()
        total_atom_hs = atom_hs.sum().item()

        if total_atom_types > 0:
            atom_type_gt = atom_types.argmax(1)
            atom_type_pred = atom_types_pred.argmax(1)
            correct = ((atom_type_gt == atom_type_pred).float() * atom_types.sum(1)).sum().item()
            atom_types_acc.update(correct / total_atom_types, total_atom_types)

        if total_atom_charges > 0:
            atom_charge_gt = atom_charges.argmax(1)
            atom_charge_pred = atom_charges_pred.argmax(1)
            correct = ((atom_charge_gt == atom_charge_pred).float() * atom_charges.sum(1)).sum().item()
            atom_charges_acc.update(correct / total_atom_charges, total_atom_charges)

        if total_atom_hs > 0:
            atom_hs_gt = atom_hs.argmax(1)
            atom_hs_pred_cls = atom_hs_pred.argmax(1)
            correct = ((atom_hs_gt == atom_hs_pred_cls).float() * atom_hs.sum(1)).sum().item()
            atom_hs_acc.update(correct / total_atom_hs, total_atom_hs)

        # ===================== СВЯЗИ =====================
        tp = (bond_peaks_pred * bond_peaks_gt).sum().item()
        pred_sum = bond_peaks_pred.sum().item()
        gt_sum = bond_peaks_gt.sum().item()

        if pred_sum > 0:
            bond_precision.update(tp / pred_sum, pred_sum)
        if gt_sum > 0:
            bond_recall.update(tp / gt_sum, gt_sum)

        # @3
        bond_gt_pooled = torch.nn.functional.max_pool2d(bond_peaks_gt, 3, 1, 1)
        tp3 = (bond_peaks_pred * bond_gt_pooled).sum().item()
        if pred_sum > 0:
            bond_precision3.update(tp3 / pred_sum, pred_sum)

        bond_pred_pooled = torch.nn.functional.max_pool2d(bond_peaks_pred, 3, 1, 1)
        rec3 = (bond_peaks_gt * bond_pred_pooled).sum().item()
        if gt_sum > 0:
            bond_recall3.update(rec3 / gt_sum, gt_sum)

        # ===== bond types =====
        total_bond_types = bond_types.sum().item()

        if total_bond_types > 0:
            bond_type_gt = bond_types.argmax(1)
            bond_type_pred = bond_types_pred.argmax(1)
            correct = ((bond_type_gt == bond_type_pred).float() * bond_types.sum(1)).sum().item()
            bond_types_acc.update(correct / total_bond_types, total_bond_types)

        # For bond rhos MAE, we still use bond_peaks_gt as the mask
        total_bonds = bond_peaks_gt.sum().item()
        if total_bonds > 0:
            mae = (torch.abs(bond_rhos_pred - bond_rhos) * bond_peaks_gt).sum().item() / total_bonds
            bond_rhos_mae.update(mae, total_bonds)

        # ===================== SMILES =====================
        for i in range(imgs.shape[0]):
            global_idx = batch_idx * BATCH_SIZE + i

            gt_smiles = df.loc[global_idx, 'Smiles']
            gt_smiles = canonical(gt_smiles)

#             atoms, bonds = extract_graph(
#                 atom_peaks_pred[i],
#                 atom_types_pred[i],
#                 bond_peaks_pred[i],
#                 bond_types_pred[i],
#                 bond_rhos_pred[i],
#                 bond_omega_types_pred[i]
#             )

#             edges = match_bonds_to_atoms(atoms, bonds)
#             pred_smiles = predict_smiles(atoms, edges)
            pred_smiles = predict_smiles_full(
            atom_peaks_pred[i],
            atom_types_pred[i],
            atom_charges_pred[i],
            atom_hs_pred[i],
            bond_peaks_pred[i],
            bond_types_pred[i],
            bond_rhos_pred[i],
            bond_omega_types_pred[i]
        )

            if pred_smiles is not None:
                pred_smiles = canonical(pred_smiles)

            total_processed += 1

            if gt_smiles is not None and pred_smiles is not None:
                if gt_smiles == pred_smiles:
                    exact_match_count += 1

        print(f"Batch {batch_idx} processed")


# ========= RESULTS =========
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ")
print("="*60)

print("\n--- АТОМЫ ---")
print(f"Precision:   {atom_precision.avg:.4f}")
print(f"Recall:      {atom_recall.avg:.4f}")
print(f"Precision@3: {atom_precision3.avg:.4f}")
print(f"Recall@3:    {atom_recall3.avg:.4f}")
print(f"Types Acc:   {atom_types_acc.avg:.4f}")
print(f"Charges Acc: {atom_charges_acc.avg:.4f}")
print(f"H-count Acc: {atom_hs_acc.avg:.4f}")

print("\n--- СВЯЗИ ---")
print(f"Precision:   {bond_precision.avg:.4f}")
print(f"Recall:      {bond_recall.avg:.4f}")
print(f"Precision@3: {bond_precision3.avg:.4f}")
print(f"Recall@3:    {bond_recall3.avg:.4f}")
print(f"Types Acc:   {bond_types_acc.avg:.4f}")
print(f"Rho MAE:     {bond_rhos_mae.avg:.4f}")

print("\n--- SMILES ---")
print(f"Exact match: {exact_match_count}/{total_processed}")
print(f"Accuracy:    {100 * exact_match_count / total_processed:.2f}%")