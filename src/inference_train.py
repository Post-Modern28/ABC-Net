import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import MolecularImageDataset, collate_fn
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========= CONFIG =========
CSV_PATH = '../train_data/processed_chembl.csv'
WEIGHTS_PATH = 'weights/unet_model_weights29.pkl'
SAVE_DIR = 'debug_outputs'
BATCH_SIZE = 8
NUM_SAMPLES = 200  # ограничим для дебага

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= DATA =========
df = pd.read_csv(CSV_PATH)
df = df[:NUM_SAMPLES].reset_index(drop=True)

dataset = MolecularImageDataset(df, amount=0.0)  # без аугментаций
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        collate_fn=collate_fn, shuffle=False)

# ========= MODEL =========
model = UNet(in_channels=1, heads=[1,14,3,2,1,360,60,60])
model = nn.DataParallel(model)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# ========= INFERENCE =========
with torch.no_grad():
    for batch_idx, (imgs, atom_targets, atom_types, atom_charges,
                    atom_hs, bond_targets, bond_types,
                    bond_rhos, bond_omega_types) in enumerate(dataloader):

        imgs = imgs.to(device)

        outputs = model(imgs)
        (atom_targets_pred,
         atom_types_pred,
         atom_charges_pred,
         atom_hs_pred,
         bond_targets_pred,
         bond_types_pred,
         bond_rhos_pred,
         bond_omega_types_pred) = outputs

        # ===== NORMALIZATION =====
        atom_targets_pred = torch.sigmoid(atom_targets_pred)
        bond_targets_pred = torch.sigmoid(bond_targets_pred)
        bond_omega_types_pred = torch.sigmoid(bond_omega_types_pred)

        atom_types_pred = torch.softmax(atom_types_pred, dim=1)
        atom_charges_pred = torch.softmax(atom_charges_pred, dim=1)
        atom_hs_pred = torch.softmax(atom_hs_pred, dim=1)

        bond_types_pred = torch.softmax(
            bond_types_pred.view(-1, 6, 60, 128, 128), dim=1
        )

        bond_rhos_pred = torch.abs(bond_rhos_pred)

        # ===== PEAK DETECTION (как в train) =====
        temp = torch.nn.functional.max_pool2d(atom_targets_pred, 3, 1, 1)
        atom_peaks = ((temp == atom_targets_pred) & (atom_targets_pred > 0.3)).float()

        temp = torch.nn.functional.max_pool2d(bond_targets_pred, 3, 1, 1)
        bond_peaks = ((temp == bond_targets_pred) & (bond_targets_pred > 0.3)).float()

        # ===== DEBUG PRINT =====
        print(f"\nBatch {batch_idx}")
        print("Avg atoms detected:", atom_peaks.sum(dim=[1,2,3]).mean().item())
        print("Avg bonds detected:", bond_peaks.sum(dim=[1,2,3]).mean().item())

        # ===== SAVE VISUALIZATION =====
        for i in range(min(4, imgs.shape[0])):

            img = imgs[i, 0].cpu().numpy()
            atom_pred = atom_targets_pred[i, 0].cpu().numpy()
            bond_pred = bond_targets_pred[i, 0].cpu().numpy()

            atom_peak = atom_peaks[i, 0].cpu().numpy()
            bond_peak = bond_peaks[i, 0].cpu().numpy()

            plt.figure(figsize=(12,4))

            plt.subplot(1,4,1)
            plt.title("Input")
            plt.imshow(img, cmap='gray')

            plt.subplot(1,4,2)
            plt.title("Atom heatmap")
            plt.imshow(atom_pred)

            plt.subplot(1,4,3)
            plt.title("Atom peaks")
            plt.imshow(atom_peak)

            plt.subplot(1,4,4)
            plt.title("Bond peaks")
            plt.imshow(bond_peak)

            plt.savefig(f"{SAVE_DIR}/batch{batch_idx}_img{i}.png")
            plt.close()
