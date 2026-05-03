"""
Инференс для датасета с колонками: path, Smiles
(без atoms_string и bonds_string)
"""

from generate_smiles import sdf2smiles
from rdkit import Chem
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from unet import UNet
from my_utils_for_test import (
    canonical,
    extract_graph,
    match_bonds_to_atoms,
    predict_smiles
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========= CONFIG =========
CSV_PATH = '../data2/UOB/uob.csv'  # Ваш датасет с колонками path, smiles
# CSV_PATH = '../train_data/processed_chembl.csv'  # Оригинальный (закомментирован)
# CSV_PATH = '../train_data/test_chembl.csv'  # Оригинальный (закомментирован)

WEIGHTS_PATH = 'weights/unet_model_weights29.pkl'
SAVE_DIR = 'debug_outputs/uob'
# SAVE_DIR = 'debug_outputs/rdkit_test'
BATCH_SIZE = 8
NUM_SAMPLES = 10_000

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= ДАТАСЕТ ДЛЯ ИНФЕРЕНСА (без atoms_string, bonds_string) =========
class SimpleInferenceDataset(Dataset):
    """
    Датасет для CSV с колонками: path, Smiles
    Предобработка полностью соответствует тренировочной (utils.py),
    но с корректной обработкой любых размеров изображений.
    """
    def __init__(self, df, amount=0.0):
        self.df = df.reset_index(drop=True)
        self.amount = amount

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        smiles = self.df.loc[idx, 'Smiles'] if 'Smiles' in self.df.columns else None
        
        if not path.startswith('/beegfs'):
            path = '../' + path
        
        # Загрузка
        temp_img = cv2.imread(path, flags=0)
        if temp_img is None:
            raise ValueError(f"Cannot load: {path}")
        temp_img = temp_img.astype('float32')
        
        # Ресайз с сохранением пропорций, чтобы вписать в 512x512
        h, w = temp_img.shape
        scale = min(512 / h, 512 / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        temp_img = cv2.resize(temp_img, (new_w, new_h))
        
        # Центрирование на белом холсте 512x512
        ddx = (512 - new_h) // 2
        ddy = (512 - new_w) // 2
        
        temp_img2 = np.ones((512, 512), dtype='float32') * 255
        temp_img2[ddx:ddx+new_h, ddy:ddy+new_w] = temp_img
        
        # Бинаризация (как в трейне)
        temp_img = ((temp_img2 / 255.0) < 0.6) * 1.0
        
        # Шум (если amount > 0)
        if self.amount > 0:
            salt_amount = np.random.uniform(0, self.amount / 100)
            salt = np.random.uniform(0, 1, temp_img.shape) < salt_amount
            temp_img = np.logical_or(temp_img, salt)
            pepper_amount = np.random.uniform(0, self.amount)
            pepper = np.random.uniform(0, 1, temp_img.shape) < pepper_amount
            temp_img = np.logical_or(1 - temp_img, pepper)
            temp_img = temp_img * 1.0
        
        # Инверсия (КАК ВЫ ИСПРАВИЛИ)
        img = np.zeros([1, 512, 512], dtype='float32')
        img[0] = temp_img  # без инверсии, т.к. выяснили, что так работает
        
        return img, smiles, path

def simple_collate_fn(batch):
    """
    Collate функция для батчей из SimpleInferenceDataset
    """
    imgs, smiles_list, paths = zip(*batch)
    imgs = [np.expand_dims(img, 0) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)
    imgs = torch.from_numpy(imgs)
    return imgs, smiles_list, paths


# ========= ЗАГРУЗКА ДАННЫХ =========
print(f"Loading data from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Проверяем колонки
print(f"Columns: {df.columns.tolist()}")
if 'smiles' in df.columns:
    df.rename(columns={'smiles': 'Smiles'}, inplace=True)

df = df[:NUM_SAMPLES].reset_index(drop=True)
print(f"Loaded {len(df)} samples")

dataset = SimpleInferenceDataset(df, amount=0.0)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        collate_fn=simple_collate_fn, shuffle=False)

# ========= ЗАГРУЗКА МОДЕЛИ =========
print(f"Loading model from: {WEIGHTS_PATH}")
model = UNet(in_channels=1, heads=[1, 14, 3, 2, 1, 360, 60, 60])
model = nn.DataParallel(model)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("Model loaded successfully")

# ========= ИНФЕРЕНС =========
results = []

with torch.no_grad():
    for batch_idx, (imgs, smiles_batch, paths_batch) in enumerate(dataloader):
        imgs = imgs.to(device)

        # Предсказание модели
        outputs = model(imgs)
        (atom_targets_pred,
         atom_types_pred,
         atom_charges_pred,
         atom_hs_pred,
         bond_targets_pred,
         bond_types_pred,
         bond_rhos_pred,
         bond_omega_types_pred) = outputs

        # Нормализация
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

        # Peak detection
        temp = torch.nn.functional.max_pool2d(atom_targets_pred, 3, 1, 1)
        atom_peaks = ((temp == atom_targets_pred) & (atom_targets_pred > 0.3)).float()

        temp = torch.nn.functional.max_pool2d(bond_targets_pred, 3, 1, 1)
        bond_peaks = ((temp == bond_targets_pred) & (bond_targets_pred > 0.3)).float()

        print(f"\n{'='*50}")
        print(f"Batch {batch_idx}")
        print(f"Avg atoms detected: {atom_peaks.sum(dim=[1,2,3]).mean().item():.1f}")
        print(f"Avg bonds detected: {bond_peaks.sum(dim=[1,2,3]).mean().item():.1f}")
        print(f"{'='*50}")

        # Обработка каждого изображения в батче
        for i in range(imgs.shape[0]):
            global_idx = batch_idx * BATCH_SIZE + i

            # Визуализация (первые 4 из батча)
            if i < 4:
                img = imgs[i, 0].cpu().numpy()
                atom_pred = atom_targets_pred[i, 0].cpu().numpy()
                bond_pred = bond_targets_pred[i, 0].cpu().numpy()
                atom_peak = atom_peaks[i, 0].cpu().numpy()
                bond_peak = bond_peaks[i, 0].cpu().numpy()

                # Извлекаем имя файла из пути
                filename = os.path.basename(paths_batch[i])

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
                # Сохраняем с именем файла
                safe_filename = filename.replace('/', '_').replace('\\', '_')
                plt.savefig(f"{SAVE_DIR}/batch{batch_idx}_img{i}_{safe_filename}.png", dpi=150)
                plt.close()
                
                
            # GT SMILES
            gt_smiles_raw = smiles_batch[i]
            gt_smiles = canonical(gt_smiles_raw) if gt_smiles_raw else None

            # Извлечение графа
            atoms, bonds = extract_graph(
                atom_peaks[i],
                atom_types_pred[i],
                bond_peaks[i],
                bond_types_pred[i],
                bond_rhos_pred[i],
                bond_omega_types_pred[i]
            )

            edges = match_bonds_to_atoms(atoms, bonds)

            # Предсказание SMILES
            pred_smiles = predict_smiles(atoms, edges)
            if pred_smiles is not None:
                pred_smiles = canonical(pred_smiles)

            # Сохраняем результат
            match = (gt_smiles == pred_smiles) if (gt_smiles and pred_smiles) else False
            results.append({
                'idx': global_idx,
                'path': paths_batch[i],
                'gt_smiles': gt_smiles,
                'pred_smiles': pred_smiles,
                'num_atoms': len(atoms),
                'num_bonds': len(bonds),
                'match': match
            })

            # Вывод
            print(f"\nImage {global_idx}: {os.path.basename(paths_batch[i])}")
            print(f"  GT:  {gt_smiles[:70] if gt_smiles else 'None'}...")
            print(f"  PR:  {pred_smiles[:70] if pred_smiles else 'None'}...")
            if gt_smiles and pred_smiles:
                if match:
                    print("  ✅ MATCH")
                else:
                    print("  ❌ MISMATCH")
            else:
                print("  ⚠️ INCOMPLETE")

# ========= СВОДКА =========
results_df = pd.DataFrame(results)
results_df.to_csv(f'{SAVE_DIR}/results.csv', index=False)

print("\n" + "="*50)
print("СВОДКА")
print("="*50)
print(f"Всего обработано: {len(results_df)}")
print(f"Успешных предсказаний (не None): {results_df['pred_smiles'].notna().sum()}")
print(f"Точных совпадений: {results_df['match'].sum()}")
if results_df['pred_smiles'].notna().sum() > 0:
    print(f"Точность: {results_df['match'].sum() / results_df['pred_smiles'].notna().sum() * 100:.2f}%")
print(f"\nСреднее атомов: {results_df['num_atoms'].mean():.1f}")
print(f"Среднее связей: {results_df['num_bonds'].mean():.1f}")
print(f"\nРезультаты сохранены в {SAVE_DIR}/")