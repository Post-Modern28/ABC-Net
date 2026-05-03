import pandas as pd
import gzip
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter
import numpy as np

print("1. Загрузка данных из локального файла ChEMBL...")
# Файл скачан с https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
# Распакуйте его командой: gunzip chembl_36_chemreps.txt.gz

# Читаем TSV файл
df = pd.read_csv('chembl_36_chemreps.txt', sep='\t', header=0)
print(f"Загружено {len(df)} молекул из файла.")

# Оставляем только нужные колонки и переименовываем
df = df[['chembl_id', 'canonical_smiles']].copy()
df.rename(columns={'chembl_id': 'ChEMBL ID', 'canonical_smiles': 'Smiles'}, inplace=True)

# Удаляем строки с пустыми SMILES
df = df.dropna(subset=['Smiles'])
print(f"После удаления пустых SMILES: {len(df)} молекул.")

# --- Шаг 2 & 3: Первичная очистка с RDKit и фильтрация по размеру ---
print("2. Первичная очистка с RDKit и фильтрация молекул по размеру (<= 50 тяжелых атомов)...")

def is_valid_and_small(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        # Подсчет тяжелых (неводородных) атомов
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)
        return heavy_atom_count <= 50
    except:
        return False

# Применяем фильтр
mask = df['Smiles'].apply(is_valid_and_small)
df_valid = df[mask].copy()
print(f"После первичной очистки и фильтрации по размеру осталось {len(df_valid)} молекул.")

# --- Шаг 4: Анализ и фильтрация по редкости ---
print("3. Анализ частоты встречаемости атомов и зарядов...")

def get_atom_and_charge_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        atoms = []
        charges = []
        if mol:
            for atom in mol.GetAtoms():
                atoms.append(atom.GetSymbol())
                charges.append(str(atom.GetFormalCharge()))
        return atoms, charges
    except:
        return [], []

# Собираем информацию со всех молекул
print("   Сбор статистики по атомам и зарядам...")
all_atoms = []
all_charges = []
total = len(df_valid)

# Для ускорения обрабатываем с прогрессом
for i, smi in enumerate(df_valid['Smiles']):
    if i % 100000 == 0:
        print(f"   Обработано {i}/{total} молекул ({100*i/total:.1f}%)")
    atoms, charges = get_atom_and_charge_info(smi)
    all_atoms.extend(atoms)
    all_charges.extend(charges)

# Подсчитываем частоты
atom_counts = Counter(all_atoms)
charge_counts = Counter(all_charges)

# Определяем "разрешенные" атомы и заряды (встречаются >= 1000 раз)
allowed_atoms = {atom for atom, count in atom_counts.items() if count >= 1000}
allowed_charges = {charge for charge, count in charge_counts.items() if count >= 1000}

print(f"Разрешено атомов: {len(allowed_atoms)}")
print(f"Разрешено зарядов: {len(allowed_charges)}")

# Функция для проверки молекулы на отсутствие "редких" признаков
def has_only_allowed_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in allowed_atoms:
                return False
            if str(atom.GetFormalCharge()) not in allowed_charges:
                return False
        return True
    except:
        return False

# Применяем финальный фильтр
print("4. Фильтрация молекул с редкими признаками...")
mask_final = df_valid['Smiles'].apply(has_only_allowed_features)
df_final = df_valid[mask_final].copy()
print(f"После удаления молекул с редкими признаками осталось {len(df_final)} молекул.")

# --- Шаг 5: Сбалансированная случайная выборка 100,000 молекул ---
print("5. Формирование сбалансированной случайной выборки из 100 000 молекул...")

if len(df_final) < 100000:
    print(f"Внимание: Найдено только {len(df_final)} молекул после фильтрации. Будет использована вся выборка.")
    df_sample = df_final
else:
    # Случайная выборка
    df_sample = df_final.sample(n=100000, random_state=42)

print(f"Финальный размер выборки: {len(df_sample)} молекул.")

# Сохраняем результат в CSV
df_sample[['ChEMBL ID', 'Smiles']].to_csv('filtered.csv', index=False)
print("Готово! Файл 'filtered.csv' создан.")
print("\nСтатистика финальной выборки:")
print(df_sample.head())