import os
import pandas as pd
from rdkit import Chem

# Пути
MOL_DIR = '../datasets/UOB_mol_ref'
IMG_DIR = '../datasets/UOB'
OUTPUT_CSV = '../data2/UOB/uob.csv'

MOL_DIR = '../datasets/USPTO_mol_ref'
IMG_DIR = '../datasets/USPTO'
OUTPUT_CSV = 'data2/USPTO/uspto.csv'

# Создаём папку для вывода
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

data = []
mol_files = sorted([f for f in os.listdir(MOL_DIR) if f.lower().endswith('.mol')])

print(f"Найдено {len(mol_files)} .mol файлов")

for mol_file in mol_files:
    mol_path = os.path.join(MOL_DIR, mol_file)
    
    # Имя без расширения
    base_name = os.path.splitext(mol_file)[0]
    
    # Ищем соответствующее изображение (может быть .png, .jpg и т.д.)
    img_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
        potential_path = os.path.join(IMG_DIR, base_name + ext)
        if os.path.exists(potential_path):
            img_path = os.path.abspath(potential_path)
            break
    
    if img_path is None:
        print(f"⚠️ Изображение для {mol_file} не найдено, пропускаем")
        continue
    
    # Читаем .mol и конвертируем в SMILES
    try:
        mol = Chem.MolFromMolFile(mol_path)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            data.append({
                'path': img_path,
                'smiles': smiles
            })
        else:
            print(f"❌ Не удалось прочитать {mol_file}")
    except Exception as e:
        print(f"❌ Ошибка при обработке {mol_file}: {e}")

# Сохраняем CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Создан CSV: {OUTPUT_CSV}")
print(f"   Всего записей: {len(df)}")