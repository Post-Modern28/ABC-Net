import os

import pandas as pd

# Путь к исходному файлу
input_csv = '../train_data/processed_chembl.csv'
# Путь для сохранения нового файла
output_csv = '../train_data/test_chembl.csv'

# Проверяем, существует ли входной файл
if not os.path.exists(input_csv):
    print(f"Ошибка: файл {input_csv} не найден!")
    exit(1)

# Читаем CSV
df = pd.read_csv(input_csv)
total = len(df)
print(f"Всего записей: {total}")

# Выбираем с 91000 индекса до конца
start_idx = 91000
df_unseen = df.iloc[start_idx:].copy()
print(f"Выбрано записей с индекса {start_idx} до {total-1}: {len(df_unseen)}")

# Сохраняем в новый CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_unseen.to_csv(output_csv, index=False)
print(f"Сохранено в {output_csv}")
