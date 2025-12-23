import pandas as pd
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--prefix', type=str, default='', help='Prefix to add to audio file paths')
parser.add_argument('--hf_dataset_path', type=str, default=None, help='Hugging Face dataset path')
parser.add_argument('--accents', nargs='*', default=None, help='List of accents to filter by')
parser.add_argument('--output_dir', type=str, required=True, help="Directory in which to save the data")
args = parser.parse_args()

ids_to_save = []

if args.hf_dataset_path:

    from datasets import load_from_disk
    ds = load_from_disk(args.hf_dataset_path)
    for idx, accent in enumerate(ds['annotated_accent']):
        if accent in args.accents: ids_to_save.append(idx)

    print(f"Mantenemos {len(ids_to_save)} filas.")




# Cargar el DataFrame
df = pd.read_csv(args.input_csv)

if ids_to_save:
    print(f"Filtrando dataset. IDs seleccionados: {len(ids_to_save)}")
    
    # 1. Convertimos la lista a un set para búsqueda O(1) (mucho más rápido)
    # 2. Formateamos el ID: f"audio_{idx:06d}.wav" asegura los 6 dígitos con ceros a la izquierda
    files_to_keep = set(f"audio_{idx:06d}.wav" for idx in ids_to_save)
    
    # 3. Filtramos el DataFrame manteniendo solo los que están en el set
    df = df[df['audio_file'].isin(files_to_keep)]

    if df.empty:
        raise ValueError("El filtrado ha eliminado todas las filas. Verifica que los IDs coincidan con los nombres de archivo en el CSV.")

# Asegurarse de que existe el directorio de salida
os.makedirs('data', exist_ok=True)

# Mezclar y dividir los datos
# random_state asegura que el experimento sea reproducible
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extraer 200 muestras para validación y el resto para entrenamiento
df_valid = df_shuffled.head(200)
df_train = df_shuffled.tail(len(df_shuffled) - 200)

def save_to_jsonl(dataframe, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            line_data = {
                "audio": args.prefix + str(row["audio_file"]), 
                "text": row["text"], 
                "duration": row["duration"]
            }
            json_line = json.dumps(line_data, ensure_ascii=False)
            f.write(json_line + '\n')

# Guardar ambos archivos
os.makedirs(args.output_dir, exist_ok=True)

save_to_jsonl(df_train, f'{args.output_dir}/train.jsonl')
save_to_jsonl(df_valid, f'{args.output_dir}/valid.jsonl')

print(f"Proceso completado. Train: {len(df_train)} muestras, Valid: {len(df_valid)} muestras.")