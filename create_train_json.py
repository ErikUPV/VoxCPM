import pandas as pd
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--prefix', type=str, default='', help='Prefix to add to audio file paths')
args = parser.parse_args()

# Cargar el DataFrame
df = pd.read_csv(args.input_csv)

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
save_to_jsonl(df_train, 'data/train.jsonl')
save_to_jsonl(df_valid, 'data/valid.jsonl')

print(f"Proceso completado. Train: {len(df_train)} muestras, Valid: {len(df_valid)} muestras.")