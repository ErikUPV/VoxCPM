import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--prefix', type=str, default='', help='Prefix to add to audio file paths')
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

train_jsons = []

for index, row in df.iterrows():
    train_json = {"audio" : args.prefix + row["audio_file"], "text" : row["text"], "duration" : row["duration"]}
    train_jsons.append(train_json)

with open('train.json', 'w') as f:
    json.dump(train_jsons, f, indent=4)