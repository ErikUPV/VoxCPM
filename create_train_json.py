import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--prefix', type=str, default='', help='Prefix to add to audio file paths')
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

# Use 'train.jsonl' as the extension
with open('data/train.jsonl', 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        # Create the dictionary for the current row
        line_data = {
            "audio": args.prefix + row["audio_file"], 
            "text": row["text"], 
            "duration": row["duration"]
        }
        
        # Write the dictionary as a single JSON string, followed by a newline
        # ensure_ascii=False is important for non-English characters in 'text'
        json_line = json.dumps(line_data, ensure_ascii=False)
        f.write(json_line + '\n')