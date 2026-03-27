import pandas as pd
import json
import os

files_to_shrink = [
    'DataSets/PAYSim/processed_paysim.csv',
    'DataSets/Synthetic/processed_nasa.csv',
    'DataSets/Credit Card Fraud/processed_cc.csv'
]

for f in files_to_shrink:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if len(df) > 5000:
            # We want to keep anomalies if possible to ensure dashboard isn't empty
            if 'Is_Anomaly' in df.columns:
                anomalies = df[df['Is_Anomaly'] == 1]
                normal = df[df['Is_Anomaly'] == 0].sample(min(4000, len(df[df['Is_Anomaly'] == 0])), random_state=42)
                df = pd.concat([anomalies, normal])
            else:
                df = df.sample(min(5000, len(df)), random_state=42)
            
            df.to_csv(f, index=False)
            print(f"Shrunk {f} to {len(df)} rows")
