import pandas as pd
import os

def fix_csv_files(directory):
    # Path to Clip4's Label.csv
    csv_path = os.path.join(directory, 'game1', 'Clip3', 'Label.csv')
    
    if os.path.exists(csv_path):
        print(f"Processing: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Set empty string for X, Y, and Trajectory Pattern where VC = 0
        df.loc[df['Visibility Class'] == 0, ['X', 'Y', 'Trajectory Pattern']] = ''
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        print(f"Fixed: {csv_path}")
    else:
        print(f"File not found: {csv_path}")

if __name__ == "__main__":
    # Path to your dataset directory
    dataset_path = 'datasets/handball'
    fix_csv_files(dataset_path) 