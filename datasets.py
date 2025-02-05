from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np
from sklearn.model_selection import train_test_split
        
class trackNetDataset(Dataset):
    def __init__(self, mode, input_height=720, input_width=1280, sequence_length=3):
        self.path_dataset = 'datasets/handball'
        assert mode in ['train', 'val'], 'incorrect mode'
        
        # Initialize DataFrame to store all labels
        self.data = pd.DataFrame()
        
        # List of clips to load
        clips = [
            ('game1', 'Clip4'),
            #('game1', 'Clip5')
        ]
        
        # Load all clips
        for game, clip in clips:
            label_file = os.path.join(self.path_dataset, game, clip, 'Label.csv')
            if os.path.exists(label_file):
                labels = pd.read_csv(label_file)
                labels['game'] = game
                labels['clip'] = clip
                self.data = pd.concat([self.data, labels], ignore_index=True)
            else:
                print(f"Label file not found: {label_file}")
        
        # Split the combined data into train and validation sets
        if len(self.data) > 0:
            # Convert any VC=3 to VC=0 (using correct column name)
            self.data.loc[self.data['Visibility Class'] == 3, 'Visibility Class'] = 0
            # Verify only valid classes remain
            assert self.data['Visibility Class'].isin([0, 1, 2]).all(), "Invalid visibility class found"
            
            train_data, val_data = train_test_split(
                self.data, 
                test_size=0.2,
                random_state=42
            )
            
            # Assign appropriate split based on mode
            if mode == 'train':
                self.data = train_data
            else:
                self.data = val_data
                
            print(f'Mode: {mode}, Total samples: {len(self.data)}')
            print(f'Data from clips: {self.data["clip"].unique()}')
        else:
            raise RuntimeError("No data was loaded!")
        
        self.height = input_height
        self.width = input_width
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['File Name']
        visibility_class = row['Visibility Class']
        x, y = row['X'], row['Y']
        
        game = row['game']
        clip = row['clip']
        
        # Load 3 consecutive frames
        frames = []
        for i in range(self.sequence_length):
            prev_frame = int(file_name.split('.')[0]) - i
            path = os.path.join(self.path_dataset, game, clip, f"{prev_frame:04d}.jpg")
            if os.path.exists(path):
                img = cv2.imread(path)
                img = cv2.resize(img, (self.width, self.height))
            else:
                img = np.zeros((self.height, self.width, 3), dtype=np.float32)
            frames.append(img)
            
        # Concatenate frames
        imgs = np.concatenate(frames, axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        
        # Create ground truth heatmap
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Only create Gaussian blob for visible balls
        if visibility_class > 0 and not np.isnan(x) and not np.isnan(y):
            # Scale coordinates to match resized dimensions
            x_scaled = int(x * self.width / 1280)
            y_scaled = int(y * self.height / 720)
            
            # Create Gaussian blob with larger size and spread
            sigma = 10  # Increased from 2.5 to 10
            size = 30   # Increased from 15 to 30
            x_grid, y_grid = np.meshgrid(
                np.arange(max(0, x_scaled - size), min(self.width, x_scaled + size + 1)) - x_scaled,
                np.arange(max(0, y_scaled - size), min(self.height, y_scaled + size + 1)) - y_scaled
            )
            gaussian = np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))
            
            # Place Gaussian on heatmap
            y_start = max(0, y_scaled - size)
            y_end = min(self.height, y_scaled + size + 1)
            x_start = max(0, x_scaled - size)
            x_end = min(self.width, x_scaled + size + 1)
            
            heatmap[y_start:y_end, x_start:x_end] = gaussian
        
        # Don't threshold the Gaussian - keep the continuous values
        heatmap = heatmap.reshape(-1)  # Flatten to 1D array
        
        return imgs, heatmap, x, y, visibility_class
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        img = np.reshape(img, (self.width * self.height))
        return img        
    
    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))

        # Check if previous image exists
        if os.path.exists(path_prev):
            img_prev = cv2.imread(path_prev)
            img_prev = cv2.resize(img_prev, (self.width, self.height))
        else:
            img_prev = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Check if pre-previous image exists
        if os.path.exists(path_preprev):
            img_preprev = cv2.imread(path_preprev)
            img_preprev = cv2.resize(img_preprev, (self.width, self.height))
        else:
            img_preprev = np.zeros((self.height, self.width, 3), dtype=np.float32)

        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0

        imgs = np.rollaxis(imgs, 2, 0)
        return imgs
