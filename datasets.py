from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np

class trackNetDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640):
        self.path_dataset = './Dataset'  # Update this to your dataset's base directory
        assert mode in ['train', 'val'], 'incorrect mode'
        
        # Initialize a list to store all data
        data_list = []
        
        # Check if the base dataset directory exists
        if not os.path.exists(self.path_dataset):
            raise FileNotFoundError(f"The dataset directory {self.path_dataset} does not exist.")
        
        # Iterate over each game directory
        for game_id in range(1, 11):
            game_path = os.path.join(self.path_dataset, f'game{game_id}')
            if not os.path.exists(game_path):
                print(f"Game directory {game_path} does not exist. Skipping.")
                continue
            
            clips = os.listdir(game_path)
            
            # Iterate over each clip directory
            for clip in clips:
                clip_path = os.path.join(game_path, clip)
                label_file = os.path.join(clip_path, 'Label.csv')
                
                if os.path.exists(label_file):
                    labels = pd.read_csv(label_file)
                    # Add additional columns for paths if needed
                    labels['path'] = clip_path
                    data_list.append(labels)
        
        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()
        
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))
        self.height = input_height
        self.width = input_width
        
        # New handball dimensions (higher resolution needed for larger ball)
        self.resize_height = 540
        self.resize_width = 960
        
        # New handball sequence length (2 frames sufficient for slower ball)
        self.sequence_length = 2
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Ensure there are enough frames to form a sequence
        if idx < 2:
            raise IndexError("Not enough frames to form a sequence")

        # Get paths for the current frame and the two preceding frames
        paths = [
            os.path.join(self.data.loc[idx - 2, 'path'], self.data.loc[idx - 2, 'File Name']),
            os.path.join(self.data.loc[idx - 1, 'path'], self.data.loc[idx - 1, 'File Name']),
            os.path.join(self.data.loc[idx, 'path'], self.data.loc[idx, 'File Name'])
        ]

        # Load and preprocess the images
        images = []
        for path in paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.width, self.height))
            img = img.astype(np.float32) / 255.0
            img = np.rollaxis(img, 2, 0)  # Change from HWC to CHW format
            images.append(img)

        # Stack images to form a sequence
        sequence = np.stack(images, axis=0)

        # Get labels for the current frame
        x = self.data.loc[idx, 'X']
        y = self.data.loc[idx, 'Y']
        status = self.data.loc[idx, 'Trajectory Pattern']
        vis = self.data.loc[idx, 'Visibility Class']

        return sequence, x, y, status, vis
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        img = np.reshape(img, (self.width * self.height))
        return img
        
    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))

        img_prev = cv2.imread(path_prev)
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        
        img_preprev = cv2.imread(path_preprev)
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))
        
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0

        imgs = np.rollaxis(imgs, 2, 0)
        return imgs
