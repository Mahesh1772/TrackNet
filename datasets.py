from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np
        
class trackNetDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640):
        # Hardcode the path to your specific clip
        self.path_dataset = 'C:/Users/Admin/Documents/Personal_TrackNet/datasets/handball'
        assert mode in ['train', 'val'], 'incorrect mode'
        
        # Initialize DataFrame to store labels
        self.data = pd.DataFrame()
        
        # Only read the specific clip's label file
        label_file = os.path.join(self.path_dataset, 'game1', 'Clip4', 'Label.csv')
        if os.path.exists(label_file):
            labels = pd.read_csv(label_file)
            labels['game'] = 'game1'
            labels['clip'] = 'Clip4'
            self.data = labels
        else:
            print(f"Label file not found: {label_file}")
        
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))         
        self.height = input_height
        self.width = input_width 

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['File Name']
        visibility_class = row['Visibility Class']
        x = row['X']
        y = row['Y']
        trajectory_pattern = row['Trajectory Pattern']
        
        game = row['game']
        clip = row['clip']
        
        # Calculate paths for current, previous, and pre-previous frames
        path = os.path.join(self.path_dataset, game, clip, file_name)
        frame_number = int(file_name.split('.')[0])
        path_prev = os.path.join(self.path_dataset, game, clip, f"{frame_number - 1:04d}.jpg")
        path_preprev = os.path.join(self.path_dataset, game, clip, f"{frame_number - 2:04d}.jpg")
        
        # Load and process the frames
        inputs = self.get_input(path, path_prev, path_preprev)
        outputs = self.get_output(path)
        
        return inputs, outputs, x, y, visibility_class
    
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