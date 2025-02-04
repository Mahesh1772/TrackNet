from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np
        
class trackNetDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640):
        # Update the path to match your dataset location
        self.path_dataset = 'C:/Users/Admin/Documents/Personal_TrackNet/datasets/images'
        assert mode in ['train', 'val'], 'incorrect mode'
        
        # Initialize an empty DataFrame to store all labels
        self.data = pd.DataFrame()
        
        # Iterate over each game and clip to read the label files
        for game in os.listdir(self.path_dataset):
            game_path = os.path.join(self.path_dataset, game)
            if os.path.isdir(game_path):
                for clip in os.listdir(game_path):
                    clip_path = os.path.join(game_path, clip)
                    if os.path.isdir(clip_path):
                        label_file = os.path.join(clip_path, 'Label.csv')
                        if os.path.exists(label_file):
                            labels = pd.read_csv(label_file)
                            labels['game'] = game
                            labels['clip'] = clip
                            self.data = pd.concat([self.data, labels], ignore_index=True)
                        else:
                            print(f"    Label file not found: {label_file}")
        
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))         
        self.height = input_height
        self.width = input_width 

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['file name']
        visibility_class = row['visibility']
        x = row['x-coordinate']
        y = row['y-coordinate']
        trajectory_pattern = row['status']
        
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