import numpy as np
import pandas as pd
import os
import cv2
import argparse

def gaussian_kernel(size, variance):
    """Create a Gaussian kernel with specified size and variance"""
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    """Create a normalized Gaussian heatmap"""
    # Create Gaussian kernel
    gaussian_kernel_array = gaussian_kernel(size, variance)
    
    # Normalize to [0, 255]
    gaussian_kernel_array = gaussian_kernel_array * 255/gaussian_kernel_array.max()
    gaussian_kernel_array = gaussian_kernel_array.astype(np.uint8)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height):
    """Create ground truth heatmap images"""
    # Generate Gaussian kernel once
    gaussian_kernel_array = create_gaussian(size, variance)
    kernel_size = gaussian_kernel_array.shape[0]
    half_kernel = kernel_size // 2
    
    # Create output directory if it doesn't exist
    if not os.path.exists(path_output):
        os.makedirs(path_output)
        
    for game_id in range(1, 11):
        game = 'game{}'.format(game_id)
        if not os.path.exists(os.path.join(path_input, game)):
            continue
            
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            if not os.path.exists(os.path.join(path_input, game, clip, 'Label.csv')):
                continue
                
            labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
            
            for idx, row in labels.iterrows():
                # Create empty heatmap
                heatmap = np.zeros((height, width), dtype=np.uint8)
                
                # Only create Gaussian for visible balls (VC 1 or 2)
                if row['Visibility Class'] in [1, 2] and not np.isnan(row['X']) and not np.isnan(row['Y']):
                    x, y = int(row['X']), int(row['Y'])
                    
                    # Calculate boundaries for the Gaussian
                    left = max(0, x - half_kernel)
                    right = min(width, x + half_kernel + 1)
                    top = max(0, y - half_kernel)
                    bottom = min(height, y + half_kernel + 1)
                    
                    # Calculate corresponding Gaussian kernel boundaries
                    kernel_left = half_kernel - (x - left)
                    kernel_right = half_kernel + (right - x)
                    kernel_top = half_kernel - (y - top)
                    kernel_bottom = half_kernel + (bottom - y)
                    
                    # Place Gaussian on heatmap
                    heatmap[top:bottom, left:right] = gaussian_kernel_array[
                        kernel_top:kernel_bottom,
                        kernel_left:kernel_right
                    ]
                
                # Save heatmap
                output_path = os.path.join(path_output, game, clip)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(os.path.join(output_path, row['File Name']), heatmap)

def create_gt_labels(path_input, path_output, train_rate=0.7):
    df = pd.DataFrame()
    for game_id in range(1,11):
        game = 'game{}'.format(game_id)
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
            labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
            labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']
            labels_target = labels[2:]
            labels_target.loc[:, 'path2'] = list(labels['path1'][1:-1])
            labels_target.loc[:, 'path3'] = list(labels['path1'][:-2])
            df = df.append(labels_target)
    df = df.reset_index(drop=True) 
    df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    df = df.sample(frac=1)
    num_train = int(df.shape[0]*train_rate)
    df_train = df[:num_train]
    df_test = df[num_train:]
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)  


if __name__ == '__main__':
    SIZE = 5  # Half size of the Gaussian kernel (smaller for ball)
    VARIANCE = 2  # Controls the spread of the Gaussian (smaller for ball)
    WIDTH = 1280  # Width of output heatmap
    HEIGHT = 720  # Height of output heatmap

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', type=str, help='path to input folder')
    parser.add_argument('--path_output', type=str, help='path to output folder')
    args = parser.parse_args()
    
    create_gt_images(args.path_input, args.path_output, SIZE, VARIANCE, WIDTH, HEIGHT)
    create_gt_labels(args.path_input, args.path_output)

                            
    
    
    
    