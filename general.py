import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        out = model(batch[0].float().to(device))
        gt = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print('train | epoch = {}, iter = [{}|{}], loss = {}, time = {}'.format(epoch, iter_id, max_iters,
                                                                                round(loss.item(), 6), duration))
        losses.append(loss.item())
        
        if iter_id > max_iters - 1:
            break
        
    return np.mean(losses)

def visualize_heatmaps(input_frames, gt_heatmap, pred_heatmap, save_path=None, num_frames=10):
    """
    Visualize input frames, ground truth heatmap and predicted heatmap side by side
    for multiple frames
    """
    for frame_idx in range(min(num_frames, len(input_frames))):
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get current frame
        frame = input_frames[frame_idx].cpu().numpy()
        if frame.shape[0] == 9:  # If we have 3 concatenated RGB frames
            frame = frame[0:3]  # Take only the first frame
        frame = np.transpose(frame, (1, 2, 0))  # CHW to HWH
        
        # Get ground truth and prediction for current frame
        gt = gt_heatmap[frame_idx].cpu().numpy()
        pred = pred_heatmap[frame_idx].cpu().numpy()
        
        # Reshape ground truth if it's flattened (921600 = 720 * 1280)
        if len(gt.shape) == 1:
            gt = gt.reshape(720, 1280)
        
        # Handle prediction shape
        if len(pred.shape) == 1:
            pred = pred.reshape(720, 1280)
        
        # Plot the three images
        axes[0].imshow(frame)
        axes[0].set_title(f'Frame {frame_idx}')
        axes[0].axis('off')
        
        axes[1].imshow(gt, cmap='hot')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap='hot')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            frame_save_path = save_path.replace('.png', f'_frame{frame_idx}.png')
            plt.savefig(frame_save_path)
        plt.close()









def validate(model, val_loader, device, epoch):
    model.eval()
    losses = []




    criterion = nn.CrossEntropyLoss()
    tp = [0]*3
    tn = [0]*3
    fp = [0]*3
    fn = [0]*3

    
    with torch.no_grad():
        for iter_id, batch in enumerate(val_loader):
            out = model(batch[0].float().to(device))
            gt = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())


            
            # Save first batch visualization
            if iter_id == 0:
                save_path = f'validation_vis_epoch_{epoch}.png'
                visualize_heatmaps(batch[0], batch[1], out, save_path, num_frames=10)
            
            # Ensure shapes are correct for loss calculation
            if out.dim() == 3:  # If output is [batch, channels, pixels]
                out = out.view(out.size(0), out.size(1), -1)  # Reshape to [batch, channels, height*width]
            if gt.dim() == 1:  # If ground truth is [pixels]
                gt = gt.view(-1)  # Reshape to [batch*height*width]
            
            # Calculate metrics



            pred = torch.argmax(out, dim=1)
            

            for i in range(len(pred)):
                x_pred, y_pred = postprocess(pred[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]
                
                if x_pred is not None and y_pred is not None:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < 7:  # min_dist parameter
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                if x_pred is None or y_pred is None:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1
            
            print('val | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} '.format(
                epoch, iter_id, len(val_loader), round(np.mean(losses), 6), sum(tp), sum(tn), sum(fp), sum(fn)))
    
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    recall = sum(tp) / (vc1 + vc2 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    print('precision = {}'.format(precision))
    print('recall = {}'.format(recall))
    print('f1 = {}'.format(f1))
    
    return np.mean(losses), precision, recall, f1


def postprocess(feature_map):
    """Process network output to get ball coordinates"""
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().detach().numpy()
    
    # Print shape for debugging
    #print(f"Feature map shape: {feature_map.shape}")
    
    # Reshape based on the actual dimensions
    if len(feature_map.shape) == 1:
        # If it's a 1D array, calculate height and width
        total_size = feature_map.shape[0]
        height = int(np.sqrt(total_size))
        width = total_size // height
        feature_map = feature_map.reshape((height, width))
    elif len(feature_map.shape) == 3:
        # If it's 3D (channels, height, width), take the first channel
        feature_map = feature_map[0]
    
    # Convert to uint8 for cv2
    feature_map = ((feature_map - feature_map.min()) * (255 / (feature_map.max() - feature_map.min()))).astype(np.uint8)
    
    # Find ball position
    x, y = -1, -1  # Default values if no ball is found
    circles = cv2.HoughCircles(feature_map, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                             param1=50, param2=2, minRadius=2, maxRadius=7)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        if len(circles) > 0:
            x, y = circles[0][0], circles[0][1]
    
    return x, y