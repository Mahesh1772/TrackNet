import torch
from model import BallTrackerNet
from datasets import trackNetDataset
import matplotlib.pyplot as plt
import numpy as np

def visualize_validation_samples():
    # Initialize dataset
    val_dataset = trackNetDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    frames_shown = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, gt_heatmap, x, y, visibility = batch
            
            # Only process frames where visibility is 1
            if visibility[0] == 1:
                # Create figure with subplots
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Get input frame (first frame of sequence)
                frame = inputs[0, 0:3].cpu().numpy()  # Take first RGB frame
                frame = np.transpose(frame, (1, 2, 0))  # CHW to HWH
                
                # Get ground truth
                gt = gt_heatmap[0].cpu().numpy()
                gt = gt.reshape(720, 1280)  # Reshape from flattened
                
                # Plot
                axes[0].imshow(frame)
                axes[0].set_title(f'Frame with Ball (x={x[0]:.1f}, y={y[0]:.1f})')
                axes[0].axis('off')
                
                axes[1].imshow(gt, cmap='hot')
                axes[1].set_title('Ground Truth Heatmap')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'validation_sample_vc1_{frames_shown}.png')
                plt.close()
                
                frames_shown += 1
                
                # Show 5 frames and then stop
                if frames_shown >= 5:
                    break

if __name__ == '__main__':
    visualize_validation_samples() 