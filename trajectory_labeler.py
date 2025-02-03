import cv2
import os
import pandas as pd
import numpy as np

class TrajectoryLabeler:
    def __init__(self, clip_path):
        self.clip_path = clip_path
        self.current_frame = 0
        self.window_name = 'Trajectory Labeler'
        
        # Load existing Label.csv
        self.csv_path = os.path.join(clip_path, 'Label.csv')
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            print("No Label.csv found!")
            return
        
        # Create window
        cv2.namedWindow(self.window_name)
        
        self.label_frames()
    
    def label_frames(self):
        while True:
            # Load frame
            frame_path = os.path.join(self.clip_path, f'{self.current_frame:04d}.jpg')
            if not os.path.exists(frame_path):
                print("Reached the end of frames. Auto-saving progress.")
                self.df.to_csv(self.csv_path, index=False)  # Auto-save on reaching the end
                break
                
            # Read and display frame
            self.current_img = cv2.imread(frame_path)
            self.display_frame()
            
            # Display frame info
            print(f"\nFrame {self.current_frame:04d}.jpg")
            print("Controls: 0: Normal Flight | 1: Caught by Player | 2: Thrown/Shot | n: Next frame | p: Previous frame | s: Save progress | q: Save and quit")
            
            key = cv2.waitKey(0)
            
            if key == ord('q'):  # Quit
                self.df.to_csv(self.csv_path, index=False)  # Save on quit
                break
            elif key == ord('s'):  # Save
                self.df.to_csv(self.csv_path, index=False)
                print("Progress saved!")
                print(self.df)  # Print the entire DataFrame
            elif key == ord('n'):  # Next frame
                self.current_frame += 1
            elif key == ord('p'):  # Previous frame
                self.current_frame = max(0, self.current_frame - 1)
            elif key in [ord('0'), ord('1'), ord('2')]:  # Set Trajectory Pattern
                traj_value = int(chr(key))
                self.df.at[self.current_frame, 'Trajectory Pattern'] = traj_value
                print(f"Trajectory Pattern set to {traj_value}")
    
    def display_frame(self):
        """Display current frame"""
        img_copy = self.current_img.copy()
        
        # Display current Trajectory Pattern value
        traj_value = self.df.at[self.current_frame, 'Trajectory Pattern']
        cv2.putText(img_copy, f'Trajectory: {traj_value}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, img_copy)

def main():
    # Path to the clip you want to label
    clip_path = r'C:\Users\Admin\Documents\handball_dataset\game1\Clip4'
    
    labeler = TrajectoryLabeler(clip_path)

if __name__ == "__main__":
    main() 