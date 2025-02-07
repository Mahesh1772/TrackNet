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
                # Clean up data before saving - set empty string for VC=0 rows
                self.df.loc[self.df['Visibility Class'] == 0, ['X', 'Y', 'Trajectory Pattern']] = ''
                self.df.to_csv(self.csv_path, index=False)  # Auto-save on reaching the end
                break
                
            # Read and display frame
            self.current_img = cv2.imread(frame_path)
            self.display_frame()
            
            # Display frame info
            print(f"\nFrame {self.current_frame:04d}.jpg")
            print("Controls: 0: Free Flight | 1: In Possession | 2: Bounce/Roll | n: Next frame | p: Previous frame | s: Save progress | q: Save and quit")
            
            key = cv2.waitKey(0)
            
            if key == ord('q'):  # Quit
                # Clean up data before saving
                self.df.loc[self.df['Visibility Class'] == 0, ['X', 'Y', 'Trajectory Pattern']] = ''
                self.df.to_csv(self.csv_path, index=False)  # Save on quit
                break
            elif key == ord('s'):  # Save
                # Clean up data before saving
                self.df.loc[self.df['Visibility Class'] == 0, ['X', 'Y', 'Trajectory Pattern']] = ''
                self.df.to_csv(self.csv_path, index=False)
                print("Progress saved!")
                print(self.df)  # Print the entire DataFrame
            elif key == ord('n'):  # Next frame
                self.current_frame += 1
            elif key == ord('p'):  # Previous frame
                self.current_frame = max(0, self.current_frame - 1)
            elif key in [ord('0'), ord('1'), ord('2')]:  # Set Trajectory Pattern
                traj_value = int(chr(key))
                # Only set trajectory value if VC > 0
                if self.df.at[self.current_frame, 'Visibility Class'] > 0:
                    self.df.at[self.current_frame, 'Trajectory Pattern'] = traj_value
                    print(f"Trajectory Pattern set to {traj_value} - " + 
                          ("Free Flight" if traj_value == 0 else 
                           "In Possession" if traj_value == 1 else 
                           "Bounce/Roll"))
                else:
                    print("Cannot set Trajectory Pattern when Visibility Class is 0")
    
    def display_frame(self):
        """Display current frame with ball position and trajectory value"""
        img_copy = self.current_img.copy()
        
        # Display current Trajectory Pattern value
        traj_value = self.df.at[self.current_frame, 'Trajectory Pattern']
        traj_text = f"Trajectory: {traj_value} - " + ("Free Flight" if traj_value == 0 else 
                                                     "In Possession" if traj_value == 1 else 
                                                     "Bounce/Roll" if traj_value == 2 else 
                                                     "Unknown")
        cv2.putText(img_copy, traj_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display ball position if coordinates exist
        x = self.df.at[self.current_frame, 'X']
        y = self.df.at[self.current_frame, 'Y']
        if pd.notna(x) and pd.notna(y):
            # Draw a red circle at ball position
            cv2.circle(img_copy, (int(x), int(y)), 2, (0, 0, 255), -1)
            # Display coordinates
            cv2.putText(img_copy, f'Ball: ({int(x)},{int(y)})', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, img_copy)

def main():
    # Path to the clip you want to label
    clip_path = r'C:\Users\Admin\Documents\Personal_Tracknet\datasets\handball\game1\Clip11'
    
    labeler = TrajectoryLabeler(clip_path)

if __name__ == "__main__":
    main() 