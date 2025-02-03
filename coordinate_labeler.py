import cv2
import os
import pandas as pd
import numpy as np

class BallLabeler:
    def __init__(self, clip_path):
        self.clip_path = clip_path
        self.current_frame = 0
        self.window_name = 'Frame Labeler'
        self.clicked = False  # Track if user clicked in current frame
        
        # Load existing Label.csv if it exists
        self.csv_path = os.path.join(clip_path, 'Label.csv')
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            # Initialize a new DataFrame if no CSV exists
            self.df = pd.DataFrame(columns=['File Name', 'Visibility Class', 'X', 'Y', 'Trajectory Pattern'])
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_event)
        
        self.label_frames()
    
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click
            # Save coordinates temporarily
            self.df.at[self.current_frame, 'X'] = x
            self.df.at[self.current_frame, 'Y'] = y
            self.df.at[self.current_frame, 'Visibility Class'] = 1  # Default to visible
            self.df.at[self.current_frame, 'Trajectory Pattern'] = 0  # Default to normal flight
            self.clicked = True
            print(f"Coordinates saved: ({x}, {y})")
            
            # Draw circle at click position
            self.display_frame()
    
    def save_empty_frame(self):
        """Save frame with no ball"""
        self.df.at[self.current_frame, 'X'] = np.nan  # Use NaN for empty coordinates
        self.df.at[self.current_frame, 'Y'] = np.nan
        self.df.at[self.current_frame, 'Visibility Class'] = 0
        self.df.at[self.current_frame, 'Trajectory Pattern'] = 0
    
    def display_frame(self):
        """Display current frame with saved coordinates"""
        img_copy = self.current_img.copy()
        
        # Draw saved coordinate for current frame if exists
        x = self.df.at[self.current_frame, 'X']
        y = self.df.at[self.current_frame, 'Y']
        if pd.notna(x) and pd.notna(y):
            color = (0, 255, 0) if self.clicked else (255, 0, 0)  # Green if clicked, else blue
            cv2.circle(img_copy, (int(x), int(y)), 5, color, -1)
            cv2.putText(img_copy, f'Current: ({int(x)},{int(y)})', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(self.window_name, img_copy)
    
    def label_frames(self):
        while True:
            # Load frame
            frame_path = os.path.join(self.clip_path, f'{self.current_frame:04d}.jpg')
            if not os.path.exists(frame_path):
                break
                
            # Read and display frame
            self.current_img = cv2.imread(frame_path)
            self.clicked = False  # Reset click status for new frame
            self.display_frame()  # Display frame with saved coordinates
            
            # Display frame info
            print(f"\nFrame {self.current_frame:04d}.jpg")
            print("Controls: Left click to set ball position | n: Next frame | p: Previous frame | s: Save progress | q: Save and quit | e: Mark as empty")
            
            key = cv2.waitKey(0)
            
            if key == ord('q'):  # Quit
                self.df.to_csv(self.csv_path, index=False)  # Save on quit
                break
            elif key == ord('s'):  # Save
                self.df.to_csv(self.csv_path, index=False)
                print("Progress saved!")
                print(self.df)  # Print the entire DataFrame
            elif key == ord('n'):  # Next frame
                # Only save as empty if explicitly marked
                self.current_frame += 1
                self.df.to_csv(self.csv_path, index=False)  # Auto-save on frame change
            elif key == ord('p'):  # Previous frame
                self.current_frame = max(0, self.current_frame - 1)
            elif key == ord('e'):  # Mark as empty
                self.save_empty_frame()
                print(f"Frame {self.current_frame:04d} marked as empty.")
        
        # Save final results
        self.df.to_csv(self.csv_path, index=False)
        cv2.destroyAllWindows()

def main():
    # Path to the clip you want to label
    clip_path = r'C:\Users\Admin\Documents\handball_dataset\game1\Clip4'
    
    labeler = BallLabeler(clip_path)

if __name__ == "__main__":
    main()