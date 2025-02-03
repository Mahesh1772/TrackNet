import cv2
import os
import numpy as np
from tqdm import tqdm

def create_dataset_structure(base_path):
    """Create the necessary folder structure for the dataset"""
    # Create base directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Create game folders (game1 through game10)
    for game_id in range(1, 11):
        game_path = os.path.join(base_path, f'game{game_id}')
        os.makedirs(game_path, exist_ok=True)

def extract_frames(input_path, output_path, game_id):
    """Extract frames from video clips and save them in the correct format"""
    # Get all clip folders
    clip_folders = [f for f in os.listdir(input_path) if f.startswith('Clip')]
    
    for clip_folder in clip_folders:
        folder_path = os.path.join(input_path, clip_folder)
        # Get all video files in the clip folder
        video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
        
        for video_file in video_files:
            print(f"Processing {clip_folder}/{video_file}...")
            video_path = os.path.join(folder_path, video_file)
            
            # Extract clip number from folder name
            clip_num = int(clip_folder.replace('Clip', ''))
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue
            
            # Create clip folder directly under game folder
            clip_folder_name = f'Clip{clip_num}'
            clip_path = os.path.join(output_path, f'game{game_id}', clip_folder_name)
            os.makedirs(clip_path, exist_ok=True)
            
            # Initialize frame counter and CSV content
            frame_count = 0
            csv_content = ['File Name,Visibility Class,X,Y,Trajectory Pattern\n']
            
            # Read and save frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Save frame
                    frame_name = f'{frame_count:04d}.jpg'
                    frame_path = os.path.join(clip_path, frame_name)
                    
                    # Resize frame to TrackNet requirements (1280x720)
                    frame = cv2.resize(frame, (1280, 720))
                    cv2.imwrite(frame_path, frame)
                    
                    # Add placeholder entry in CSV
                    csv_content.append(f'{frame_name},0,,,0\n')
                    
                    frame_count += 1
                    pbar.update(1)
            
            # Save Label.csv file in the clip folder
            csv_path = os.path.join(clip_path, 'Label.csv')
            with open(csv_path, 'w') as f:
                f.writelines(csv_content)
            
            cap.release()
            print(f"Processed {frame_count} frames from {video_file}")

def main():
    # Define paths
    input_path = r'C:\Users\Admin\Documents\Gameplay'
    output_path = r'C:\Users\Admin\Documents\handball_dataset'
    
    # Create dataset structure
    create_dataset_structure(output_path)
    
    # Process all clips as game1 (modify if you have multiple games)
    game_id = 1
    extract_frames(input_path, output_path, game_id)
    
    print("\nDataset creation completed!")
    print("\nFolder structure created:")
    print(f"{output_path}/")
    print("├── game1/")
    print("│   ├── Clip1/")
    print("│   │   ├── 0000.jpg")
    print("│   │   ├── 0001.jpg")
    print("│   │   ├── ...")
    print("│   │   └── Label.csv")
    print("│   ├── Clip2/")
    print("│   │   ├── 0000.jpg")
    print("│   │   ├── 0001.jpg")
    print("│   │   ├── ...")
    print("│   │   └── Label.csv")
    print("│   └── ...")
    print("├── game2/")
    print("└── ...")

if __name__ == "__main__":
    main()