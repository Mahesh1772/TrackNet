from model import BallTrackerNet
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance

def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def infer_model(frames, model):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = 720  # Match your training resolution
    width = 1280
    dists = [-1]*2
    ball_track = [(None,None)]*2
    
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():  # Add this for inference
        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (width, height))
            img_prev = cv2.resize(frames[num-1], (width, height))
            img_preprev = cv2.resize(frames[num-2], (width, height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = model(torch.from_numpy(inp).float().to(device), testing=True)  # Add testing=True
            output = out.argmax(dim=1).detach().cpu().numpy()
            
            # Add debug print
            print(f"Output shape: {output.shape}, range: {output.min():.4f} to {output.max():.4f}")
            
            x_pred, y_pred = postprocess(output[0])  # Process first item in batch
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
    return ball_track, dists 

def remove_outliers(positions, dists, max_dist=50):
    """ Remove outliers from model prediction    
    :params
        positions: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        positions: list of ball points
    """
    for i in range(len(positions)-1):  # Changed from len(positions)
        if (dists[i] > max_dist) | (dists[i] == -1):
            positions[i] = None
            
    # Handle the last position separately if needed
    if len(positions) > 0:
        if dists[-1] > max_dist or dists[-1] == -1:
            positions[-1] = None
            
    return positions

def split_track(ball_track):
    # Create list marking frames with detections (0) and without detections (1)
    list_det = [0 if x is not None else 1 for x in ball_track]
    
    # Find sequences of consecutive frames without detections
    gaps = []
    start = None
    for i, val in enumerate(list_det):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            gaps.append((start, i-1))
            start = None
    if start is not None:
        gaps.append((start, len(list_det)-1))
    
    # Split track into subtracks based on gaps
    subtracks = []
    last_end = 0
    for gap_start, gap_end in gaps:
        if gap_start - last_end > 1:  # If there are detections before gap
            subtrack = ball_track[last_end:gap_start]
            subtracks.append((last_end, gap_start))  # Store indices instead of actual subtrack
        last_end = gap_end + 1
    
    # Add final subtrack if there are detections after last gap
    if last_end < len(ball_track):
        if any(x is not None for x in ball_track[last_end:]):  # Only add if contains detections
            subtracks.append((last_end, len(ball_track)))
            
    return subtracks

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nons), y[~nons])

    track = [*zip(x,y)]
    return track

def write_track(frames, ball_track, path_video_out, fps):
    """ Write output video with visualized ball track
    :params
        frames: list of video frames
        ball_track: list of detected ball points
        path_video_out: path to output video file
        fps: frames per second
    """
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_video_out, fourcc, fps, (width, height))
    
    for num, frame in enumerate(frames):
        # Draw current position
        if ball_track[num] is not None:  # Check if detection exists
            x, y = ball_track[num]
            if x is not None and y is not None:  # Check if coordinates exist
                cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)
        
        # Draw track for previous 10 frames
        for i in range(1, 11):
            if num-i >= 0:  # Check if previous frame exists
                if ball_track[num-i] is not None:  # Check if detection exists
                    x, y = ball_track[num-i]
                    if x is not None and y is not None:  # Check if coordinates exist
                        cv2.circle(frame, (int(x), int(y)), 2, (0,0,255), -1)
        
        out.write(frame)
    
    out.release()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--video_out_path', type=str, help='path to output video')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track extrapolation')
    args = parser.parse_args()
    
    model = BallTrackerNet()
    device = 'cuda'
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # After loading model
    print("Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    frames, fps = read_video(args.video_path)
    ball_track, dists = infer_model(frames, model)
    ball_track = remove_outliers(ball_track, dists)    
    
    if args.extrapolation:
        ranges = split_track(ball_track)
        for start, end in ranges:  # Unpack the tuple directly
            ball_subtrack = ball_track[start:end]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[start:end] = ball_subtrack
        
    write_track(frames, ball_track, args.video_out_path, fps)    
    
    
    
    
    
