import cv2
import numpy as np
import sys
import torch
from sklearn.cluster import KMeans
import time

# Import the necessary components from gpu.py
from src.gpu import create_device, create_engine, create_tensor

def get_grass_color(img, visualize=False):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    
    # Visualization code
    if visualize:
        # Create a small visualization window with fixed size
        viz_size = (400, 300)
        viz_img = cv2.resize(img, (viz_size[0]//2, viz_size[1]//2))
        mask_small = cv2.resize(mask, (viz_size[0]//2, viz_size[1]//2))
        masked_small = cv2.resize(masked_img, (viz_size[0]//2, viz_size[1]//2))
        
        # Create color sample
        color_sample = np.zeros((viz_size[1]//2, viz_size[0]//2, 3), dtype=np.uint8)
        color_sample[:, :] = grass_color[:3]
        
        # Convert mask to BGR for display
        mask_viz = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        
        # Create the visualization layout
        top_row = np.hstack([viz_img, mask_viz])
        bottom_row = np.hstack([masked_small, color_sample])
        full_viz = np.vstack([top_row, bottom_row])
        
        # Add text information
        cv2.putText(full_viz, f"Grass BGR: {tuple(map(int, grass_color[:3]))}", 
                   (10, full_viz.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Display the visualization
        cv2.imshow("Grass Color Detection", full_viz)
        cv2.waitKey(100)  # Display for 100ms
    
    return grass_color[:3]

def get_players_boxes(boxes, classes, scores, frame, conf_threshold=0.6, visualize=False):
    players_imgs = []
    players_boxes = []
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        # Use class 1 for players (based on your updated reference code)
        if cls == 1 and score >= conf_threshold:
            x1, y1, x2, y2 = box
            # Make sure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Only proceed if we have a valid bounding box
            if x2 > x1 and y2 > y1:
                player_img = frame[y1:y2, x1:x2]
                if player_img.size > 0:  # Make sure we have a valid image
                    players_imgs.append(player_img)
                    players_boxes.append(box)
    return players_imgs, players_boxes

def get_ball_box(boxes, classes, scores, conf_threshold=0.5):
    ball_boxes = []
    ball_scores = []
    
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        # Assuming class 0 for the ball (adjust as needed)
        if cls == 0 and score >= conf_threshold:
            x1, y1, x2, y2 = box
            ball_boxes.append(box)
            ball_scores.append(score)
    
    # Return the ball with highest confidence if any is found
    if ball_boxes:
        max_idx = np.argmax(ball_scores)
        return ball_boxes[max_idx]
    
    return None

def get_kits_colors(players, grass_hsv=None, frame=None, visualize=False):
    kits_colors = []
    if grass_hsv is None and frame is not None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    # Create visualization canvas if needed
    if visualize and len(players) > 0:
        viz_width = 800
        viz_height = 400
        viz_img = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 50  # Dark gray background
        
    for i, player_img in enumerate(players):
        try:
            # Skip very small images that might cause problems
            if player_img.shape[0] < 10 or player_img.shape[1] < 10:
                continue
                
            # Convert image to HSV color space
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

            # Define range of green color in HSV
            lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Bitwise-AND mask and original image
            mask = cv2.bitwise_not(mask)
            
            # Focus on upper body for more consistent kit color
            upper_mask = np.zeros(player_img.shape[:2], np.uint8)
            upper_height = max(10, player_img.shape[0] // 3)  # At least 10 pixels or 1/3 of height
            upper_mask[0:upper_height, 0:player_img.shape[1]] = 255
            mask = cv2.bitwise_and(mask, upper_mask)

            # Get mean color of the kit
            kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
            
            # Only add if the color is valid (not black or nearly black)
            if np.mean(kit_color) > 15:  # Simple check to filter out too dark regions
                kits_colors.append(kit_color)
                
                # Add to visualization if requested and within display limit
                if visualize and i < 5:
                    # Calculate position for this player
                    x_offset = i * (viz_width // 5)
                    
                    # Resize player image and mask for display
                    display_size = (120, 160)
                    resized_player = cv2.resize(player_img, display_size)
                    resized_mask = cv2.resize(mask, display_size)
                    
                    # Display original player
                    y_offset = 20
                    h, w = resized_player.shape[:2]
                    viz_img[y_offset:y_offset+h, x_offset+20:x_offset+20+w] = resized_player
                    
                    # Display mask as RGB
                    mask_rgb = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
                    viz_img[y_offset+h+10:y_offset+h+10+h, x_offset+20:x_offset+20+w] = mask_rgb
                    
                    # Display color patch
                    color_patch = np.ones((40, 40, 3), dtype=np.uint8)
                    color_patch[:, :] = kit_color
                    viz_img[y_offset+h*2+20:y_offset+h*2+60, x_offset+60:x_offset+100] = color_patch
                    
                    # Display BGR values
                    cv2.putText(viz_img, f"BGR: {tuple(map(int, kit_color))}", 
                               (x_offset+20, y_offset+h*2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"Error processing player image: {e}")
            continue
    
    # Display visualization
    if visualize and len(kits_colors) > 0:
        cv2.imshow("Kit Color Extraction", viz_img)
        cv2.waitKey(100)  # Display for 100ms
            
    return kits_colors


def get_kits_classifier(kits_colors):
    if len(kits_colors) < 2:
        return None
        
    kits_kmeans = KMeans(n_clusters=2, n_init=10)  # Increased n_init for better clustering
    kits_kmeans.fit(kits_colors)
    
    # Get the cluster centers as team colors
    team_colors = kits_kmeans.cluster_centers_
    
    # Here is where to add the code for minimum color distance validation
    if np.linalg.norm(kits_kmeans.cluster_centers_[0] - kits_kmeans.cluster_centers_[1]) < 30:
        # If clusters too similar, force them to be more distinct
        # Add code to enhance differences
        print("Warning: Team colors too similar, adjusting for better separation")
        # Option 1: Artificially increase the distance between cluster centers
        direction = kits_kmeans.cluster_centers_[1] - kits_kmeans.cluster_centers_[0]
        direction = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize
        kits_kmeans.cluster_centers_[0] -= direction * 15
        kits_kmeans.cluster_centers_[1] += direction * 15
    
    # Print team colors for debugging
    print(f"Team A color (BGR): {team_colors[0]}")
    print(f"Team B color (BGR): {team_colors[1]}")
    
    return kits_kmeans

def classify_kits(kits_classifier, kits_colors):
    if kits_classifier is None:
        return np.zeros(len(kits_colors))
        
    team = kits_classifier.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    if kits_clf is None or len(players_boxes) < 2 or len(kits_colors) < 2:
        return 0
        
    left_team_label = 0
    team_0 = []
    team_1 = []

    # Process up to min(len(players_boxes), len(kits_colors)) to avoid index errors
    max_idx = min(len(players_boxes), len(kits_colors))
    
    for i in range(max_idx):
        x1, y1, x2, y2 = players_boxes[i]
        
        try:
            team = classify_kits(kits_clf, [kits_colors[i]]).item()
            if team == 0:
                team_0.append(np.array([x1]))
            else:
                team_1.append(np.array([x1]))
        except Exception as e:
            print(f"Error classifying player {i}: {e}")
            continue

    if len(team_0) > 0 and len(team_1) > 0:
        team_0 = np.array(team_0)
        team_1 = np.array(team_1)

        if np.average(team_0) - np.average(team_1) > 0:
            left_team_label = 1

    return left_team_label

def resize_frame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (800, 600)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def process_detections(output, orig_shape):
    num_dets, bboxes, scores, labels = output
    num_dets = int(num_dets[0])
    bboxes = bboxes[0, :num_dets]
    scores = scores[0, :num_dets]
    labels = labels[0, :num_dets]
    orig_h, orig_w = orig_shape[:2]
    boxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Scale coordinates to original frame dimensions
        x1 = int(x1 * orig_w / W)
        y1 = int(y1 * orig_h / H)
        x2 = int(x2 * orig_w / W)
        y2 = int(y2 * orig_h / H)
        boxes.append([x1, y1, x2, y2])
    
    return np.array(boxes), scores.cpu().numpy(), labels.cpu().numpy()

def annotate_video(video_path, engine_id="1280", gpu_id=0):
    # Initialize GPU device and load engine
    device = create_device(gpu_id)
    engines = []
    global W, H
    W, H = create_engine(device, engines, engine_id)
    engine = engines[0]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Set output video dimensions
    height = 896
    width = 2300

    # Setup output video
    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('./output/'+video_name.split('.')[0] + "_out.mp4",
                                   fourcc,
                                   30.0,
                                   (width, height))
    
    # Variables to maintain state across frames
    kits_clf = None
    left_team_label = None
    grass_hsv = None
    team_colors = None
    
    # Confidence thresholds
    player_conf_threshold = 0.6  # Higher threshold for player detection
    ball_conf_threshold = 0.5    # Lower threshold for ball detection
    
    # Initialize frame counter
    current_frame_idx = 0
    
    # Storage for kit colors to improve consistency
    accumulated_kit_colors = []
    
    # Define label names for displaying
    label_names = {
        0: "Player-L",  # Left team player
        1: "Player-R",  # Right team player
        2: "Ball"       # Ball
    }
    
    # Process video frame by frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        current_frame_idx += 1
        
        # Process frame
        annotated_frame = cv2.resize(frame.copy(), (width, height))
        
        # Convert to GPU tensor
        tensor = create_tensor(annotated_frame, device, W, H)
        if tensor is None:
            continue
            
        # Run inference
        output = engine(tensor)
        
        # Post-processing
        boxes, scores, classes = process_detections(output, annotated_frame.shape)
        
        # Get player images and boxes with higher confidence threshold
        players_imgs, players_boxes = get_players_boxes(boxes, classes, scores, annotated_frame, player_conf_threshold, visualize=False)
        
        # Get kit colors for detected players
        if len(players_imgs) > 0:
            frame_kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)
            
            # Wait until we have enough players for reliable team classification
            if len(frame_kits_colors) >= 4:
                accumulated_kit_colors.extend(frame_kits_colors)
                if len(accumulated_kit_colors) > 100:
                    accumulated_kit_colors = accumulated_kit_colors[-100:]
                
                # Initialize team classification only once with sufficient data
                if kits_clf is None and len(accumulated_kit_colors) >= 30:
                    print(f"Initializing team classification with {len(accumulated_kit_colors)} kit colors")
                    kits_clf = get_kits_classifier(accumulated_kit_colors)
                    if kits_clf is not None:
                        left_team_label = get_left_team_label(players_boxes, frame_kits_colors, kits_clf)
                        grass_color = get_grass_color(annotated_frame)
                        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
                        print(f"Left team label: {left_team_label}")
                        # Save team colors for future reference
                        team_colors = kits_clf.cluster_centers_
            
            # Detect ball (class 0) with its own threshold
            ball_box = get_ball_box(boxes, classes, scores, ball_conf_threshold)
            
            # Draw all detections on the frame
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                x1, y1, x2, y2 = box
                display_label = None
                
                # Handle player detection (class 1)
                if cls == 1 and score >= player_conf_threshold:
                    if kits_clf is not None and left_team_label is not None:
                        # Get kit color for this player
                        player_img = annotated_frame[y1:y2, x1:x2]
                        kit_colors = get_kits_colors([player_img], grass_hsv)
                        
                        if len(kit_colors) > 0:
                            team = classify_kits(kits_clf, kit_colors).item()
                            if team == left_team_label:
                                label_key = 0  # Player-L
                            else:
                                label_key = 1  # Player-R
                        else:
                            # Default assignment if kit color extraction fails
                            if x1 < width / 2:
                                label_key = 0  # Player-L
                            else:
                                label_key = 1  # Player-R
                    else:
                        # Before team classification is established
                        label_key = 0  # Default to Player-L
                    
                    # Make sure the key exists in our box_colors dictionary
                    display_label = label_names.get(label_key, "Player")
                    label_key = str(label_key)
                
                # Handle ball detection (class 0)
                elif cls == 0 and score >= ball_conf_threshold:
                    display_label = label_names.get(2, "Ball")
                    label_key = "2"
                
                # Draw detection if we have a valid label_key and display_label
                if display_label is not None and label_key in box_colors:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[label_key], 2)
                    cv2.putText(annotated_frame, f"{display_label} {score:.2f}", (x1 - 10, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_colors[label_key], 2)
            
            # Display and save video
            resized_frame = resize_frame(annotated_frame, scale=0.75)
            cv2.imshow('Video', resized_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            output_video.write(annotated_frame)
    
    # Cleanup
    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

if __name__ == "__main__":
    # Box colors for different objects
    box_colors = {
        "0": (150, 50, 50),     # Team 1 (Left) - Red
        "1": (37, 47, 150),     # Team 2 (Right) - Blue
        "2": (155, 62, 157),    # Ball - Purple
    }
    
    # Input video file path
    video_path = "video_clip.mp4"
    
    # Select engine ID (1280 or 960)
    engine_id = "1280"  # Use high-resolution model
    
    # Select GPU ID
    gpu_id = 0  # Use first GPU
    
    # Run video annotation
    annotate_video(video_path, engine_id, gpu_id)