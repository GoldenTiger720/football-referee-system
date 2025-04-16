import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans

class TeamIdentifier:
    """
    A class for identifying teams and players in football/soccer images.
    This class handles the detection and classification of players into different teams
    based on their kit colors.
    """
    
    def __init__(self, engine_model_path=None, device_id=0):
        """
        Initialize the TeamIdentifier with optional model path and device ID.
        
        Args:
            engine_model_path (str, optional): Path to the TensorRT engine model file.
                If None, will attempt to use a default model.
            device_id (int, optional): CUDA device ID to use. Defaults to 0.
        """
        self.device = None
        self.engine = None
        self.W = None
        self.H = None
        
        # State variables
        self.kits_clf = None
        self.left_team_label = None
        self.grass_hsv = None
        self.team_colors = None
        self.accumulated_kit_colors = []
        
        # Detection thresholds
        self.player_conf_threshold = 0.6
        self.ball_conf_threshold = 0.5
        
        # Box colors for different objects
        self.box_colors = {
            "0": (150, 50, 50),     # Team 1 (Left) - Red
            "1": (37, 47, 150),     # Team 2 (Right) - Blue
            "2": (155, 62, 157),    # Ball - Purple
        }
        
        # Label names
        self.label_names = {
            0: "Player-L",  # Left team player
            1: "Player-R",  # Right team player
            2: "Ball"       # Ball
        }
        
        # Initialize GPU if model path is provided
        if engine_model_path:
            self.initialize_model(engine_model_path, device_id)
    
    def initialize_model(self, engine_model_path, device_id=0):
        """
        Initialize the TensorRT model.
        
        Args:
            engine_model_path (str): Path to the TensorRT engine model file.
            device_id (int, optional): CUDA device ID to use. Defaults to 0.
            
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Try to import the necessary modules for TensorRT
            from models import TRTModule  # isort:skip
            from models.torch_utils import det_postprocess
            from models.utils import blob, letterbox, path_to_list
            
            # Initialize the device
            self.device = torch.device(f"cuda:{device_id}")
            
            # Load the engine
            self.engine = TRTModule(engine_model_path, self.device)
            self.H, self.W = self.engine.inp_info[0].shape[-2:]
            self.engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
            
            print(f"Engine loaded: Actual input shape {self.H}x{self.W}")
            return True
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
        
    def initialize_teams(self, player_samples):
        """
        Initialize team classification from player samples
        
        Args:
            player_samples: Dictionary of player_id -> (image, keypoints) samples
        """
        # Extract kit colors from player samples
        kit_colors = []
        for player_id, (player_img, _) in player_samples.items():
            if player_img is not None and player_img.size > 0:
                # Get kit colors for this player
                player_kit_colors = self.get_kits_colors([player_img])
                if player_kit_colors:
                    kit_colors.extend(player_kit_colors)
        
        # Initialize the classifier if we have enough colors
        if len(kit_colors) >= 2:
            self.kits_clf = self.get_kits_classifier(kit_colors)
            
            # Store team colors for future reference
            if self.kits_clf is not None:
                self.team_colors = self.kits_clf.cluster_centers_
                print(f"Team A color (BGR): {self.team_colors[0]}")
                print(f"Team B color (BGR): {self.team_colors[1]}")
                return True
                
        print(f"Warning: Not enough kit colors ({len(kit_colors)}) to initialize team classifier")
        return False

    def get_team_for_player(self, player_id, player_img):
        """
        Get the team ID for a player
        
        Args:
            player_id: Unique ID for the player
            player_img: Image of the player
            
        Returns:
            team_id: 0 or 1 for team, -1 if classification failed
        """
        # Check if the input is valid
        if player_img is None or player_img.size == 0:
            return -1
            
        try:
            # Get kit colors for this player
            kit_colors = self.get_kits_colors([player_img])
            
            # Classify if we have colors and a classifier
            if kit_colors and len(kit_colors) > 0 and self.kits_clf is not None:
                team = self.classify_kits(self.kits_clf, kit_colors).item()
                return int(team)
        except Exception as e:
            print(f"Error processing player image: {e}")
        
        return -1
    
    def get_display_color_for_player(self, player_id):
        """
        Get the display color for a player based on their team
        
        Args:
            player_id: Unique ID for the player
            
        Returns:
            color: BGR color tuple
        """
        # You can add a player-team mapping dictionary if needed
        # For now, just return default colors
        
        # If we have team colors from the KMeans classifier
        if hasattr(self, 'team_colors') and self.team_colors is not None:
            # Return the actual team colors (BGR format)
            print("=========================")
            team_id = 0  # Default to team 0
            if hasattr(self, 'player_team_map') and player_id in self.player_team_map:
                team_id = self.player_team_map[player_id]

            print("color =====> ", team_id)
            
            return tuple(map(int, self.team_colors[team_id]))
        
        # Default colors if team colors aren't available
        team_colors = {
            0: (150, 50, 50),   # Team 0 (red)
        1: (37, 47, 150),   # Team 1 (blue)
        }
    
        # Default to team 0 (red)
        return team_colors[0]
    
    def get_grass_color(self, img, visualize=False):
        """
        Finds the color of the grass in the background of the image.
        
        Args:
            img: np.array object of shape (WxHx3) that represents the BGR value of the frame pixels.
            visualize: Boolean flag to enable visualization of the grass detection.
            
        Returns:
            grass_color: Tuple of the BGR value of the grass color in the image
        """
        # Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range of green color in HSV
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate the mean value of the pixels that are not masked
        grass_color = cv2.mean(img, mask=mask)
        
        # Visualization code
        if visualize:
            # Create a small visualization window with fixed size
            viz_size = (400, 300)
            viz_img = cv2.resize(img, (viz_size[0]//2, viz_size[1]//2))
            mask_small = cv2.resize(mask, (viz_size[0]//2, viz_size[1]//2))
            masked_img = cv2.bitwise_and(img, img, mask=mask)
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
    
    def create_tensor(self, frame):
        """
        Create a tensor from an image frame for model input.
        
        Args:
            frame: The input image frame.
            
        Returns:
            tensor: The tensor for model input.
        """
        try:
            from models.utils import blob, letterbox
            
            # Preprocess the image
            bgr, ratio, dwdh = letterbox(frame, (self.W, self.H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
            tensor = torch.asarray(tensor, device=self.device)
            
            return tensor
        except Exception as e:
            print(f"Error creating tensor: {e}")
            return None
    
    def process_detections(self, output, orig_shape):
        """
        Process the raw detection outputs from the TensorRT model.
        
        Args:
            output: Raw outputs from TensorRT model (num_dets, bboxes, scores, labels).
            orig_shape: Original shape of the frame.
            
        Returns:
            boxes: Array of bounding boxes [x1, y1, x2, y2].
            scores: Array of confidence scores.
            classes: Array of class ids.
        """
        # Unpack the output
        num_dets, bboxes, scores, labels = output
        
        # Get the number of detections in the first batch
        num_dets = int(num_dets[0])
        
        # Extract valid detections
        bboxes = bboxes[0, :num_dets]
        scores = scores[0, :num_dets]
        labels = labels[0, :num_dets]
        
        # Convert boxes to [x1, y1, x2, y2] format and scale to original image dimensions
        orig_h, orig_w = orig_shape[:2]
        boxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Scale coordinates to original frame dimensions
            x1 = int(x1 * orig_w / self.W)
            y1 = int(y1 * orig_h / self.H)
            x2 = int(x2 * orig_w / self.W)
            y2 = int(y2 * orig_h / self.H)
            boxes.append([x1, y1, x2, y2])
        
        return np.array(boxes), scores.cpu().numpy(), labels.cpu().numpy()
    
    def get_players_boxes(self, boxes, classes, scores, frame, visualize=False):
        """
        Finds the images of the players in the frame and their bounding boxes.
        
        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2].
            classes: Array of class ids.
            scores: Array of confidence scores.
            frame: Original video frame.
            visualize: Boolean flag to enable visualization.
            
        Returns:
            players_imgs: List of np.array objects that contain the BGR values of the
                         cropped parts of the image that contains players.
            players_boxes: List of bounding boxes for players.
        """
        players_imgs = []
        players_boxes = []
        
        # Create visualization canvas if needed
        if visualize:
            viz_img = frame.copy()
            detection_info = []
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            # Use class 1 for players (based on your updated reference code)
            if cls == 1 and score >= self.player_conf_threshold:
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
                        
                        # Add to visualization
                        if visualize:
                            # Draw bounding box
                            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label with score
                            label = f"Player {i+1}: {score:.2f}"
                            cv2.putText(viz_img, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Save information for player crops display
                            detection_info.append((player_img, i+1, score))
        
        # Display visualization with player crops
        if visualize and len(players_imgs) > 0:
            # Create a separate window for player crops
            panel_height = 200
            panel_width = 1200
            panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 50  # Dark gray background
            
            # Display up to 6 player crops
            max_display = min(6, len(detection_info))
            crop_width = panel_width // max_display
            
            for i in range(max_display):
                player_crop, idx, score = detection_info[i]
                
                # Resize crop to fit panel
                crop_height = min(150, panel_height - 40)
                crop_w = int(player_crop.shape[1] * (crop_height / player_crop.shape[0]))
                crop_w = min(crop_w, crop_width - 20)
                
                if crop_w > 0 and crop_height > 0:
                    resized_crop = cv2.resize(player_crop, (crop_w, crop_height))
                    
                    # Calculate position
                    x_offset = i * crop_width + (crop_width - crop_w) // 2
                    
                    # Place crop in panel
                    y_offset = 10
                    h, w = resized_crop.shape[:2]
                    panel[y_offset:y_offset+h, x_offset:x_offset+w] = resized_crop
                    
                    # Add info text
                    cv2.putText(panel, f"Player {idx}: {score:.2f}", 
                              (x_offset, y_offset+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display both windows separately
            cv2.imshow("Player Detections Frame", cv2.resize(viz_img, (1200, 600)))
            cv2.imshow("Player Crops", panel)
            cv2.waitKey(100)  # Display for 100ms
        
        return players_imgs, players_boxes
    
    def get_ball_box(self, boxes, classes, scores):
        """
        Finds the ball in the frame.
        
        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2].
            classes: Array of class ids.
            scores: Array of confidence scores.
            
        Returns:
            ball_box: Bounding box of the ball, or None if no ball is detected.
        """
        ball_boxes = []
        ball_scores = []
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            # Assuming class 0 for the ball (adjust as needed)
            if cls == 0 and score >= self.ball_conf_threshold:
                ball_boxes.append(box)
                ball_scores.append(score)
        
        # Return the ball with highest confidence if any is found
        if ball_boxes:
            max_idx = np.argmax(ball_scores)
            return ball_boxes[max_idx]
        
        return None
    
    def get_kits_colors(self, players, frame=None, visualize=False):
        """
        Finds the kit colors of all the players in the current frame.
        
        Args:
            players: List of np.array objects that contain the BGR values of the image
                    portions that contain players.
            frame: Original video frame.
            visualize: Boolean flag to enable visualization of kit color extraction.
            
        Returns:
            kits_colors: List of np arrays that contain the BGR values of the kits color of all
                        the players in the current frame.
        """
        kits_colors = []
        
        # Get grass color if we don't have it yet and frame is provided
        if self.grass_hsv is None and frame is not None:
            try:
                grass_color = self.get_grass_color(frame)
                self.grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
            except Exception as e:
                print(f"Error getting grass color: {e}")
                # Default grass HSV values if extraction fails
                self.grass_hsv = np.array([[[60, 100, 100]]], dtype=np.uint8)

        # Create visualization canvas if needed
        if visualize and len(players) > 0:
            viz_width = 800
            viz_height = 400
            viz_img = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 50  # Dark gray background
            
        for i, player_img in enumerate(players):
            try:
                # Check if player image is valid
                if player_img is None or not isinstance(player_img, np.ndarray):
                    continue
                    
                # Skip very small images that might cause problems
                if player_img.shape[0] < 10 or player_img.shape[1] < 10:
                    continue
                    
                # Convert image to HSV color space
                hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

                # Define range of green color in HSV
                # Use default values if grass_hsv is None
                if self.grass_hsv is None:
                    lower_green = np.array([40, 40, 40])
                    upper_green = np.array([80, 255, 255])
                else:
                    lower_green = np.array([self.grass_hsv[0, 0, 0] - 10, 40, 40])
                    upper_green = np.array([self.grass_hsv[0, 0, 0] + 10, 255, 255])

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
                
                # Visualization code remains unchanged...
                
            except Exception as e:
                print(f"Error processing player image: {e}")
                continue
        
        return kits_colors

    def get_kits_classifier(self, kits_colors):
        """
        Creates a KMeans classifier for team kit colors.
        
        Args:
            kits_colors: List of kit colors to classify.
            
        Returns:
            kits_kmeans: KMeans classifier for team classification.
        """
        if len(kits_colors) < 2:
            return None
            
        kits_kmeans = KMeans(n_clusters=2, n_init=10)  # Increased n_init for better clustering
        kits_kmeans.fit(kits_colors)
        
        # Get the cluster centers as team colors
        team_colors = kits_kmeans.cluster_centers_
        
        # Add minimum color distance validation
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
    
    def classify_kits(self, kits_classifier, kits_colors):
        """
        Classifies the player into one of the two teams according to the player's kit color.
        
        Args:
            kits_classifier: sklearn.cluster.KMeans object that can classify the
                            players kits into 2 teams according to their color.
            kits_colors: List of np.array objects that contain the BGR values of
                        the colors of the kits of the players found in the current frame.
                        
        Returns:
            team: np.array object containing a single integer that carries the player's
                 team number (0 or 1).
        """
        if kits_classifier is None:
            return np.zeros(len(kits_colors))
            
        team = kits_classifier.predict(kits_colors)
        return team
    
    def get_left_team_label(self, players_boxes, kits_colors, kits_clf):
        """
        Finds the label of the team that is on the left of the screen.
        
        Args:
            players_boxes: List of bounding boxes for players.
            kits_colors: List of np.array objects that contain the BGR values of
                        the colors of the kits of the players found in the current frame.
            kits_clf: sklearn.cluster.KMeans object that can classify the players kits
                    into 2 teams according to their color.
                    
        Returns:
            left_team_label: Int that holds the number of the team that's on the left of the image
                           either (0 or 1).
        """
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
                team = self.classify_kits(kits_clf, [kits_colors[i]]).item()
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
    
    def process_frame(self, frame, visualize=False):
        """
        Process a single frame to detect and classify players and ball.
        
        Args:
            frame: Input frame to process.
            visualize: Whether to display visualizations.
            
        Returns:
            annotated_frame: Frame with annotations.
            detections: Dict with detected objects information.
        """
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Ensure we have a model loaded
        if self.engine is None:
            print("Warning: No model loaded, cannot process frame")
            return annotated_frame, {}
        
        # Prepare frame for inference
        tensor = self.create_tensor(annotated_frame)
        if tensor is None:
            return annotated_frame, {}
        
        # Run inference
        output = self.engine(tensor)
        
        # Post-processing
        boxes, scores, classes = self.process_detections(output, annotated_frame.shape)
        
        # Get player images and boxes
        players_imgs, players_boxes = self.get_players_boxes(boxes, classes, scores, annotated_frame, visualize)
        
        # Detect ball
        ball_box = self.get_ball_box(boxes, classes, scores)
        
        # Process kit colors if we have players
        if len(players_imgs) > 0:
            frame_kits_colors = self.get_kits_colors(players_imgs, annotated_frame, visualize)
            
            # Accumulate kit colors for more stable team classification
            if len(frame_kits_colors) > 0:
                self.accumulated_kit_colors.extend(frame_kits_colors)
                
                # Keep the accumulated kit colors list from growing too large
                if len(self.accumulated_kit_colors) > 100:
                    self.accumulated_kit_colors = self.accumulated_kit_colors[-100:]
                
                # Initialize team classification if we have enough data
                if self.kits_clf is None and len(self.accumulated_kit_colors) >= 30:
                    print(f"Initializing team classification with {len(self.accumulated_kit_colors)} kit colors")
                    self.kits_clf = self.get_kits_classifier(self.accumulated_kit_colors)
                    if self.kits_clf is not None:
                        self.left_team_label = self.get_left_team_label(players_boxes, frame_kits_colors, self.kits_clf)
                        grass_color = self.get_grass_color(annotated_frame)
                        self.grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
                        print(f"Left team label: {self.left_team_label}")
                        # Save team colors for future reference
                        self.team_colors = self.kits_clf.cluster_centers_
        
        # Store detections
        detections = {
            'players': [],
            'ball': None
        }
        
        # Draw all detections on the frame
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            x1, y1, x2, y2 = box
            display_label = None
            team_id = None
            
            # Handle player detection (class 1)
            if cls == 1 and score >= self.player_conf_threshold:
                player_data = {
                    'box': box.tolist(),
                    'score': float(score),
                    'team': None
                }
                
                if self.kits_clf is not None and self.left_team_label is not None:
                    # Get kit color for this player
                    player_img = annotated_frame[y1:y2, x1:x2]
                    kit_colors = self.get_kits_colors([player_img], None)
                    
                    if len(kit_colors) > 0:
                        team = self.classify_kits(self.kits_clf, kit_colors).item()
                        team_id = int(team)
                        player_data['team'] = team_id
                        
                        if team == self.left_team_label:
                            label_key = 0  # Player-L
                        else:
                            label_key = 1  # Player-R
                    else:
                        # Default assignment if kit color extraction fails
                        if x1 < annotated_frame.shape[1] / 2:
                            label_key = 0  # Player-L
                            team_id = self.left_team_label
                        else:
                            label_key = 1  # Player-R
                            team_id = 1 - self.left_team_label
                        
                        player_data['team'] = team_id
                else:
                    # Before team classification is established
                    label_key = 0  # Default to Player-L
                
                detections['players'].append(player_data)
                
                # Make sure the key exists in our box_colors dictionary
                display_label = self.label_names.get(label_key, "Player")
                label_key = str(label_key)
            
            # Handle ball detection (class 0)
            elif cls == 0 and score >= self.ball_conf_threshold:
                display_label = self.label_names.get(2, "Ball")
                label_key = "2"
                
                detections['ball'] = {
                    'box': box.tolist(),
                    'score': float(score)
                }
            
            # Draw detection if we have a valid label_key and display_label
            if display_label is not None and label_key in self.box_colors:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.box_colors[label_key], 2)
                cv2.putText(annotated_frame, f"{display_label} {score:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.box_colors[label_key], 2)
        
        # Return the annotated frame and detections
        return annotated_frame, detections

# Create a simplified interface for external systems
def process_image(image, model_path=None, device_id=0, visualize=False):
    """
    Process an image to identify teams and players.
    
    Args:
        image: Input image as numpy array (BGR format).
        model_path: Path to the TensorRT engine model file.
        device_id: CUDA device ID to use.
        visualize: Whether to display visualizations.
        
    Returns:
        annotated_image: Image with annotations.
        detections: Dict with detected objects information.
    """
    # Create a TeamIdentifier instance
    identifier = TeamIdentifier(model_path, device_id)
    
    # Process the image
    return identifier.process_frame(image, visualize)

# Function to create a persistent identifier for processing multiple frames
def create_team_identifier(model_path=None, device_id=0):
    """
    Create a TeamIdentifier instance for processing multiple frames.
    
    Args:
        model_path: Path to the TensorRT engine model file.
        device_id: CUDA device ID to use.
        
    Returns:
        TeamIdentifier: An instance of TeamIdentifier that can be used for processing multiple frames.
    """
    return TeamIdentifier(model_path, device_id)


# Sample usage for processing a video
def process_video(video_path, output_path=None, model_path=None, device_id=0):
    """
    Process a video file to identify teams and players.
    
    Args:
        video_path: Path to input video file.
        output_path: Path to save output video (optional).
        model_path: Path to the TensorRT engine model file.
        device_id: CUDA device ID to use.
        
    Returns:
        None: Displays processed video and optionally saves to output_path.
    """
    # Create a TeamIdentifier instance
    identifier = TeamIdentifier(model_path, device_id)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        annotated_frame, detections = identifier.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Processed Video', annotated_frame)
        
        # Write to output file if specified
        if out:
            out.write(annotated_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage with an image file
    import argparse
    
    parser = argparse.ArgumentParser(description='Team Identification for Soccer/Football')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--output', type=str, help='Path to output image or video (optional)')
    parser.add_argument('--model', type=str, help='Path to TensorRT engine model file (optional)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    
    args = parser.parse_args()
    
    if args.input:
        if args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process video
            process_video(args.input, args.output, args.model, args.device)
        elif args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Process single image
            image = cv2.imread(args.input)
            annotated_image, detections = process_image(image, args.model, args.device, args.visualize)
            
            # Display the result
            cv2.imshow('Processed Image', annotated_image)
            cv2.waitKey(0)
            
            # Save the result if output path is specified
            if args.output:
                cv2.imwrite(args.output, annotated_image)
                print(f"Output saved to {args.output}")
            
            cv2.destroyAllWindows()
        else:
            print(f"Unsupported file format: {args.input}")
    else:
        print("Please provide an input file path with --input")