import time
import math
import logging
import numpy as np
import cv2
from src.gpu import *
from src.detector_utils import *
from src.camera import *
from src.aisettings import *
from src.ballradiustracker import *
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans

CLASSID_BALL = 0
CLASSID_FASTBALL = 1
CLASSID_PLAYER = 1
ballRadiusChecker = BallRadiusTracker()

# Team color mapping (BGR format in OpenCV)
TEAM_COLORS = {
    0: (255, 0, 0),    # Blue team 
    1: (0, 0, 255),    # Red team
    -1: (0, 255, 0)    # Unknown/Referee (Green)
}

class AIDetector:
    def __init__(self, ai_settings, cameras, paralell_models, max_fps) -> None:
        logging.basicConfig(level=logging.DEBUG, filename='detector.log', filemode='w',
                    format='%(message)s')        
        self.cameras = cameras
        self.ai_settings = ai_settings
        self.max_fps = max_fps
        self.paralell_models = paralell_models
        self.engines_960 = []
        self.engines_1280 = []
        self.goal_engines = []
        self.futures = []
        self.completed_futures = set()
        self.W_960 = 0
        self.H_960 = 0
        self.W_1280 = 0
        self.H_1280 = 0

        self.ball_color_total = defaultdict(int)
        self.ball_color_counter = defaultdict(int)

        self.goal_W = 0
        self.goal_H = 0
        self.goal_l_reference_taken = False
        self.goal_r_reference_taken = False
        self.false_detection = 0        
        self.best_camera = -1
        self.last_best_camera = -1
        self.last_best_camera_obj = None
        self.last_proc_time = 0
        self.second_best_camera = -1
        self.detector_cycle = 0
        self.current_stage = 1
        self.executor = ThreadPoolExecutor(max_workers=self.paralell_models)
        self.outstanding_detections = 0
        self.last_outstanding_detections = 0
        self.device = create_device(0)

        self.stage_start_time = 0
        self.last_left_goalkeeper = None
        self.broadcast_camera = -1

        # Team classification related attributes
        self.kits_clf = None
        self.left_team_label = None
        self.grass_hsv = None
        self.team_colors = None
        self.current_frame_idx = 0
        self.accumulated_kit_colors = []
        
        # Initialize the classifier as None - will be populated once we have enough data
        self.labels_name = {
            0: "Player-L",  # Left team player
            1: "Player-R",  # Right team player
            2: "Ball"       # Ball
        }

        for m_id in range(self.paralell_models):
            self.W_1280, self.H_1280 = create_engine(0, self.engines_1280, "1280")

    # Team classification methods
    def get_grass_color(self, img):
        """Extract the average color of the grass from the image"""
        # Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        cv2.imshow("player", mask)

        # Calculate the mean value of the pixels that are not masked
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        grass_color = cv2.mean(img, mask=mask)
        
        return grass_color[:3]

    def get_kits_colors(self, players_imgs, grass_hsv=None, frame=None):
        """Extract kit colors from player images, filtering out grass colors"""
        kits_colors = []
        if grass_hsv is None and frame is not None:
            grass_color = self.get_grass_color(frame)
            grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

        for player_img in players_imgs:
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
                    
            except Exception as e:
                print(f"Error processing player image: {e}")
                continue
                
        return kits_colors

    def get_kits_classifier(self, kits_colors):
        """Create a KMeans classifier for team colors"""
        if len(kits_colors) < 2:
            return None
            
        kits_kmeans = KMeans(n_clusters=2, n_init=10)  # Increased n_init for better clustering
        kits_kmeans.fit(kits_colors)
        
        # Get the cluster centers as team colors
        team_colors = kits_kmeans.cluster_centers_
        
        # Check if clusters are too similar
        if np.linalg.norm(kits_kmeans.cluster_centers_[0] - kits_kmeans.cluster_centers_[1]) < 30:
            # If clusters too similar, force them to be more distinct
            print("Warning: Team colors too similar, adjusting for better separation")
            direction = kits_kmeans.cluster_centers_[1] - kits_kmeans.cluster_centers_[0]
            direction = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize to avoid div by zero
            kits_kmeans.cluster_centers_[0] -= direction * 15
            kits_kmeans.cluster_centers_[1] += direction * 15
        
        # Print team colors for debugging
        print(f"Team A color (BGR): {team_colors[0]}")
        print(f"Team B color (BGR): {team_colors[1]}")
        
        return kits_kmeans

    def classify_kits(self, kits_colors):
        """Classify kit colors into teams using the KMeans classifier"""
        if self.kits_clf is None:
            return [-1] * len(kits_colors)
            
        team = self.kits_clf.predict(kits_colors)
        return team

    def get_left_team_label(self, players_boxes, kits_colors):
        """Determine which team label corresponds to the left side of the field"""
        if self.kits_clf is None or len(players_boxes) < 2 or len(kits_colors) < 2:
            return 0
            
        left_team_label = 0
        team_0 = []
        team_1 = []

        # Process up to min(len(players_boxes), len(kits_colors)) to avoid index errors
        max_idx = min(len(players_boxes), len(kits_colors))
        
        for i in range(max_idx):
            try:
                x1, y1, x2, y2 = players_boxes[i]
                
                team = self.classify_kits([kits_colors[i]]).item()
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

    def start_cycle(self, detector_cycle):
        """Start a new detection cycle"""
        self.detector_cycle = detector_cycle

    def get_number_of_outstanding_detections(self):
        """Get the number of detections that are still pending processing"""
        self.outstanding_detections = 0
        for camera in self.cameras:
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None and obj.skip == False and obj.processed == False:
                    self.outstanding_detections += 1
        return self.outstanding_detections

    def run_detection(self):
        """Submit detection tasks to the thread pool for parallel processing"""
        engine_position = 0
        start = time.monotonic_ns()
        for camera in self.cameras:
            if (camera.capture_only_device == True):
                continue
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for det_obj in detector_objs:
                if det_obj is not None and det_obj.skip == False and det_obj.processed == False and det_obj.in_progress == False:
                    det_obj.in_progress = True
                    engine = self.engines_960
                    engine_w = self.W_960
                    engine_h = self.H_960
                    if camera.frame_w == 1280 or camera.frame_w == 2300:
                        engine = self.engines_1280
                        engine_w = self.W_1280
                        engine_h = self.H_1280
                    self.futures.append(self.executor.submit(self.process_frame, camera, self.detector_cycle, det_obj, engine[engine_position], engine_w, engine_h))
                    engine_position += 1
                    if (engine_position >= self.paralell_models):
                        engine_position = 0
        elapsed = (time.monotonic_ns() - start) / 1000 / 1000
        if (elapsed > 2):
            print("run_detection process time:", elapsed)
    
    def process(self, current_detector_cycle):
        """Main processing loop for the detector"""
        if (current_detector_cycle <= self.detector_cycle and self.current_stage == 1):
            return
        self.process_results_non_blocking()
        if self.futures:
            return

        self.get_number_of_outstanding_detections()
        if self.outstanding_detections > 0:
            return

        if self.current_stage == 1:
            self.stage_start_time = time.monotonic_ns()
            self.run_stage_1(number_of_frames=self.ai_settings.detection_first_stage_frames)
            self.current_stage = 2
            return
        
        if self.current_stage == 2:
            self.make_camera_selection()
            return
        
        if self.current_stage == 3:
            if (self.best_camera != -1):
                processed_cntr = 0
                ball_on = 0
                for cam in self.cameras:
                    if (cam.capture_only_device == True):
                        continue                    
                    if cam.camera_id == self.best_camera:
                        detector_objs = cam.get_detector_collection(self.detector_cycle)
                        for obj in detector_objs:
                            if obj is not None:
                                if obj.ball > 0:
                                    ball_on += 1
                                if obj.processed == True:
                                    processed_cntr += 1

                if ball_on < 4 and self.second_best_camera != -1:
                    self.do_detection_selected_camera(self.second_best_camera, 6)
                    self.run_detection()
                    self.current_stage = 6
                    return
                self.current_stage = 99
            else:
                print("NO Best camera found......\n\n")
                self.current_stage = 99

            return

        if self.current_stage == 4:
            self.current_stage = 99
            return

        if self.current_stage == 6:
            self.make_camera_selection(just_do_selection_nothing_else=True)
            self.current_stage = 99
            return        

        if (self.current_stage == 99):
            self.current_stage = 110
            return

        if (self.current_stage == 110):
            self.current_stage = 111
            return

        if (self.current_stage == 111):
            self.run_stage_final()
            self.detector_cycle += 1
            self.current_stage = 1

    def get_best_camera(self):
        """Determine the best camera based on ball detection confidence"""
        best_camera = -1
        second_best_camera = -1
        best_camera_avrg_conf = 0
        second_best_camera_avrg_conf = 0
        best_camera_ball_on = 0  # Track the number of valid detections for the best camera
        second_best_camera_ball_on = 0  # Track the number of valid detections for the second best camera
        total_ball_on = 0

        for camera in self.cameras:
            ball_on = 0
            confid = 0
            if (camera.capture_only_device == True):
                continue            
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None and obj.ball > 0:
                    ball_on += 1
                    total_ball_on += 1
                    confid += obj.confidence
            if ball_on > 0:
                this_avrg_conf = confid / ball_on
                # Update based on ball_on first, then average confidence
                if ball_on > best_camera_ball_on or (ball_on == best_camera_ball_on and this_avrg_conf > best_camera_avrg_conf):
                    # Move current best to second best
                    second_best_camera = best_camera
                    second_best_camera_avrg_conf = best_camera_avrg_conf
                    second_best_camera_ball_on = best_camera_ball_on
                    # Update the best camera
                    best_camera = camera.camera_id
                    best_camera_avrg_conf = this_avrg_conf
                    best_camera_ball_on = ball_on
                elif ball_on > second_best_camera_ball_on or (ball_on == second_best_camera_ball_on and this_avrg_conf > second_best_camera_avrg_conf):
                    # Update the second best camera without changing the best
                    second_best_camera = camera.camera_id
                    second_best_camera_avrg_conf = this_avrg_conf
                    second_best_camera_ball_on = ball_on

        return best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on

    def make_camera_selection(self, just_do_selection_nothing_else=False):
        """Select the best cameras for detection"""
        best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on = self.get_best_camera()

        self.best_camera = best_camera
        self.second_best_camera = second_best_camera

        if (just_do_selection_nothing_else == True):
            self.current_stage = 99
            return

        if (best_camera != -1):
            self.do_detection_selected_camera(best_camera, self.ai_settings.detection_last_stage_on_best_frames)
            self.run_detection()
            self.current_stage = 3
        else:
            self.do_detection_selected_camera(2, self.ai_settings.detection_last_stage_on_both_centers_frames)
            self.run_detection()
            outs = self.get_number_of_outstanding_detections()
            self.current_stage = 4

    def do_detection_selected_camera(self, camera_id, nbr_of_frames=6):
        """Mark specific frames for detection on the selected camera"""
        frames_marked = [-1, self.max_fps+1]  # we have an entire second as a starting reference
        available_frames_to_process = []
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                detector_objs = camera.get_detector_collection(self.detector_cycle)
                for det_obj in detector_objs:
                    if det_obj is not None:
                        if det_obj.skip == True and det_obj.in_progress == False:
                            available_frames_to_process.append(det_obj.position)
                        else:
                            frames_marked.append(det_obj.position)
        if len(available_frames_to_process) > 0:
            for i in range(nbr_of_frames):
                frame_pos = insert_for_even_distribution(frames_marked, available_frames_to_process)
                if (frame_pos is not None):
                    for det_obj in detector_objs:
                        if det_obj is not None and det_obj.position == frame_pos:
                            det_obj.skip = False                        
                            frames_marked.append(frame_pos)

    def run_stage_final(self):
        """Final stage of processing for a detection cycle"""
        best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on = self.get_best_camera()
        self.best_camera = best_camera
        self.second_best_camera = second_best_camera

        processed_cntr = 0
        goal_processed_cntr = 0
        ball_on = 0
        goal_ball_on = 0
        frames = 0
        goal_frames = 0
        framepos = []
        for cam in self.cameras:
            if (cam.capture_only_device == True):
                continue            
            detector_objs = cam.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None:
                    frames += 1
                    if obj.ball > 0:
                        ball_on += 1
                        framepos.append(obj.position)
                    if obj.processed == True:
                        processed_cntr += 1
                        
        if self.best_camera == -1:
            if self.last_best_camera != -1:
                self.best_camera = self.last_best_camera
            else:
                self.best_camera = self.cameras[0].camera_id
                
        for cam in self.cameras:
            if cam.camera_id == self.best_camera:
                detector_objs = cam.get_detector_collection(self.detector_cycle)
                self.fill_gaps(cam, self.detector_cycle, self.second_best_camera)
                self.last_best_camera_obj = cam
                
                det_frames = []
                for det in detector_objs:
                    if det is not None and det.processed == True:
                        det_frames.append(det.position)
                break
        
        for camera in self.cameras:
            camera.detector_cycle_completed(self.detector_cycle - 4)
        
        self.broadcast_camera = self.best_camera
        if (self.best_camera != 0 and self.best_camera != 1):
            if second_best_camera_ball_on > 3:
                self.broadcast_camera = second_best_camera

        self.last_best_camera = self.best_camera
        elapsed = round((time.monotonic_ns() - self.stage_start_time) / 1000 / 1000)
        framepos = sorted(set(framepos))
        self.last_proc_time = elapsed
        logging.info(f'{self.detector_cycle};{self.best_camera};{ball_on};{processed_cntr};{frames};{goal_ball_on};{goal_processed_cntr};{goal_frames};{elapsed};{framepos}')

    def linear_interpolate(self, start_value, end_value, steps):
        """Linearly interpolate between two values over a given number of steps."""
        return [(start_value + (end_value - start_value) * step / (steps - 1)) for step in range(steps)]

    def identify_wrong_numbers(self, x_coords):
        """Identify outliers in a sequence of coordinates"""
        # Helper function to calculate the average change between non-missing, consecutive values
        def calculate_average_change(cleaned_coords):
            diffs = [abs(cleaned_coords[i+1] - cleaned_coords[i]) for i in range(len(cleaned_coords)-1)]
            return sum(diffs) / len(diffs) if diffs else 0
        
        # Clean the input list by removing missing values (-1)
        cleaned_coords = [x for x in x_coords if x != -1]
        
        # Calculate the average change to set a dynamic threshold
        average_change = calculate_average_change(cleaned_coords)
        threshold = average_change * 6  # Adjust the threshold as needed
        
        wrong_numbers = []  # List to store identified wrong numbers
        
        # Iterate through the list of cleaned coordinates to identify outliers
        for i, x in enumerate(cleaned_coords):
            # For each point, check if it significantly deviates from the trajectory of its neighbors
            if i > 0 and i < len(cleaned_coords) - 1:  # Ensure there are both previous and next items
                prev_val = cleaned_coords[i-1]
                next_val = cleaned_coords[i+1]
                if abs(prev_val - x) > threshold and abs(next_val - x) > threshold:
                    wrong_numbers.append(x)
                    
        return wrong_numbers

    def remove_wrong(self, cam, cycle):
        """Remove incorrectly detected coordinates"""
        numbers_x = []
        numbers_y = []
        detector_objs = cam.get_detector_collection(cycle)
        for obj in detector_objs:
            if obj is not None:
                if obj.x1 > -1:
                    numbers_x.append(obj.x1)
                if obj.y1 > -1:
                    numbers_y.append(obj.y1)
        wrong_x = self.identify_wrong_numbers(numbers_x)
        wrong_y = self.identify_wrong_numbers(numbers_y)
        for obj in detector_objs:
            if obj is not None and (obj.x1 in wrong_x or obj.y1 in wrong_y):
                obj.x1 = -1
                obj.x2 = -1
                obj.y1 = -4
                obj.y2 = -1
                obj.confidence = 0

    def calculate_speed(self, detector_objs):
        """Calculate ball speed and direction from sequential detections"""
        processing_complete = False
        for obj in detector_objs:
            if obj is not None and obj.x1 != -1:  # Check if data is valid
                if (self.ball_color_counter[obj.camera_id] > 100):
                    curr_avrg = self.ball_color_total[obj.camera_id] / self.ball_color_counter[obj.camera_id]
                    if abs(curr_avrg - obj.mean_saturation) > 50:
                        obj.x1 = -1
                        obj.y1 = -1
                        obj.x2 = -1
                        obj.y2 = -1
                        obj.confidence = -9
                        obj.x_2d = -1
                        obj.y_2d = -1                            

        while not processing_complete:
            last_valid_obj_x = None
            last_valid_obj_y = None
            last_valid_obj_x_2d = None
            last_valid_obj_y_2d = None
            last_valid_obj_position = None
            frame_gap_count = 0  # Counter for frames between valid data points
            restart_iteration = False

            for obj in detector_objs:
                if obj is not None and obj.x1 != -1 and obj.x_2d != -1:  # Check if data is valid
                    if last_valid_obj_position is not None:
                        time_s = 0.04 * (frame_gap_count + 1)  
                        # Calculate distance in pixels
                        distance_px = math.sqrt((obj.x_2d - last_valid_obj_x_2d) ** 2 + (obj.y_2d - last_valid_obj_y_2d) ** 2)

                        # Convert distance from pixels to meters (using given field dimensions and pixel ranges)
                        distance_m = (distance_px / 500) * 25  # 500 pixels = 25 meters width

                        # Calculate speed in m/s
                        speed_ms = distance_m / time_s
                        speed_kmh = speed_ms * 3.6

                        # Calculate direction in radians
                        dy = obj.y1 - last_valid_obj_y
                        dx = obj.x1 - last_valid_obj_x
                        angle_radians = math.atan2(dy, dx)
                        # Convert radians to degrees
                        angle_degrees = math.degrees(angle_radians)
                        angle_degrees = (angle_degrees + 360) % 360

                        # Store speeds in the object
                        obj.ball_speed_ms = round(speed_ms, 1)
                        obj.ball_speed_kmh = round(speed_kmh, 1)
                        obj.ball_direction = int(angle_degrees)
                        if speed_kmh > 135:
                            obj.x1 = -1
                            obj.y1 = -1
                            obj.x2 = -1
                            obj.y2 = -1
                            obj.confidence = -8
                            obj.x_2d = -1
                            obj.y_2d = -1                            
                            restart_iteration = True
                            break

                        self.ball_color_counter[obj.camera_id] += 1
                        self.ball_color_total[obj.camera_id] += obj.mean_saturation
                        frame_gap_count = 0
                    last_valid_obj_x = obj.x1
                    last_valid_obj_y = obj.y1
                    last_valid_obj_x_2d = obj.x_2d
                    last_valid_obj_y_2d = obj.y_2d
                    last_valid_obj_position = obj.position
                else:
                    frame_gap_count += 1

            if not restart_iteration:
                processing_complete = True
            else:
                last_valid_obj_position = None
                frame_gap_count = 0

    def fill_gaps(self, cam, cycle, second_cam_id):
        """Fill in missing detections by interpolation and from second camera"""
        cam.initial_gap_fill(cycle)
        second_cam = None
        for seccam in self.cameras:
            if seccam.camera_id == second_cam_id:
                second_cam = seccam
        detector_objs = cam.get_detector_collection(cycle)
        second_detector_objs = None
        if (second_cam is not None):
            second_detector_objs = second_cam.get_detector_collection(cycle)
        
        if (second_detector_objs is not None):
            c = 0
            for obj in detector_objs:
                if obj.x1 == -1 and c < len(second_detector_objs) and second_detector_objs[c] is not None and second_detector_objs[c].x1 != -1:
                    obj.x1 = second_detector_objs[c].x1
                    obj.y1 = second_detector_objs[c].y1
                    obj.x2 = second_detector_objs[c].x2
                    obj.y2 = second_detector_objs[c].y2
                    obj.x_2d = second_detector_objs[c].x_2d
                    obj.y_2d = second_detector_objs[c].y_2d
                    obj.confidence = 0.19
                c += 1
        cntr = 0
        total_confidence = 0
        valid_obj_collection = []
        for obj in detector_objs:
            if obj is not None and obj.confidence > 0:
                valid_obj_collection.append(obj)
                cntr += 1
                total_confidence += obj.confidence
        avrg_confidence = 0
        if cntr > 0:
            avrg_confidence = total_confidence / cntr
        length = len(valid_obj_collection)
        if length > 1:
            if valid_obj_collection[0].x1 == -1 and self.last_best_camera_obj is not None:
                last, _ = self.last_best_camera_obj.get_last_value(cycle-1)
                if last is not None and last.position > 20 and last.x1 != -1:
                    valid_obj_collection[0].x1 = last.x1
                    valid_obj_collection[0].y1 = last.y1
                    valid_obj_collection[0].x2 = last.x2
                    valid_obj_collection[0].y2 = last.y2
                    valid_obj_collection[0].x_2d = last.x_2d
                    valid_obj_collection[0].y_2d = last.y_2d                    
                    valid_obj_collection[0].ball_speed_ms = last.ball_speed_ms
                    valid_obj_collection[0].ball_speed_kmh = last.ball_speed_kmh
                    valid_obj_collection[0].ball_direction = last.ball_direction
                    valid_obj_collection[0].confidence = 0.09

            for c in range(length - 1):
                obj_left = valid_obj_collection[c]
                obj_right = valid_obj_collection[c + 1]
                dif = obj_right.position - obj_left.position
                if (dif) > 1:
                    x2_step = (obj_right.x2 - obj_left.x2) / dif
                    x2 = obj_left.x2
                    x1_step = (obj_right.x1 - obj_left.x1) / dif
                    x1 = obj_left.x1
                    y2_step = (obj_right.y2 - obj_left.y2) / dif
                    y2 = obj_left.y2
                    y1_step = (obj_right.y1 - obj_left.y1) / dif
                    y1 = obj_left.y1

                    x2d_step = (obj_right.x_2d - obj_left.x_2d) / dif
                    x_2d = obj_left.x_2d
                    y2d_step = (obj_right.y_2d - obj_left.y_2d) / dif
                    y_2d = obj_left.y_2d
                    speed_ms = obj_left.ball_speed_ms                    
                    speed_kmh = obj_left.ball_speed_kmh                    
                    direction = obj_left.ball_direction                    

                    for i in range(obj_left.position + 1, obj_right.position):
                        x2 += x2_step
                        x1 += x1_step
                        y1 += y1_step
                        y2 += y2_step
                        x_2d += x2d_step
                        y_2d += y2d_step
                        detector_objs[i].x2 = int(x2)
                        detector_objs[i].x1 = int(x1)
                        detector_objs[i].y2 = int(y2)
                        detector_objs[i].y1 = int(y1)
                        detector_objs[i].x_2d = int(x_2d)
                        detector_objs[i].y_2d = int(y_2d)
                        
                        try:
                            detector_objs[i].ball_speed_ms = round(speed_ms, 1)
                            detector_objs[i].ball_speed_kmh = int(speed_kmh)
                            detector_objs[i].ball_direction = int(direction)
                        except:
                            detector_objs[i].ball_speed_ms = -1
                            detector_objs[i].ball_speed_kmh = -1
                            detector_objs[i].ball_direction = 0
                        detector_objs[i].confidence = 0.11
                        
                        if (obj_right.confidence == 0.19):
                           detector_objs[i].x1 = -1
                           detector_objs[i].x2 = -1
                           detector_objs[i].y1 = -1 
                           detector_objs[i].y2 = -1
                        else:
                            cam.convert_to_2d(detector_objs[i])

        self.calculate_speed(detector_objs)

    def deep_check2(self, cycle_id, frame, x1, y1, x2, y2):
        """Check if a region is likely to be a ball by analyzing color properties in HSV space"""
        # Calculate center and radius of the circle within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]

        # Create a circular mask
        mask = np.zeros(region.shape[:2], dtype="uint8")
        cv2.circle(mask, (center_x - x1, center_y - y1), radius, 255, -1)  # Adjust circle position in mask

        # Apply the mask to the region to isolate the circle
        masked_region = cv2.bitwise_and(region, region, mask=mask)
        if masked_region is None:
            return 255, 255, 0

        # Convert the masked region to HSV color space
        hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)

        # Calculate the average saturation and value in the masked region
        masked_hsv = cv2.bitwise_and(hsv_region, hsv_region, mask=mask)
        s_channel = masked_hsv[:, :, 1]  # Saturation channel
        v_channel = masked_hsv[:, :, 2]  # Value (brightness) channel
        mean_saturation = np.mean(s_channel[s_channel > 0])  # Mean saturation of non-zero mask areas
        mean_value = np.mean(v_channel[v_channel > 0]) 
        deviation = np.sqrt((255 - mean_value) ** 2 + mean_saturation ** 2)

        if mean_saturation == np.nan:
            mean_saturation = 255
        if mean_value == np.nan:
            mean_value = 255

        return mean_saturation, mean_value, 0

    def deep_check3(self, cycle_id, frame, x1, y1, x2, y2):
        """Enhanced ball detection with edge detection and whiteness check"""
        # Calculate center and radius of the circle within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]

        # Create a circular mask
        mask = np.zeros(region.shape[:2], dtype="uint8")
        cv2.circle(mask, (center_x - x1, center_y - y1), radius, 255, -1)  # Adjust circle position in mask

        # Apply the mask to the region to isolate the circle
        masked_region = cv2.bitwise_and(region, region, mask=mask)
        if masked_region is None:
            return 255, 255, 0

        # Convert the masked region to HSV color space
        hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)

        # Calculate the average saturation and value in the masked region
        masked_hsv = cv2.bitwise_and(hsv_region, hsv_region, mask=mask)
        s_channel = masked_hsv[:, :, 1]  # Saturation channel
        v_channel = masked_hsv[:, :, 2]  # Value (brightness) channel
        
        # Only consider the non-zero mask areas
        non_zero_mask = mask > 0
        if np.count_nonzero(non_zero_mask) == 0:
            return 255, 255, 0  # Return if the mask is completely zero
        
        mean_saturation = np.mean(s_channel[non_zero_mask])  # Mean saturation of non-zero mask areas
        mean_value = np.mean(v_channel[non_zero_mask])  # Mean value of non-zero mask areas
        deviation = np.sqrt((255 - mean_value) ** 2 + mean_saturation ** 2)
        is_white = mean_saturation < 20 and mean_value > 235  # Thresholds can be adjusted
        
        # Edge detection to help handle partial occlusions
        gray_masked = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_masked, 100, 200)
        edge_density = np.sum(edges > 0) / (np.pi * radius * radius)
        is_edge_dense = edge_density < 0.2  # Adjust based on ball size and edge density expectations

        if is_white and is_edge_dense:
            return mean_saturation, mean_value, 1  # Return a flag indicating it is likely a ball
        else:
            return mean_saturation, mean_value, 0
    
    def run_stage_1(self, camera_id=-1, number_of_frames=6):
        """Initial detection stage - select frames to process"""
        if number_of_frames == 0:
            return
        total_frm = 0
        for camera in self.cameras:
            detector_objs = camera.get_non_none_detector_collection(self.detector_cycle)
            total_frm += len(detector_objs)
        if (total_frm < 12):
            self.current_stage = 99
            return

        for camera in self.cameras:
            frames_marked = []
            if camera_id == -1 or camera.camera_id == camera_id:
                if (camera.capture_only_device == True):
                    continue

                available_frames_to_process = []
                detector_objs = camera.get_detector_collection(self.detector_cycle)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.skip == True and det_obj.in_progress == False:
                        available_frames_to_process.append(det_obj.position)
                if len(available_frames_to_process) > 0:
                    for i in range(number_of_frames):
                        frame_pos = insert_for_even_distribution(frames_marked, available_frames_to_process)
                        if (frame_pos is not None and frame_pos >= 0):
                            try:
                                for det_obj in detector_objs:
                                    if det_obj is not None and det_obj.position == frame_pos:
                                        det_obj.skip = False
                                        frames_marked.append(frame_pos)
                                        break
                            except:
                                print(f'ERROR - FAILED TO get frame from position {frame_pos} on camera {camera.camera_id}')

        if self.goal_l_reference_taken == False:
            for camera in self.cameras:
                if camera.camera_id == 11:
                    l_cntr = 0
                    detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                    for det_obj in detector_objs:
                        if (det_obj is not None and det_obj.frame is not None):
                            det_obj.skip = False
                            l_cntr += 1
                            if l_cntr >= 6:
                                break
                if camera.camera_id == 10:
                    r_cntr = 0
                    detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                    for det_obj in detector_objs:
                        if (det_obj is not None and det_obj.frame is not None):
                            det_obj.skip = False
                            r_cntr += 1
                            if r_cntr >= 6:
                                break
        self.run_detection()

    def process_results_non_blocking(self):
        """Process detection results without blocking the main thread"""
        start_time = time.time()  # Capture start time

        # Temporarily hold futures that are not yet completed
        pending_futures = []

        for future in self.futures:
            if future.done():  # Check if the future is done
                try:
                    camera_id, cycle_id, det_obj = future.result(timeout=0)  # Non-blocking
                    det_obj.processed = True
                    # Mark this future as completed, process if necessary
                except Exception as e:
                    print(f"Operation generated an exception: {e}")
            else:
                pending_futures.append(future)  # Re-queue the future if it's not done

        # Update the list of futures with those still pending
        self.futures = pending_futures

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        return elapsed_time  # Optionally return the elapsed time for monitoring
    
    def get_grass_color(self, img):
        """Extract the average color of the grass from the image"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        grass_color = cv2.mean(img, mask=mask)
        return grass_color[:3]
    
    def process_frame(self, camera, cycle_id, det_obj, Engine, W, H):
        """Process a frame for object detection, including team classification for players"""
        width, height = 2300, 896

        if det_obj.frame is not None:
            height, width = det_obj.frame.shape[:2]
        if det_obj.frame.shape[1] > 1000:
            width, height = 2300, 896
        else:
            width, height = 960, 576
        resized_frame = cv2.resize(det_obj.frame, (width, height))

        det_obj.tensor = create_tensor(resized_frame, self.device, width, height)
        if (det_obj.tensor is None):
            return camera.camera_id, cycle_id, det_obj

        data = Engine(det_obj.tensor)  # Process the batch through the engine

        bboxes, scores, labels = det_postprocess(data)
        highest_score_ball = -1
        player_imgs = []
        player_boxes = []
        
        if bboxes.numel() == 0:
            pass
        else:
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                x1 = bbox[:2][0]
                y1 = bbox[:2][1]
                x2 = bbox[2:][0]
                y2 = bbox[2:][1]       
                if cls_id == CLASSID_PLAYER and score >= self.ai_settings.people_confidence: 
                    if (camera.camera_id == 2 or camera.camera_id == 3) and y2 < 76:
                        pass
                    elif y2 - y1 > 90: 
                        det_obj.people += 1
                        player = Player()
                        player.x1 = int(x1)
                        player.x2 = int(x2)
                        player.y1 = int(y1)
                        player.y2 = int(y2)
                        player.confidence = round(float(score), 2)
                        
                        # Calculate center coordinates for the player
                        player.centerx = (player.x1 + player.x2) // 2
                        player.centery = (player.y1 + player.y2) // 2
                        
                        # Add player image for team classification
                        player_img = resized_frame[player.y1:player.y2, player.x1:player.x2]
                        if player_img.shape[0] > 0 and player_img.shape[1] > 0:
                            player_imgs.append(player_img)
                            player_boxes.append([player.x1, player.y1, player.x2, player.y2])
                        
                        det_obj.players.append(player)
                        
                if (cls_id == CLASSID_BALL) and score >= self.ai_settings.ball_confidence:
                    if score < highest_score_ball:
                        det_obj.ball += 1
                    else:
                        MIN_BALL_SIZE = self.ai_settings.min_ball_size   #12
                        rad = min(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)))
                        if (rad >= MIN_BALL_SIZE):
                            if self.ai_settings.ball_do_deep_check == True:
                                det_obj.mean_saturation, det_obj.mean_value, det_obj.white_gray = self.deep_check2(cycle_id, det_obj.frame, x1, y1, x2, y2)
                                if (det_obj.mean_saturation > self.ai_settings.ball_mean_saturation or det_obj.mean_value < self.ai_settings.ball_mean_value):
                                    continue                        
                            highest_score_ball = score
                            det_obj.ball += 1
                            det_obj.x1 = int(x1)
                            det_obj.x2 = int(x2)
                            det_obj.y1 = int(y1)
                            det_obj.y2 = int(y2)
                            det_obj.ball_radius = int(rad)
                            det_obj.confidence = round(float(score), 2)
                            camera.convert_to_2d(det_obj)

                        else:
                            det_obj.x1 = -1
                            det_obj.x2 = -1
                            det_obj.y1 = -1
                            det_obj.y2 = -1
                            det_obj.x_2d = -1
                            det_obj.y_2d = -1
                            det_obj.ball_radius = -1
                            det_obj.confidence = -1
            
            # Process team classification if we have player images
            if len(player_imgs) >= 2:
                # Get grass color if not already defined
                if self.grass_hsv is None and det_obj.frame is not None:
                    grass_color = self.get_grass_color(det_obj.frame)
                    self.grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
                
                # Extract kit colors
                kit_colors = self.get_kits_colors(player_imgs, self.grass_hsv, det_obj.frame)
                
                # Add to accumulated kit colors for more robust classification
                self.accumulated_kit_colors.extend(kit_colors)
                if len(self.accumulated_kit_colors) > 100:
                    self.accumulated_kit_colors = self.accumulated_kit_colors[-50:]  # Keep last 100 samples
                
                # Initialize team classifier if we have enough data
                if self.kits_clf is None and len(self.accumulated_kit_colors) >= 20:
                    print(f"Initializing team classification with {len(self.accumulated_kit_colors)} kit colors")
                    self.kits_clf = self.get_kits_classifier(self.accumulated_kit_colors)
                    if self.kits_clf is not None:
                        self.team_colors = self.kits_clf.cluster_centers_
                
                # Determine left team label if classifier is ready
                if self.kits_clf is not None and self.left_team_label is None:
                    self.left_team_label = self.get_left_team_label(player_boxes, kit_colors)
                    print(f"Left team label set to: {self.left_team_label}")
                
                # Classify players to teams if we have a valid classifier
                if self.kits_clf is not None and len(kit_colors) > 0:
                    team_predictions = self.classify_kits(kit_colors)
                    
                    # Assign team IDs to players based on classification
                    for i, player in enumerate(det_obj.players):
                        if i < len(team_predictions):  # Make sure we don't go out of bounds
                            # Convert to left/right team based on left_team_label
                            if self.left_team_label is not None:
                                if team_predictions[i] == self.left_team_label:
                                    player.team_id = 0  # Left team (blue in TEAM_COLORS)
                                else:
                                    player.team_id = 1  # Right team (red in TEAM_COLORS)
                            else:
                                # Temporary assignment before left_team_label is determined
                                player.team_id = int(team_predictions[i])
                        else:
                            # Default assignment
                            player.team_id = -1
                            
        return camera.camera_id, cycle_id, det_obj