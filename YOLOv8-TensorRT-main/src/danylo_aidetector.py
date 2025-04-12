import time
import math
import logging
import os
import cv2
import numpy as np
import torch
from src.gpu import *
from src.detector_utils import *
from src.camera import *
from src.aisettings import *
from src.ballradiustracker import *
from src.danylo_detection_engine import DetectionEngine
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

#50-77%
CLASSID_BALL=0  # Class ID for ball detection
CLASSID_FASTBALL=1  # Class ID for fast-moving ball
CLASSID_PLAYER=2  # Class ID for player detection

# Initialize ball radius tracker
ballRadiusChecker = BallRadiusTracker()

class AIDetector:
    def __init__(self, ai_settings, cameras, paralell_models, max_fps) -> None:
        logging.basicConfig(level=logging.DEBUG, filename='detector.log', filemode='w',
                    format='%(message)s')
        
        print("Initializing AIDetector for video processing...")
        
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

        # self.ball_color_total = defaultdict(int)
        # self.ball_color_counter = defaultdict(int)

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
        
        # Initialize CUDA device
        try:
            self.device = create_device(0)
            print(f"CUDA device initialized: {self.device}")
        except Exception as e:
            print(f"WARNING: Failed to create CUDA device: {e}")
            self.device = torch.device("cpu")
            print("Using CPU for processing. This will be much slower.")

        self.stage_start_time = 0
        self.last_left_goalkeeper = None
        self.broadcast_camera = -1
        
        # Check if models directory exists
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"WARNING: Models directory {model_dir} created. Please place model files here.")
        
        # Initialize detection engines
        print("Loading detection engines...")
        try:
            self.W_960, self.H_960 = create_engine(self.device, self.engines_960, "960")
            print(f"Successfully loaded 960 engine: {self.W_960}x{self.H_960}")
        except Exception as e:
            print(f"ERROR loading 960 engine: {e}")
            self.W_960, self.H_960 = 640, 384
            
        try:
            self.W_1280, self.H_1280 = create_engine(self.device, self.engines_1280, "1280")
            print(f"Successfully loaded 1280 engine: {self.W_1280}x{self.H_1280}")
        except Exception as e:
            print(f"ERROR loading 1280 engine: {e}")
            self.W_1280, self.H_1280 = 1280, 720
            
        try:
            self.goal_W, self.goal_H = create_engine(self.device, self.goal_engines, "", create_goal_engine=True)
            print(f"Successfully loaded goal engine: {self.goal_W}x{self.goal_H}")
        except Exception as e:
            print(f"ERROR loading goal engine: {e}")
            self.goal_W, self.goal_H = 320, 320
        
        # Log the state of engines
        print(f"Engines initialized. 960 engines: {len(self.engines_960)}, 1280 engines: {len(self.engines_1280)}")
        for i, engine in enumerate(self.engines_960):
            print(f"Engine 960 #{i}: {'Loaded' if engine is not None else 'FAILED'}")
        for i, engine in enumerate(self.engines_1280):
            print(f"Engine 1280 #{i}: {'Loaded' if engine is not None else 'FAILED'}")

    def process_video_frame(self, frame, cycle, position):
        """
        Process a single video frame directly for ball and player detection.
        This is an alternative to the camera-based approach.
        
        Args:
            frame: The video frame to process
            cycle: The current cycle number
            position: The position within the cycle
            
        Returns:
            A Detector object with detection results
        """
        if frame is None:
            print("WARNING: Received None frame")
            return None
        
        # Create a detector object
        det_obj = Detector(0, 0, position, cycle, DetectionType.NORMAL)
        det_obj.frame = frame.copy()
        det_obj.timestamp = int(time.time() * 1000)
        
        # Select engine based on frame size
        frame_width = frame.shape[1]
        
        if frame_width > 1000:
            # Use 1280 engine for larger frames
            engine = self.engines_1280[0] if self.engines_1280 else None
            W, H = self.W_1280, self.H_1280
        else:
            # Use 960 engine for smaller frames
            engine = self.engines_960[0] if self.engines_960 else None
            W, H = self.W_960, self.H_960
        
        if engine is None:
            print("WARNING: No suitable engine found for detection")
            return det_obj
        try:
            from models import TRTModule  # isort:skip
            engine = TRTModule("best_downloaded_0429_960.engine", torch.device('cuda:0'))
            engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
            W, H = engine.inp_info[0].shape[-2:] 
        except Exception as e:
            print(f"Error loading TRT module: {e}")
           
            from danylo_detection_engine import DetectionEngine
            engine = DetectionEngine("best_downloaded_0429_960.engine")
            W = engine.input_width
            H = engine.input_height
            
            engine.detect_wrapper = lambda tensor: engine.detect_objects(get_image_from_gpu(tensor))
            engine.__call__ = engine.detect_wrapper
        
        try:
            print(f"Processing frame: {frame.shape}, engine size: {W}x{H}")
            
            # Create tensor
            det_obj.tensor = create_tensor(frame, self.device, W, H)
            if det_obj.tensor is None:
                print("ERROR: Failed to create tensor from frame")
                return det_obj
            
            # Run detection
            data = engine(det_obj.tensor)
            
            # Post-process results
            bboxes, scores, labels = det_postprocess(data)
            print(f"Detection results: {bboxes.numel()} objects found")
            
            # Process detections
            if bboxes.numel() > 0:
                highest_score_ball = -1
                
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    
                    x1 = bbox[:2][0]
                    y1 = bbox[:2][1]
                    x2 = bbox[2:][0]
                    y2 = bbox[2:][1]
                    
                    # Handle player detection
                    if cls_id == CLASSID_PLAYER and score >= self.ai_settings.people_confidence:
                        print(f"Player detected: ({x1},{y1})-({x2},{y2}), score: {score:.2f}")
                        det_obj.people += 1
                        player = Player()
                        player.x1 = int(x1)
                        player.x2 = int(x2)
                        player.y1 = int(y1)
                        player.y2 = int(y2)
                        player.confidence = round(float(score), 2)
                        det_obj.players.append(player)
                    
                    # Handle ball detection
                    elif (cls_id == CLASSID_BALL or cls_id == CLASSID_FASTBALL) and score >= self.ai_settings.ball_confidence:
                        if score < highest_score_ball:
                            det_obj.ball += 1
                        else:
                            MIN_BALL_SIZE = self.ai_settings.min_ball_size
                            rad = min(abs(int(x2)-int(x1)), abs(int(y2)-int(y1)))
                            
                            if rad >= MIN_BALL_SIZE:
                                print(f"Ball detected: ({x1},{y1})-({x2},{y2}), radius: {rad}, score: {score:.2f}")
                                if self.ai_settings.ball_do_deep_check:
                                    det_obj.mean_saturation, det_obj.mean_value, det_obj.white_gray = self.deep_check2(cycle, frame, x1, y1, x2, y2)
                                    if det_obj.mean_saturation > self.ai_settings.ball_mean_saturation or det_obj.mean_value < self.ai_settings.ball_mean_value:
                                        print(f"Ball rejected by color check: saturation={det_obj.mean_saturation}, value={det_obj.mean_value}")
                                        continue
                                
                                highest_score_ball = score
                                det_obj.ball += 1
                                det_obj.x1 = int(x1)
                                det_obj.x2 = int(x2)
                                det_obj.y1 = int(y1)
                                det_obj.y2 = int(y2)
                                det_obj.ball_radius = int(rad)
                                det_obj.confidence = round(float(score), 2)
                                
                                # Simple 2D coordinate calculation for tracking
                                h, w = frame.shape[:2]
                                det_obj.x_2d = int((det_obj.x1 + det_obj.x2) / 2 / w * 500)
                                det_obj.y_2d = int((det_obj.y1 + det_obj.y2) / 2 / h * 250)
                            else:
                                print(f"Ball rejected: too small (radius={rad})")
            
            det_obj.processed = True
            return det_obj
            
        except Exception as e:
            print(f"ERROR processing frame: {e}")
            import traceback
            traceback.print_exc()
            return det_obj
    
    def process_goal_frame(self, frame, cycle, position):
        """Process a frame for goal detection"""
        if frame is None:
            return None
            
        # Create detector object
        det_obj = Detector(0, 10 if position % 2 == 0 else 11, position, cycle, DetectionType.GOAL)
        det_obj.frame = frame.copy()
        det_obj.timestamp = int(time.time() * 1000)
        
        # Use goal engine if available
        if not self.goal_engines:
            return det_obj
            
        engine = self.goal_engines[0]
        if engine is None:
            return det_obj
            
        try:
            # Resize frame for goal detection
            resized_frame = cv2.resize(frame, (self.goal_W, self.goal_H))
            
            # Create tensor
            tensor = create_tensor(resized_frame, self.device, self.goal_W, self.goal_H)
            if tensor is None:
                return det_obj
                
            # Run detection
            data = engine(tensor)
            
            # Post-process
            bboxes, scores, labels = det_postprocess(data)
            
            # Process goal detections
            if bboxes.numel() > 0:
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    
                    # Count all persons in goal area
                    if cls_id == CLASSID_PLAYER and score >= 0.3:
                        det_obj.people += 1
            
            det_obj.processed = True
            return det_obj
            
        except Exception as e:
            print(f"ERROR processing goal frame: {e}")
            return det_obj

    def start_cycle(self, cycle):
        """Start a new detection cycle"""
        print(f"Starting detection cycle {cycle}")
        self.detector_cycle = cycle
        self.current_stage = 1
        self.stage_start_time = time.monotonic_ns()
    
    def process(self, cycle):
        """
        Process frames for detection. In video mode, this is a simpler process
        without the complex stage management needed for live camera feeds.
        """
        # Process any pending results
        self.process_results_non_blocking()
        
        # If we're waiting for results, keep waiting
        if self.futures:
            return
        
        # In video mode, we simply process frames as they become available
        for cam in self.cameras:
            detector_objs = cam.get_detector_collection(cycle)
            for det_obj in detector_objs:
                if det_obj is not None and det_obj.skip == False and det_obj.processed == False and det_obj.in_progress == False:
                    det_obj.in_progress = True
                    
                    # Submit for processing
                    engine_idx = 0  # Use first engine for simplicity
                    
                    if cam.frame_w > 1000:
                        engine = self.engines_1280
                        engine_w = self.W_1280
                        engine_h = self.H_1280
                    else:
                        engine = self.engines_960
                        engine_w = self.W_960
                        engine_h = self.H_960
                    
                    if cam.detection_type == DetectionType.GOAL:
                        self.futures.append(self.executor.submit(
                            self.process_frame_goal, cam, cycle, det_obj, 
                            self.goal_engines[engine_idx], self.goal_W, self.goal_H))
                    else:
                        self.futures.append(self.executor.submit(
                            self.process_frame, cam, cycle, det_obj, 
                            engine[engine_idx], engine_w, engine_h))
    
    def get_best_camera(self):
        """Find the best camera based on ball detection performance"""
        best_camera = -1
        second_best_camera = -1
        best_camera_avrg_conf = 0
        second_best_camera_avrg_conf = 0
        best_camera_ball_on = 0
        second_best_camera_ball_on = 0
        
        for camera in self.cameras:
            ball_on = 0
            confid = 0
            
            if camera.capture_only_device:
                continue
                
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None and obj.ball > 0:
                    ball_on += 1
                    confid += obj.confidence
            
            if ball_on > 0:
                this_avrg_conf = confid / ball_on
                
                if ball_on > best_camera_ball_on or (ball_on == best_camera_ball_on and this_avrg_conf > best_camera_avrg_conf):
                    second_best_camera = best_camera
                    second_best_camera_avrg_conf = best_camera_avrg_conf
                    second_best_camera_ball_on = best_camera_ball_on
                    
                    best_camera = camera.camera_id
                    best_camera_avrg_conf = this_avrg_conf
                    best_camera_ball_on = ball_on
                elif ball_on > second_best_camera_ball_on or (ball_on == second_best_camera_ball_on and this_avrg_conf > second_best_camera_avrg_conf):
                    second_best_camera = camera.camera_id
                    second_best_camera_avrg_conf = this_avrg_conf
                    second_best_camera_ball_on = ball_on
        
        return best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on
    
    def process_results_non_blocking(self):
        """Process completed futures without blocking"""
        pending_futures = []
        
        for future in self.futures:
            if future.done():
                try:
                    camera_id, cycle_id, det_obj = future.result(timeout=0)
                    det_obj.processed = True
                    det_obj.in_progress = False
                except Exception as e:
                    print(f"Error in future processing: {e}")
            else:
                pending_futures.append(future)
        
        self.futures = pending_futures
    
    def get_number_of_outstanding_detections(self):
        """Count outstanding (unprocessed) detections"""
        self.outstanding_detections = 0
        for camera in self.cameras:
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None and obj.skip == False and obj.processed == False:
                    self.outstanding_detections += 1
        return self.outstanding_detections
    
    def get_image(self, tensor):
        """Get image from tensor"""
        return get_image_from_gpu(tensor)
    
    def deep_check2(self, cycle_id, frame, x1, y1, x2, y2):
        """Check if a detected region is likely to be a ball based on color"""
        # Calculate center and radius of the circle within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return 255, 255, 0

        # Create a circular mask
        mask = np.zeros(region.shape[:2], dtype="uint8")
        try:
            cv2.circle(mask, (center_x - x1, center_y - y1), radius, 255, -1)
        except Exception as e:
            print(f"Error creating mask: {e}")
            return 255, 255, 0

        # Apply the mask to the region to isolate the circle
        masked_region = cv2.bitwise_and(region, region, mask=mask)
        if masked_region is None or np.sum(mask) == 0:
            return 255, 255, 0

        # Convert the masked region to HSV color space
        hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)

        # Calculate the average saturation and value in the masked region
        masked_hsv = cv2.bitwise_and(hsv_region, hsv_region, mask=mask)
        s_channel = masked_hsv[:, :, 1]  # Saturation channel
        v_channel = masked_hsv[:, :, 2]  # Value (brightness) channel
        mean_saturation = np.mean(s_channel[s_channel > 0]) if np.any(s_channel > 0) else 255
        mean_value = np.mean(v_channel[v_channel > 0]) if np.any(v_channel > 0) else 255

        if np.isnan(mean_saturation):
            mean_saturation = 255
        if np.isnan(mean_value):
            mean_value = 255

        return mean_saturation, mean_value, 0
    
    def process_frame(self, camera, cycle_id, det_obj, Engine, W, H):
        """Process a frame using the detection engine"""
        print(f"Processing frame for camera {camera.camera_id}, cycle {cycle_id}, position {det_obj.position}")
        
        # Check if engine is available
        if Engine is None:
            print("WARNING: No engine available for detection")
            return camera.camera_id, cycle_id, det_obj
        
        # Create tensor from frame
        det_obj.tensor = create_tensor(det_obj.frame, self.device, W, H)
        if det_obj.tensor is None:
            print("ERROR: Failed to create tensor from frame")
            return camera.camera_id, cycle_id, det_obj

        # Process through engine
        try:
            if hasattr(Engine, 'detect_objects'):
                detections = Engine.detect_objects(det_obj.frame)
                for det in detections:
                    if det['class'] == 'player' and det['confidence'] >= self.ai_settings.people_confidence:
                        det_obj.people += 1
                        player = Player()
                        player.x1 = int(det['x1'])
                        player.x2 = int(det['x2'])
                        player.y1 = int(det['y1'])
                        player.y2 = int(det['y2'])
                        player.confidence = round(float(det['confidence']), 2)
                        det_obj.players.append(player)
                    elif det['class'] == 'ball' and det['confidence'] >= self.ai_settings.ball_confidence:
                        det_obj.ball += 1
                        det_obj.x1 = int(det['x1'])
                        det_obj.x2 = int(det['x2'])
                        det_obj.y1 = int(det['y1'])
                        det_obj.y2 = int(det['y2'])
                        det_obj.ball_radius = min(det_obj.x2 - det_obj.x1, det_obj.y2 - det_obj.y1) // 2
                        det_obj.confidence = round(float(det['confidence']), 2)
                        camera.convert_to_2d(det_obj)
            else:
                try:
                    data = Engine(det_obj.tensor)
                    bboxes, scores, labels = det_postprocess(data)
                except Exception as e:
                    print(f"ERROR in detection: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Post-process
            bboxes, scores, labels = det_postprocess(data)
            print(f"Detection results: {bboxes.numel()} objects found")
            
            # Process detections
            if bboxes.numel() > 0:
                highest_score_ball = -1
                
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    
                    x1 = bbox[:2][0]
                    y1 = bbox[:2][1]
                    x2 = bbox[2:][0]
                    y2 = bbox[2:][1]
                    
                    # Handle player detection
                    if cls_id == CLASSID_PLAYER and score >= self.ai_settings.people_confidence:
                        det_obj.people += 1
                        player = Player()
                        player.x1 = int(x1)
                        player.x2 = int(x2)
                        player.y1 = int(y1)
                        player.y2 = int(y2)
                        player.confidence = round(float(score), 2)
                        det_obj.players.append(player)
                    
                    # Handle ball detection
                    elif (cls_id == CLASSID_BALL or cls_id == CLASSID_FASTBALL) and score >= self.ai_settings.ball_confidence:
                        if score < highest_score_ball:
                            det_obj.ball += 1
                        else:
                            MIN_BALL_SIZE = self.ai_settings.min_ball_size
                            rad = min(abs(int(x2)-int(x1)), abs(int(y2)-int(y1)))
                            
                            if rad >= MIN_BALL_SIZE:
                                if self.ai_settings.ball_do_deep_check:
                                    det_obj.mean_saturation, det_obj.mean_value, det_obj.white_gray = self.deep_check2(cycle_id, det_obj.frame, x1, y1, x2, y2)
                                    if det_obj.mean_saturation > self.ai_settings.ball_mean_saturation or det_obj.mean_value < self.ai_settings.ball_mean_value:
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
            
        except Exception as e:
            print(f"ERROR in detection: {e}")
            import traceback
            traceback.print_exc()
        
        return camera.camera_id, cycle_id, det_obj
    
    def process_frame_goal(self, camera, cycle_id, det_obj, Engine, W, H):
        """Process a frame for goal detection"""
        if Engine is None:
            return camera.camera_id, cycle_id, det_obj
        
        try:
            # Resize frame for goal detection
            resized_frame = cv2.resize(det_obj.frame, (W, H))
            
            # Create tensor
            tensor = create_tensor(resized_frame, self.device, W, H)
            if tensor is None:
                return camera.camera_id, cycle_id, det_obj
                
            # Run detection
            data = Engine(tensor)
            
            # Post-process
            bboxes, scores, labels = det_postprocess(data)
            
            # Process goal detections
            if bboxes.numel() > 0:
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    
                    # Count all persons in goal area
                    if cls_id == CLASSID_PLAYER and score >= 0.3:
                        det_obj.people += 1
            
        except Exception as e:
            print(f"ERROR processing goal frame: {e}")
        
        return camera.camera_id, cycle_id, det_obj