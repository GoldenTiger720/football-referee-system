import cv2
import multiprocessing as mp
import time
import numpy as np
import threading
import queue
import os
from enum import Enum

class DetectionType(Enum):
    NORMAL = 1
    GOAL = 2

class Player:
    def __init__(self):
        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.confidence = 0
        self.closest = False
        self.x_2d = -1
        self.y_2d = -1
        self.cam_id = -1
        self.ball_distance = -1
        self.ball_x = -1
        self.ball_y = -1
        self.player_id = -1

class Detector:
    def __init__(self, field_id, camera_id, position, cycle, dtype=DetectionType.NORMAL) -> None:
        self.field_id=field_id
        self.camera_id=camera_id
        self.position=position
        self.cycle=cycle
        self.detection_type=dtype
        self.timestamp=0
        self.ball=0
        self.x1=-1
        self.y1=-1
        self.x2=-1
        self.y2=-1
        self.x_2d=-1
        self.y_2d=-1
        self.confidence=0
        self.goal=0
        self.people=0
        self.players=[]
        self.processed=False
        self.skip=True
        self.in_progress=False
        self.tensor=None
        self.frame=None
        self.ball_speed_ms=-1
        self.ball_speed_kmh=-1
        self.ball_radius=-1
        self.ball_direction=-1
        self.ball_direction_change=False
        self.fake=False
        self.mean_saturation=-1
        self.mean_value=-1

class CameraObj:
    def __init__(self, field_id, camera_id, frame_id, position, cycle, tensor, timestamp, processed, slave, slave_camera_id, slave_frame_id, slave_position, x_2d, y_2d):
        self.field_id=field_id
        self.camera_id=camera_id
        self.frame_id=frame_id
        self.position=position
        self.cycle=cycle
        self.tensor=tensor
        self.timestamp=timestamp
        self.processed=processed
        self.slave=slave
        self.slave_camera_id=slave_camera_id
        self.slave_frame_id=slave_frame_id
        self.slave_position=slave_position
        self.x_2d = x_2d
        self.y_2d = y_2d

class Camera:
    def __init__(self, field_id, camera_id, url, thread_running, frame_start, max_fps, feed_fps, description="", capture_only_device=False, slaves=None, detection_type=DetectionType.NORMAL):
        if slaves is None:
            slaves = []
        
        self.field_id = field_id
        self.url = url
        self.camera_id = camera_id
        self.max_fps = max_fps
        self.thread_running = thread_running
        self.frame_start = frame_start
        self.description = description
        self.capture_only_device = capture_only_device
        self.slaves = slaves
        self.detection_type = detection_type
        self.feed_fps = feed_fps
        self.next_frame_please = mp.Value('i', 1)
        self.cap = None
        self.current_frame_idx = 0
        self.frame_w = -1
        self.frame_h = -1
        self.output_queue = queue.Queue()
        self.frame_counter = 0
        self.segment_id = 0
        self.cycle_timer = {}
        self.current_live_cycle = 1
        self.detector_objects = {}
        self.current_segment_fps = max_fps
        
        # New variables for video file handling
        self.is_video_file = False
        self.frame_buffer = {}
        self.frame_count = 0
        
        # Initialize corners for 2D projection
        self.corners = []
        self.initialize_corners()
        
        # Start the camera thread if URL is provided
        if len(url) > 0:
            self.thread = threading.Thread(target=self.camera_thread)
            self.thread.daemon = True
            self.thread.start()
    
    def initialize_corners(self):
        # Default corners for 2D projection
        if self.camera_id == 0:
            self.corners = [(68, 76), (431, -10), (626, 62), (280, 576)]
        elif self.camera_id == 1:
            self.corners = [(217, 32), (644, 8), (494, 656), (36, 115)]
        # Add additional camera corners as needed
    
    def set_corners(self, corners):
        self.corners = corners
    
    def initialize_capture(self):
        # Initialize video capture if URL is provided
        if len(self.url) > 0:
            # Determine if this is a video file or a stream
            if os.path.exists(self.url):
                print(f"Opening video file: {self.url}")
                self.is_video_file = True
                self.cap = cv2.VideoCapture(self.url)
                if not self.cap.isOpened():
                    print(f"Error: Could not open video file: {self.url}")
                    return False
                self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video loaded: {self.url} - Resolution: {self.frame_w}x{self.frame_h}, Frames: {self.frame_count}")
                return True
            else:
                # For RTSP or other stream URLs
                print(f"Opening stream: {self.url}")
                self.cap = cv2.VideoCapture(self.url)
                if not self.cap.isOpened():
                    print(f"Error: Could not open camera/stream: {self.url}")
                    return False
                self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return True
        return True

    def camera_thread(self):
        # Initialize capture
        if not self.initialize_capture():
            print(f"Failed to initialize camera {self.camera_id}")
            return

        # If it's a video file, don't process in a thread
        if self.is_video_file:
            print(f"Video file loaded for camera {self.camera_id}. Using frame-by-frame processing.")
            return
        
        # For live streams, process frames continuously
        retry_counter = 0
        prev_frame_time = time.time()
        
        while self.thread_running.value == 1:
            # Read a frame from the capture
            if self.cap is not None and self.cap.isOpened():
                # Wait until we're allowed to capture the next frame
                if self.next_frame_please.value == 0:
                    time.sleep(0.001)
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    # If failed to read frame, retry or reset
                    retry_counter += 1
                    if retry_counter > 10:
                        print(f"Camera {self.camera_id} failed to read frame after 10 retries. Reinitializing...")
                        self.cap.release()
                        self.initialize_capture()
                        retry_counter = 0
                    time.sleep(0.1)
                    continue
                
                # Reset retry counter on success
                retry_counter = 0
                
                # Control frame rate
                current_time = time.time()
                time_diff = current_time - prev_frame_time
                if time_diff < 1.0 / self.max_fps:
                    time.sleep(max(0, (1.0 / self.max_fps) - time_diff))
                
                prev_frame_time = time.time()
                
                # Process frame
                self.process_frame(frame)
                
                # Wait until next frame is requested
                self.next_frame_please.value = 0
            else:
                # If capture is not available, try to initialize
                print(f"Camera {self.camera_id} capture not available. Trying to initialize...")
                self.initialize_capture()
                time.sleep(1)
    
    def process_frame(self, frame):
        # Create a new cycle dictionary if it doesn't exist
        cycle = self.current_live_cycle
        if cycle not in self.detector_objects:
            self.detector_objects[cycle] = {}
            for pos in range(self.max_fps):
                self.detector_objects[cycle][pos] = None
        
        # Determine position in the current cycle
        position = self.frame_counter % self.max_fps
        self.frame_counter += 1
        
        # Create detector object
        obj = Detector(self.field_id, self.camera_id, position, cycle, self.detection_type)
        obj.frame = frame
        obj.timestamp = int(time.time() * 1000)
        obj.skip = True
        
        # Add to detector objects
        self.detector_objects[cycle][position] = obj
        
        # Update cycle if needed
        if position == self.max_fps - 1:
            self.current_live_cycle += 1
    
    def add_frame(self, frame, detector_cycle, position):
        """
        Add a frame directly to the detection queue for video file processing
        """
        if frame is None:
            return
        
        # Create detector object for this frame
        det_obj = Detector(self.field_id, self.camera_id, position, detector_cycle, self.detection_type)
        det_obj.frame = frame.copy()
        det_obj.skip = True
        det_obj.timestamp = int(time.time() * 1000)
        
        # Create cycle dictionary if it doesn't exist
        if detector_cycle not in self.detector_objects:
            self.detector_objects[detector_cycle] = {}
            for pos in range(self.max_fps):
                self.detector_objects[detector_cycle][pos] = None
        
        # Add to detector objects
        self.detector_objects[detector_cycle][position] = det_obj
        
        # Update current live cycle
        if detector_cycle > self.current_live_cycle:
            self.current_live_cycle = detector_cycle
            
        # Update frame counter for internal state
        self.frame_counter += 1
    
    def get_detector_collection(self, cycle, detection_type=None):
        # Get collection of detector objects for a cycle
        if detection_type is None:
            detection_type = self.detection_type
            
        if cycle in self.detector_objects:
            return [det_obj for det_obj in self.detector_objects[cycle].values() 
                   if det_obj is not None and det_obj.detection_type == detection_type]
        return []
    
    def get_non_none_detector_collection(self, cycle):
        # Get non-None detector objects for a cycle
        if cycle in self.detector_objects:
            return [det_obj for det_obj in self.detector_objects[cycle].values() if det_obj is not None]
        return []
    
    def get_goal_frame(self, cycle, position):
        # Get goal frame for a specific cycle and position
        if cycle in self.detector_objects and position in self.detector_objects[cycle]:
            det_obj = self.detector_objects[cycle][position]
            if det_obj is not None and det_obj.detection_type == DetectionType.GOAL:
                return det_obj.frame
        return None
    
    def reset_segment(self, cycle):
        # Reset segment for a new cycle
        if cycle not in self.detector_objects:
            self.detector_objects[cycle] = {}
            for i in range(self.max_fps):
                self.detector_objects[cycle][i] = None
    
    def finalize_detector_array(self, cycle):
        # Finalize detector array for a cycle
        if cycle in self.detector_objects:
            # Fill gaps
            for pos in range(self.max_fps):
                if pos not in self.detector_objects[cycle] or self.detector_objects[cycle][pos] is None:
                    self.detector_objects[cycle][pos] = Detector(self.field_id, self.camera_id, pos, cycle, self.detection_type)
    
    def get_cycle_min_max_times(self):
        # Get min and max timestamps for a cycle
        min_time = None
        max_time = None
        
        detector_objs = self.get_non_none_detector_collection(self.current_live_cycle)
        for obj in detector_objs:
            if obj.timestamp > 0:
                if min_time is None or obj.timestamp < min_time:
                    min_time = obj.timestamp
                if max_time is None or obj.timestamp > max_time:
                    max_time = obj.timestamp
        
        return min_time, max_time
    
    def initial_gap_fill(self, cycle):
        # Initial gap fill for a cycle
        if cycle in self.detector_objects:
            # Find first and last valid positions
            first_valid = None
            last_valid = None
            valid_positions = []
            
            for pos in range(self.max_fps):
                if pos in self.detector_objects[cycle] and self.detector_objects[cycle][pos] is not None:
                    if self.detector_objects[cycle][pos].x1 != -1:
                        if first_valid is None:
                            first_valid = pos
                        last_valid = pos
                        valid_positions.append(pos)
            
            # Fill gaps between valid positions
            if first_valid is not None and last_valid is not None and first_valid != last_valid:
                for i in range(len(valid_positions) - 1):
                    pos1 = valid_positions[i]
                    pos2 = valid_positions[i + 1]
                    
                    if pos2 - pos1 > 1:
                        # Linear interpolation
                        obj1 = self.detector_objects[cycle][pos1]
                        obj2 = self.detector_objects[cycle][pos2]
                        
                        for pos in range(pos1 + 1, pos2):
                            obj = self.detector_objects[cycle][pos]
                            if obj is None:
                                obj = Detector(self.field_id, self.camera_id, pos, cycle, self.detection_type)
                                self.detector_objects[cycle][pos] = obj
                            
                            # Calculate interpolation factor
                            factor = (pos - pos1) / (pos2 - pos1)
                            
                            # Interpolate values
                            obj.x1 = int(obj1.x1 + factor * (obj2.x1 - obj1.x1))
                            obj.y1 = int(obj1.y1 + factor * (obj2.y1 - obj1.y1))
                            obj.x2 = int(obj1.x2 + factor * (obj2.x2 - obj1.x2))
                            obj.y2 = int(obj1.y2 + factor * (obj2.y2 - obj1.y2))
                            obj.confidence = 0.11
    
    def get_last_value(self, cycle):
        # Get last valid value from a cycle
        if cycle in self.detector_objects:
            last_valid = None
            last_valid_pos = -1
            
            for pos in range(self.max_fps - 1, -1, -1):
                if pos in self.detector_objects[cycle] and self.detector_objects[cycle][pos] is not None:
                    if self.detector_objects[cycle][pos].x1 != -1:
                        last_valid = self.detector_objects[cycle][pos]
                        last_valid_pos = pos
                        break
            
            return last_valid, last_valid_pos
        return None, -1
    
    def detector_cycle_completed(self, cycle):
        # Clean up old cycles to free memory
        if cycle in self.detector_objects:
            del self.detector_objects[cycle]
    
    def convert_to_2d(self, detector_obj):
        # Convert 3D coordinates to 2D
        # This is a simplistic conversion for demonstration
        if len(self.corners) == 4 and detector_obj.x1 != -1:
            center_x = (detector_obj.x1 + detector_obj.x2) // 2
            center_y = (detector_obj.y1 + detector_obj.y2) // 2
            
            # Calculate normalized position within the frame
            norm_x = center_x / self.frame_w
            norm_y = center_y / self.frame_h
            
            # Map to 2D pitch (0-500 in x, 0-250 in y)
            detector_obj.x_2d = int(norm_x * 500)
            detector_obj.y_2d = int(norm_y * 250)
            
            return True
        return False
    
    def convert_player_to_2d(self, x, y):
        # Convert player coordinates to 2D
        # Similar to convert_to_2d but for players
        if len(self.corners) == 4:
            # Calculate normalized position
            norm_x = x / self.frame_w
            norm_y = y / self.frame_h
            
            # Map to 2D pitch
            x_2d = int(norm_x * 500)
            y_2d = int(norm_y * 250)
            
            return x_2d, y_2d
        return -1, -1
    
    def convert_to_3d(self, x_2d, y_2d):
        # Convert 2D coordinates back to 3D (approximate)
        if len(self.corners) == 4:
            # Calculate normalized position
            norm_x = x_2d / 500
            norm_y = y_2d / 250
            
            # Map to frame coordinates
            x = int(norm_x * self.frame_w)
            y = int(norm_y * self.frame_h)
            
            return x, y
        return -1, -1