import torch
import cv2
import numpy as np
import os
import time
import traceback

# Import GPU utilities from the provided modules
from src.gpu import create_device, create_tensor, get_image_from_gpu
from models.torch_utils import det_postprocess

class DetectionEngine:
    """
    Detection engine for soccer ball and player detection using TensorRT engine.
    Adapted to handle specific engine input requirements (960x576).
    """
    
    def __init__(self, engine_path):
        """
        Initialize the detection engine with a TensorRT engine file
        
        Args:
            engine_path: Path to the TensorRT engine file
        """
        self.engine_path = engine_path
        
        # Explicitly set the input dimensions for your engine
        self.input_width = 960
        self.input_height = 576
        
        # Check if engine file exists
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
        
        # Initialize CUDA device
        try:
            self.device = create_device(0)  # Use first GPU
            print(f"CUDA device initialized: {self.device}")
        except Exception as e:
            print(f"Warning: Failed to create CUDA device: {e}")
            print("Falling back to CPU (this will be much slower)")
            self.device = torch.device("cpu")
        
        # Load TensorRT engine
        self.engine = self.load_engine()
        
        # Define class mapping - Changed "fastball" to "player" for class ID 1
        self.class_map = {
            0: "ball",      # Ball class ID
            1: "player",    # Changed from "fastball" to "player"
           
        }
        
        # Test engine compatibility
        if not self.test_engine():
            print("WARNING: Engine compatibility test failed. Detection may not work properly.")
            print("Try using an engine built for input dimensions:", self.input_width, "x", self.input_height)
        
        # Detection parameters - improved defaults for ball detection
        self.ball_confidence_threshold = 0.4  # Increased from 0.3 to reduce false positives
        self.player_confidence_threshold = 0.5
        self.min_ball_size = 6
        
        # Ball validation parameters
        self.ball_circularity_threshold = 0.7  # Min ratio of width/height for a ball (0-1)
        self.max_ball_size = 50  # Maximum ball size (radius in pixels)
        
        # History tracking for balls to filter out unstable detections
        self.ball_history = []  # List of last few ball detections for stability analysis
        self.ball_history_max_size = 5  # Keep track of last 5 frames
        
        # Counter and stats
        self.detection_count = 0
        self.avg_detection_time = 0
    
    def load_engine(self):
        """Load TensorRT engine from file"""
        try:
            from models import TRTModule  # Import here to handle potential missing module
            
            print(f"Loading TensorRT engine: {self.engine_path}")
            print(f"Engine expects input dimensions: {self.input_width}x{self.input_height}")
            
            engine = TRTModule(self.engine_path, self.device)
            
            # Get input shape from engine to verify
            H, W = engine.inp_info[0].shape[-2:]
            print(f"Engine loaded: Actual input shape {H}x{W}")
            
            # Update our dimensions to match what the engine actually wants
            self.input_height = H
            self.input_width = W
            
            # Configure engine
            engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
            
            return engine
            
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            print("Stack trace:")
            traceback.print_exc()
            
            # Create a dummy engine for testing if real engine fails
            print("Creating dummy detection engine for testing")
            from collections import namedtuple
            
            DummyEngine = namedtuple('DummyEngine', ['inp_info'])
            dummy_shape = [(3, self.input_height, self.input_width)]  # Use specified dimensions
            
            class DummyModule:
                def __init__(self):
                    self.inp_info = dummy_shape
                
                def __call__(self, tensor):
                    # Return dummy detection results
                    # Format mimics the det_postprocess expected output
                    return {
                        'num_dets': torch.tensor([1]),
                        'bboxes': torch.tensor([[100, 100, 120, 120]]),
                        'scores': torch.tensor([0.9]),
                        'labels': torch.tensor([0])  # Ball class
                    }
                
                def set_desired(self, outputs):
                    pass
            
            return DummyModule()
    
    def test_engine(self):
        """Test if the engine is working correctly with a sample image"""
        try:
            print("Testing engine compatibility...")
            
            # Create a test image with the correct dimensions
            test_image = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            
            # Draw some simple shapes to make the test more realistic
            cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
            cv2.circle(test_image, (400, 300), 50, (0, 0, 255), -1)  # Red circle
            
            # Try to run inference
            try:
                # Create tensor
                tensor = create_tensor(test_image, self.device, self.input_width, self.input_height)
                if tensor is None:
                    print("Failed to create tensor from test image")
                    return False
                
                # Print tensor shape for debugging
                print(f"Test tensor shape: {tensor.shape}")
                
                # Run inference
                data = self.engine(tensor)
                
                # Check if we got expected outputs
                if 'num_dets' in data and 'bboxes' in data and 'scores' in data and 'labels' in data:
                    print("Engine test successful! Engine is compatible.")
                    return True
                else:
                    print("Engine test failed: Missing expected outputs")
                    return False
                
            except Exception as e:
                print(f"Engine inference test failed: {e}")
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Engine test error: {e}")
            traceback.print_exc()
            return False
    
    def detect_objects(self, frame):
        """
        Detect objects (ball and players) in a video frame
        
        Args:
            frame: OpenCV frame (BGR format)
        
        Returns:
            List of detection results, each containing:
            - class: 'ball', 'player'
            - confidence: Detection confidence score
            - x1, y1, x2, y2: Bounding box coordinates
            - center_x, center_y: Center point coordinates
            - width, height: Width and height of bounding box
        """
        if frame is None:
            return []
        
        # Start timer
        start_time = time.time()
        
        # Get original frame dimensions for scaling back
        orig_height, orig_width = frame.shape[:2]
        
        # Prepare frame for detection
        try:
            # Resize frame to match engine input requirements exactly
            resized_frame = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Create tensor
            tensor = create_tensor(resized_frame, self.device, self.input_width, self.input_height)
            if tensor is None:
                print("Failed to create tensor from frame")
                return []
            
            # Run detection
            data = self.engine(tensor)
            
            # Post-process results
            bboxes, scores, labels = det_postprocess(data)
            
            # Process detections
            results = []
            ball_detections = []
            player_detections = []
            
            if bboxes.numel() > 0:
                scale_x = orig_width / self.input_width
                scale_y = orig_height / self.input_height
                
                for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    confidence = float(score)
                    
                    # Get box coordinates in resized frame
                    x1_resized, y1_resized = bbox[:2]
                    x2_resized, y2_resized = bbox[2:]
                    
                    # Scale coordinates back to original frame
                    x1 = int(x1_resized * scale_x)
                    y1 = int(y1_resized * scale_y)
                    x2 = int(x2_resized * scale_x)
                    y2 = int(y2_resized * scale_y)
                    
                    # Calculate dimensions
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Map class ID to name
                    class_name = self.class_map.get(cls_id, "unknown")
                    
                    # Create detection object
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height
                    }
                    
                    # Categorize detection by class
                    if class_name == 'ball':
                        ball_detections.append(detection)
                    elif class_name == 'player':
                        if confidence >= self.player_confidence_threshold:
                            player_detections.append(detection)
                            results.append(detection)
            
            # Apply additional filtering for ball detections
            valid_balls = self._filter_ball_detections(ball_detections)
            results.extend(valid_balls)
            
            # Update stats
            self.detection_count += 1
            elapsed = time.time() - start_time
            self.avg_detection_time = (self.avg_detection_time * (self.detection_count - 1) + elapsed) / self.detection_count
            
            return results
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            traceback.print_exc()
            return []
    
    def _filter_ball_detections(self, ball_detections):
        """Apply additional filtering for ball detections to reduce false positives"""
        if not ball_detections:
            return []
        
        valid_balls = []
        
        # First, apply basic confidence and size filters
        for ball in ball_detections:
            # Skip low confidence balls
            if ball['confidence'] < self.ball_confidence_threshold:
                continue
            
            # Get ball dimensions
            width = ball['width']
            height = ball['height'] 
            radius = min(width, height) / 2
            
            # Skip too small balls
            if radius < self.min_ball_size:
                continue
                
            # Skip too large balls
            if radius > self.max_ball_size:
                continue
            
            # Check circularity (aspect ratio)
            aspect_ratio = min(width, height) / max(width, height)
            
            # If highly confident or very circular, accept as valid ball
            if ball['confidence'] > 0.7 or aspect_ratio > 0.85:
                valid_balls.append(ball)
            # If moderate confidence but still reasonably circular, mark as uncertain
            elif ball['confidence'] > self.ball_confidence_threshold and aspect_ratio > self.ball_circularity_threshold:
                ball['uncertain'] = True
                valid_balls.append(ball)
        
        # If we have multiple valid balls, keep only the highest confidence one
        if len(valid_balls) > 1:
            # Sort by confidence (highest first)
            valid_balls.sort(key=lambda x: x['confidence'], reverse=True)
            
            # If highest confidence ball is much more confident, only keep that one
            if valid_balls[0]['confidence'] > valid_balls[1]['confidence'] * 1.5:
                valid_balls = [valid_balls[0]]
        
        # Update ball history for stability analysis
        self._update_ball_history(valid_balls)
        
        return valid_balls
    
    def _update_ball_history(self, valid_balls):
        """Update ball detection history for stability analysis"""
        # Add current detections to history
        self.ball_history.append(valid_balls)
        
        # Keep history at max size
        if len(self.ball_history) > self.ball_history_max_size:
            self.ball_history.pop(0)
    
    def get_stats(self):
        """Return detection engine statistics"""
        return {
            'detection_count': self.detection_count,
            'avg_detection_time': self.avg_detection_time,
            'engine_dimensions': f"{self.input_width}x{self.input_height}"
        }
        
    def visualize_preprocessed_input(self, frame):
        """
        Process a frame exactly as it would be for the engine and return the visualization
        
        This is useful for debugging preprocessing issues.
        """
        # Create a debug visualization showing what we're feeding to the engine
        # Resize to engine input dimensions
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Add text showing the dimensions
        cv2.putText(
            resized,
            f"Engine input: {self.input_width}x{self.input_height}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        return resized