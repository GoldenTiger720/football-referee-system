import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import time
import os
import queue
from PIL import Image, ImageTk

# Local modules
from danylo_detection_engine import DetectionEngine
from danylo_match_analyzer import MatchAnalyzer
from danylo_pitch_visualizer import PitchVisualizer
from danylo_utils import resize_image_to_fit, convert_cv2_to_tkinter

class SoccerRefereeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soccer Match Automatic Referee System")
        
        # Make the application full screen
        self.root.state('zoomed')  # For Windows
        # For Linux/Mac, use:
        # self.root.attributes('-zoomed', True)
        
        self.root.configure(bg="#333333")
        
        # Initialize variables
        self.engine_path = None
        self.video_path = None
        self.rtsp_url = None
        self.is_running = False
        self.current_frame = None
        self.video_capture = None
        self.detection_engine = None
        self.match_analyzer = None
        self.pitch_visualizer = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Define video display size (increased from original)
        self.video_display_width = 640  # Increased from 640
        self.video_display_height = 480  # Increased from 480
        
        # Video source selection
        self.video_source = tk.StringVar(value="file")
        
        # Ball detection parameters
        self.min_ball_size = tk.IntVar(value=6)  # Default minimum ball size
        self.ball_confidence_threshold = tk.DoubleVar(value=0.4)  # Increased from 0.3
        
        # Setup UI components
        self.create_ui()
        
        # Create background workers
        self.detection_thread = None
        self.video_thread = None
        self.analysis_thread = None
        self.display_thread = None
    
    # Define stop_analysis method BEFORE it's referenced in create_ui
    def stop_analysis(self):
        """Stop the analysis and clean up resources"""
        self.is_running = False
        
        # Wait for threads to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        # Reset UI
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.engine_button.config(state="normal")
        self.video_button.config(state="normal")
        self.rtsp_entry.config(state="normal")
        self.rtsp_test_button.config(state="normal")
        
        # Reset variables
        self.current_frame = None
        self.detection_engine = None
        self.match_analyzer = None
        self.pitch_visualizer = None
        
        # Clear queues
        while not self.frame_queue.empty():
            self.frame_queue.get()
        
        while not self.result_queue.empty():
            self.result_queue.get()
        
        self.status_var.set("Analysis stopped")
    
    def create_ui(self):
        # Create main frames with weights for proper scaling
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create subframes in top frame
        self.left_goal_frame = ttk.LabelFrame(self.top_frame, text="Left Goal Detection")
        self.left_goal_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        self.center_video_frame = ttk.LabelFrame(self.top_frame, text="Video Feed")
        self.center_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_goal_frame = ttk.LabelFrame(self.top_frame, text="Right Goal Detection")
        self.right_goal_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        self.control_frame = ttk.LabelFrame(self.top_frame, text="Controls")
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # Add canvases for goal posts
        self.left_goal_canvas = tk.Canvas(self.left_goal_frame, width=200, height=200, bg="black")
        self.left_goal_canvas.pack(padx=5, pady=5)
        
        self.right_goal_canvas = tk.Canvas(self.right_goal_frame, width=200, height=200, bg="black")
        self.right_goal_canvas.pack(padx=5, pady=5)
        
        # Add fixed-size canvas for video (increased size)
        self.video_canvas_container = ttk.Frame(self.center_video_frame)
        self.video_canvas_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.video_canvas = tk.Canvas(
            self.video_canvas_container, 
            width=self.video_display_width, 
            height=self.video_display_height, 
            bg="black"
        )
        self.video_canvas.pack(expand=True)
        # Add control buttons
        ttk.Label(self.control_frame, text="TensorRT Engine:").pack(anchor="w", padx=5, pady=5)
        self.engine_label = ttk.Label(self.control_frame, text="No engine selected")
        self.engine_label.pack(anchor="w", padx=5, pady=5)
        
        self.engine_button = ttk.Button(self.control_frame, text="Select Engine", command=self.select_engine)
        self.engine_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Video source selection radio buttons
        ttk.Label(self.control_frame, text="Video Source:").pack(anchor="w", padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Video File", variable=self.video_source, value="file").pack(anchor="w", padx=20, pady=2)
        ttk.Radiobutton(self.control_frame, text="Camera Stream", variable=self.video_source, value="rtsp").pack(anchor="w", padx=20, pady=2)
        
        # File source option
        self.file_frame = ttk.Frame(self.control_frame)
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.file_frame, text="Video File:").pack(anchor="w")
        self.video_label = ttk.Label(self.file_frame, text="No video selected")
        self.video_label.pack(anchor="w")
        self.video_button = ttk.Button(self.file_frame, text="Select Video", command=self.select_video)
        self.video_button.pack(fill=tk.X)
        
        # RTSP source option
        self.rtsp_frame = ttk.Frame(self.control_frame)
        self.rtsp_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.rtsp_frame, text="RTSP URL:").pack(anchor="w")
        self.rtsp_entry = ttk.Entry(self.rtsp_frame)
        self.rtsp_entry.pack(fill=tk.X)
        self.rtsp_entry.insert(0, 'rtsp://admin:GCexperience@192.168.1.100:554/Streaming/channels/101')
        self.rtsp_test_button = ttk.Button(self.rtsp_frame, text="Test Connection", command=self.test_rtsp_connection)
        self.rtsp_test_button.pack(fill=tk.X)
        
        # Ball detection parameters
        ttk.Label(self.control_frame, text="Ball Detection Parameters:").pack(anchor="w", padx=5, pady=5)
        
        # Min ball size slider
        size_frame = ttk.Frame(self.control_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(size_frame, text="Min Ball Size:").pack(side=tk.LEFT)
        ttk.Scale(size_frame, from_=3, to=15, variable=self.min_ball_size, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(size_frame, textvariable=self.min_ball_size).pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold slider
        conf_frame = ttk.Frame(self.control_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.ball_confidence_threshold, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(conf_frame, textvariable=self.ball_confidence_threshold).pack(side=tk.LEFT, padx=5)
        
        # Engine info
        self.engine_info_label = ttk.Label(self.control_frame, text="Engine Info: Not loaded")
        self.engine_info_label.pack(anchor="w", padx=5, pady=5)
        
        self.start_button = ttk.Button(self.control_frame, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_analysis)
        self.stop_button.pack(fill=tk.X, padx=5, pady=5)
        self.stop_button.config(state="disabled")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Detection stats
        self.detection_stats_var = tk.StringVar()
        self.detection_stats_var.set("Detection Stats: N/A")
        self.detection_stats_label = ttk.Label(self.control_frame, textvariable=self.detection_stats_var)
        self.detection_stats_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Add pitch visualization
        self.pitch_canvas = tk.Canvas(self.bottom_frame, bg="green")
        self.pitch_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load and display the pitch background
        try:
            self.pitch_image = Image.open("pitch.jpg")
            self.update_pitch_background()
        except Exception as e:
            print(f"Failed to load pitch image: {e}")
            # Create a generic green pitch with white lines
            self.create_default_pitch()
    
    def test_rtsp_connection(self):
        """Test connection to the RTSP stream"""
        rtsp_url = self.rtsp_entry.get().strip()
        
        if not rtsp_url:
            messagebox.showerror("Error", "Please enter an RTSP URL")
            return
        
        self.status_var.set("Testing RTSP connection...")
        self.root.update()
        
        # Try to open the RTSP stream in a separate thread to avoid freezing the UI
        test_thread = threading.Thread(target=self._test_rtsp_connection_thread, args=(rtsp_url,))
        test_thread.daemon = True
        test_thread.start()
    def _test_rtsp_connection_thread(self, rtsp_url):
        """Thread to test RTSP connection"""
        try:
            # Open the RTSP stream
            cap = cv2.VideoCapture(rtsp_url)
            
            # Check if the stream is opened
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to connect to RTSP stream"))
                self.root.after(0, lambda: self.status_var.set("RTSP connection failed"))
                return
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if not ret:
                self.root.after(0, lambda: messagebox.showerror("Error", "Connected but failed to read frame"))
                self.root.after(0, lambda: self.status_var.set("RTSP stream: no frames"))
                cap.release()
                return
            
            # Success - show a preview
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Update UI from main thread
            self.root.after(0, lambda: self.status_var.set(f"RTSP: {width}x{height}, {fps} FPS"))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Successfully connected to RTSP stream"))
            
            # Display a preview frame
            preview = cv2.resize(frame, (self.video_display_width, self.video_display_height))
            preview_tk = convert_cv2_to_tkinter(preview)
            
            self.root.after(0, lambda: self._update_preview(preview_tk))
            
            # Store the successful URL
            self.rtsp_url = rtsp_url
            
            # Release the capture
            cap.release()
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"RTSP connection error: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("RTSP connection error"))
    
    def _update_preview(self, preview_image):
        """Update the video canvas with a preview image"""
        self.video_canvas.create_image(0, 0, anchor="nw", image=preview_image)
        self.video_canvas.image = preview_image  # Keep a reference
    
    def update_pitch_background(self):
        if hasattr(self, 'pitch_image'):
            # Get canvas dimensions
            canvas_width = self.pitch_canvas.winfo_width()
            canvas_height = self.pitch_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Resize image to fit canvas
                resized_image = self.pitch_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                self.tk_pitch_image = ImageTk.PhotoImage(resized_image)
                
                # Update canvas
                self.pitch_canvas.create_image(0, 0, anchor="nw", image=self.tk_pitch_image)
        
        # Schedule this method to run again after the window is fully updated
        self.root.after(100, self.update_pitch_background)
    
    def create_default_pitch(self):
        # Create a green pitch with white lines if no pitch.jpg is available
        width = self.pitch_canvas.winfo_width()
        height = self.pitch_canvas.winfo_height()
        
        if width > 1 and height > 1:
            # Create a green field
            self.pitch_canvas.create_rectangle(0, 0, width, height, fill="green", outline="")
            
            # Draw white lines
            # Field outline
            self.pitch_canvas.create_rectangle(50, 50, width-50, height-50, outline="white", width=2)
            
            # Center line
            self.pitch_canvas.create_line(width/2, 50, width/2, height-50, fill="white", width=2)
            
            # Center circle
            self.pitch_canvas.create_oval(width/2-50, height/2-50, width/2+50, height/2+50, outline="white", width=2)
            
            # Goal areas
            # Left goal
            self.pitch_canvas.create_rectangle(50, height/2-75, 125, height/2+75, outline="white", width=2)
            # Right goal
            self.pitch_canvas.create_rectangle(width-125, height/2-75, width-50, height/2+75, outline="white", width=2)
    
    def select_engine(self):
        engine_path = filedialog.askopenfilename(
            title="Select TensorRT Engine",
            filetypes=[("TensorRT Engine", "*.engine"), ("All Files", "*.*")]
        )
        if engine_path:
            self.engine_path = engine_path
            self.engine_label.config(text=os.path.basename(engine_path))
            
            # Try to load the engine to check compatibility
            try:
                temp_engine = DetectionEngine(engine_path)
                engine_info = f"Engine dimensions: {temp_engine.input_width}x{temp_engine.input_height}"
                self.engine_info_label.config(text=engine_info)
                
                del temp_engine  # Clean up
            except Exception as e:
                self.engine_info_label.config(text=f"Engine error: {str(e)}")
    def select_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")]
        )
        if video_path:
            self.video_path = video_path
            self.video_label.config(text=os.path.basename(video_path))
            
            # Try to open the video to check compatibility
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.status_var.set(f"Video: {width}x{height}, {fps} FPS, {frames} frames")
                    
                    # Show a preview frame
                    ret, frame = cap.read()
                    if ret:
                        preview = cv2.resize(frame, (self.video_display_width, self.video_display_height))
                        preview_tk = convert_cv2_to_tkinter(preview)
                        self.video_canvas.create_image(0, 0, anchor="nw", image=preview_tk)
                        self.video_canvas.image = preview_tk  # Keep a reference
                        
                cap.release()
            except Exception as e:
                self.status_var.set(f"Video error: {str(e)}")
    
    def start_analysis(self):
        # Validate selections
        if not self.engine_path or not os.path.exists(self.engine_path):
            messagebox.showerror("Error", "Please select a valid TensorRT engine file")
            return
        
        # Check video source
        video_source = self.video_source.get()
        if video_source == "file":
            if not self.video_path or not os.path.exists(self.video_path):
                messagebox.showerror("Error", "Please select a valid video file")
                return
        else:  # rtsp
            self.rtsp_url = self.rtsp_entry.get().strip()
            if not self.rtsp_url:
                messagebox.showerror("Error", "Please enter an RTSP URL")
                return
        
        # Disable start button and enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.engine_button.config(state="disabled")
        self.video_button.config(state="disabled")
        self.rtsp_entry.config(state="disabled")
        self.rtsp_test_button.config(state="disabled")
        
        # Set running flag
        self.is_running = True
        self.status_var.set("Initializing...")
        
        # Initialize components
        try:
            # Initialize detection engine
            self.detection_engine = DetectionEngine(self.engine_path)
            
            # Update engine info
            engine_info = f"Engine dimensions: {self.detection_engine.input_width}x{self.detection_engine.input_height}"
            self.engine_info_label.config(text=engine_info)
            
            # Update detection parameters from UI
            self.detection_engine.min_ball_size = self.min_ball_size.get()
            self.detection_engine.ball_confidence_threshold = self.ball_confidence_threshold.get()
            
            # Initialize match analyzer
            self.match_analyzer = MatchAnalyzer()
            
            # Initialize pitch visualizer
            self.pitch_visualizer = PitchVisualizer(self.pitch_canvas)
            
            # Start threads
            self.start_threads()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
            self.stop_analysis()
    
    def start_threads(self):
        # Start video reading thread
        self.video_thread = threading.Thread(target=self.video_reader_thread)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_thread_func)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analysis_thread_func)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        # Start display updater
        self.root.after(30, self.update_display)
        
        self.status_var.set("Analysis running...")

    def video_reader_thread(self):
        """Thread for reading frames from the video source"""
        try:
            # Determine video source
            video_source = self.video_source.get()
            
            if video_source == "file":
                # Open the video file
                self.video_capture = cv2.VideoCapture(self.video_path)
                source_name = os.path.basename(self.video_path)
            else:  # rtsp
                # Open the RTSP stream
                self.video_capture = cv2.VideoCapture(self.rtsp_url)
                source_name = "RTSP Stream"
            
            if not self.video_capture.isOpened():
                raise ValueError(f"Failed to open video source: {source_name}")
            
            # Get video properties
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            
            # For RTSP streams, FPS might not be available or accurate
            if video_source == "rtsp" and (fps <= 0 or fps > 60):
                fps = 25  # Default to 25 FPS for RTSP
            
            # For file, get total frames
            if video_source == "file":
                total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video opened: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
            else:
                total_frames = float('inf')  # Stream has unlimited frames
                print(f"Stream opened: {frame_width}x{frame_height}, using {fps} FPS")
            
            frame_count = 0
            processing_start_time = time.time()
            
            while self.is_running:
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret:
                    if video_source == "file":
                        # End of video file
                        print("End of video reached")
                        self.status_var.set("End of video reached")
                        break
                    else:
                        # For RTSP, try to reconnect
                        print("RTSP stream connection lost. Attempting to reconnect...")
                        self.status_var.set("RTSP reconnecting...")
                        
                        # Close and reopen the stream
                        self.video_capture.release()
                        time.sleep(2)  # Wait before reconnecting
                        self.video_capture = cv2.VideoCapture(self.rtsp_url)
                        
                        if not self.video_capture.isOpened():
                            print("Failed to reconnect to RTSP stream")
                            self.status_var.set("RTSP reconnection failed")
                            break
                        
                        continue  # Skip this iteration
                
                # Put frame in queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_count, frame))
                else:
                    # If queue is full, wait a bit
                    time.sleep(0.01)
                
                # Update frame count
                frame_count += 1
                
                # For file source, control processing speed to match video FPS
                if video_source == "file":
                    elapsed_time = time.time() - processing_start_time
                    expected_time = frame_count / fps
                    
                    if expected_time > elapsed_time:
                        time.sleep(expected_time - elapsed_time)
                
                # Update status every 30 frames
                if frame_count % 30 == 0:
                    current_time = frame_count / fps
                    minutes = int(current_time / 60)
                    seconds = int(current_time % 60)
                    
                    if video_source == "file":
                        self.status_var.set(f"Processing: {minutes:02d}:{seconds:02d} - Frame {frame_count}/{total_frames}")
                    else:
                        self.status_var.set(f"Streaming: {minutes:02d}:{seconds:02d} - Frame {frame_count}")
            
        except Exception as e:
            print(f"Error in video reader thread: {e}")
            self.status_var.set(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up
            if self.video_capture:
                self.video_capture.release()
    def detection_thread_func(self):
        """Thread for running object detection on frames"""
        try:
            while self.is_running:
                # Get frame from queue
                if not self.frame_queue.empty():
                    frame_idx, frame = self.frame_queue.get()
                    
                    # Create goal camera views from the frame
                    h, w = frame.shape[:2]
                    goal_l = frame[:, :w//3].copy()  # Left third of the frame
                    goal_r = frame[:, 2*w//3:].copy()  # Right third of the frame
                    
                    goal_l = cv2.resize(goal_l, (200, 200))
                    goal_r = cv2.resize(goal_r, (200, 200))  # Fixed typo here (was 200, 2)
                    
                    # Create visualization of what's being fed to the engine
                    debug_visualization = self.detection_engine.visualize_preprocessed_input(frame)
                    
                    # Set detection parameters from UI (in case they were updated)
                    self.detection_engine.min_ball_size = self.min_ball_size.get()
                    self.detection_engine.ball_confidence_threshold = self.ball_confidence_threshold.get()
                    
                    # Run detection on the frame
                    results = self.detection_engine.detect_objects(frame)
                    
                    # Additional filtering for ball detections to reduce false positives
                    filtered_results = self.filter_ball_detections(results)
                    
                    # Update detection stats
                    stats = self.detection_engine.get_stats()
                    self.detection_stats_var.set(
                        f"Processed: {stats['detection_count']} frames | "
                        f"Avg time: {stats['avg_detection_time']:.3f}s | "
                        f"Engine: {stats['engine_dimensions']}"
                    )
                    
                    # Put results in queue
                    if not self.result_queue.full():
                        self.result_queue.put({
                            'frame_idx': frame_idx,
                            'frame': frame,
                            'goal_l': goal_l,
                            'goal_r': goal_r,
                            'debug_frame': debug_visualization,
                            'results': filtered_results
                        })
                else:
                    # If queue is empty, wait a bit
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Error in detection thread: {e}")
            self.status_var.set(f"Detection error: {str(e)}")
            import traceback
            traceback.print_exc()
    def filter_ball_detections(self, detection_results):
        """
        Additional filtering to reduce false positives in ball detection
        """
        if not detection_results:
            return []
        
        filtered_results = []
        ball_detections = []
        
        # First, separate balls from other detections
        for obj in detection_results:
            if obj['class'] == 'ball':
                ball_detections.append(obj)
            else:
                filtered_results.append(obj)
        
        # If we have multiple ball detections, keep only the most confident one
        if len(ball_detections) > 1:
            # Sort by confidence (highest first)
            ball_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Check if the highest confidence ball is significantly more confident than others
            highest_conf = ball_detections[0]['confidence']
            second_highest = ball_detections[1]['confidence']
            
            # If there's a significant confidence gap, only keep the highest
            if highest_conf > second_highest * 1.5:  # 50% more confident
                filtered_results.append(ball_detections[0])
            else:
                # Check if balls are of reasonable circular shape
                best_ball = None
                best_circularity = 0
                
                for ball in ball_detections[:2]:  # Check only top 2 candidates
                    # Calculate circularity (1.0 is perfect circle)
                    width = ball['width']
                    height = ball['height']
                    aspect_ratio = min(width, height) / max(width, height)
                    circularity = aspect_ratio  # Better approximation would use contour analysis
                    
                    if circularity > best_circularity:
                        best_circularity = circularity
                        best_ball = ball
                
                # If we found a reasonably circular ball, keep it
                if best_ball and best_circularity > 0.7:  # At least 70% circular
                    filtered_results.append(best_ball)
                else:
                    # If no good circle, just keep the highest confidence one but mark it
                    highest_ball = ball_detections[0]
                    highest_ball['uncertain'] = True
                    filtered_results.append(highest_ball)
        else:
            # If only one ball detection, check its shape
            if ball_detections:
                ball = ball_detections[0]
                width = ball['width']
                height = ball['height']
                aspect_ratio = min(width, height) / max(width, height)
                
                # If it's reasonably circular, keep it
                if aspect_ratio > 0.7:
                    filtered_results.append(ball)
                elif ball['confidence'] > 0.6:  # If very confident despite shape, keep but mark
                    ball['uncertain'] = True
                    filtered_results.append(ball)
                # Otherwise it's filtered out
        
        return filtered_results
    
    def analysis_thread_func(self):
        """Thread for analyzing detection results"""
        try:
            while self.is_running:
                # Get results from queue
                if not self.result_queue.empty():
                    result_data = self.result_queue.get()
                    
                    # Analyze the detection results
                    analysis_results = self.match_analyzer.analyze_frame(
                        result_data['frame_idx'],
                        result_data['results']
                    )
                    
                    # Update visualizations
                    self.current_frame = {
                        'frame': result_data['frame'],
                        'goal_l': result_data['goal_l'],
                        'goal_r': result_data['goal_r'],
                        'debug_frame': result_data.get('debug_frame'),
                        'results': result_data['results'],
                        'analysis': analysis_results
                    }
                else:
                    # If queue is empty, wait a bit
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"Error in analysis thread: {e}")
            self.status_var.set(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()

    def draw_detections_on_frame(self, frame, detection_results):
        """Draw detection boxes and labels on the frame"""
        # Make a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Draw the detection results
        if detection_results:
            for obj in detection_results:
                if obj['class'] == 'ball':
                    # Draw ball as a red circle
                    center = (int(obj['center_x']), int(obj['center_y']))
                    radius = int(obj['width'] / 2)
                    
                    # Check if the ball detection is uncertain
                    if obj.get('uncertain', False):
                        # Draw uncertain ball with dashed circle in orange
                        color = (0, 165, 255)  # Orange
                        # Draw dashed circle (approximation)
                        for i in range(0, 360, 30):
                            angle1 = i * np.pi / 180
                            angle2 = (i + 15) * np.pi / 180
                            pt1 = (int(center[0] + radius * np.cos(angle1)), 
                                   int(center[1] + radius * np.sin(angle1)))
                            pt2 = (int(center[0] + radius * np.cos(angle2)), 
                                   int(center[1] + radius * np.sin(angle2)))
                            cv2.line(display_frame, pt1, pt2, color, 2)
                    else:
                        # Draw confirmed ball as solid red circle
                        cv2.circle(display_frame, center, radius, (0, 0, 255), -1)
                        cv2.circle(display_frame, center, radius+1, (255, 255, 255), 1)
                    
                    # Add class label and confidence text
                    confidence_str = f"{obj['confidence']:.2f}"
                    label_text = f"Ball: {confidence_str}"
                    if obj.get('uncertain', False):
                        label_text = f"Ball? {confidence_str}"
                    
                    # Position the text above the ball
                    text_x = center[0] - 20
                    text_y = center[1] - radius - 10
                    
                    # Add white background for better visibility
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        display_frame,
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0], text_y + 5),
                        (255, 255, 255),
                        -1
                    )
                    
                    # Add text
                    cv2.putText(
                        display_frame, 
                        label_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 255) if not obj.get('uncertain', False) else (0, 165, 255),
                        2
                    )
                elif obj['class'] == 'player':
                    # Draw player bounding box
                    x1, y1 = int(obj['x1']), int(obj['y1'])
                    x2, y2 = int(obj['x2']), int(obj['y2'])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare label text
                    if 'player_id' in obj:
                        label_text = f"Player #{obj['player_id']}: {obj['confidence']:.2f}"
                    else:
                        label_text = f"Player: {obj['confidence']:.2f}"
                    
                    # Add white background for better visibility
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        display_frame,
                        (x1, y1 - text_size[1] - 5),
                        (x1 + text_size[0], y1),
                        (255, 255, 255),
                        -1
                    )
                    
                    # Add text
                    cv2.putText(
                        display_frame, 
                        label_text,
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )
        
        # Add detection stats to frame
        if self.detection_engine:
            stats = self.detection_engine.get_stats()
            count = stats.get('detection_count', 0)
            time_ms = stats.get('avg_detection_time', 0) * 1000
            
            # Add stats to top of frame
            cv2.putText(display_frame, f"Detections: {count} | Avg time: {time_ms:.1f}ms", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    

    def update_display(self):
        """Update the UI with the latest processed frame and analysis results"""
        if self.current_frame and self.is_running:
            try:
                # Get current frame data
                frame = self.current_frame['frame']
                goal_l = self.current_frame['goal_l']
                goal_r = self.current_frame['goal_r']
                debug_frame = self.current_frame.get('debug_frame')
                results = self.current_frame['results']
                analysis = self.current_frame['analysis']
                
                # Display the main video frame
                display_frame = self.draw_detections_on_frame(frame, results)
                
                # Resize the frame to fit the fixed canvas size
                display_frame = cv2.resize(display_frame, 
                                          (self.video_display_width, self.video_display_height), 
                                          interpolation=cv2.INTER_AREA)
                
                video_tk = convert_cv2_to_tkinter(display_frame)
                self.video_canvas.create_image(0, 0, anchor="nw", image=video_tk)
                self.video_canvas.image = video_tk  # Keep a reference
                
                # Display goal frames
                goal_l_tk = convert_cv2_to_tkinter(goal_l)
                self.left_goal_canvas.create_image(0, 0, anchor="nw", image=goal_l_tk)
                self.left_goal_canvas.image = goal_l_tk  # Keep a reference
                
                goal_r_tk = convert_cv2_to_tkinter(goal_r)
                self.right_goal_canvas.create_image(0, 0, anchor="nw", image=goal_r_tk)
                self.right_goal_canvas.image = goal_r_tk  # Keep a reference
                
                # Update pitch visualization with analysis results
                if self.pitch_visualizer:
                    self.pitch_visualizer.update(analysis)
                
            except Exception as e:
                print(f"Error updating display: {e}")
                import traceback
                traceback.print_exc()
        
        # Schedule next update
        if self.is_running:
            self.root.after(30, self.update_display)

# Main function to run the application standalone
def main():
    """Main function to run the application"""
    # Parse command-line arguments if needed
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Soccer Referee App')
        parser.add_argument('--engine', help='Path to TensorRT engine file')
        parser.add_argument('--video', help='Path to video file')
        parser.add_argument('--rtsp', help='RTSP URL for camera stream')
        parser.add_argument('--min-ball-size', type=int, default=6, help='Minimum ball size in pixels')
        parser.add_argument('--ball-confidence', type=float, default=0.4, help='Ball confidence threshold (0.0-1.0)')
        args = parser.parse_args()
    except:
        args = None
    
    root = tk.Tk()
    app = SoccerRefereeApp(root)
    
    # Set initial values from command line if provided
    if args:
        if args.engine and os.path.exists(args.engine):
            app.engine_path = args.engine
            app.engine_label.config(text=os.path.basename(args.engine))
        
        if args.video and os.path.exists(args.video):
            app.video_path = args.video
            app.video_label.config(text=os.path.basename(args.video))
            app.video_source.set("file")
        
        if args.rtsp:
            app.rtsp_url = args.rtsp
            app.rtsp_entry.delete(0, tk.END)
            app.rtsp_entry.insert(0, args.rtsp)
            app.video_source.set("rtsp")
        
        # Set detection parameters if provided
        if args.min_ball_size > 0:
            app.min_ball_size.set(args.min_ball_size)
        
        if 0.0 <= args.ball_confidence <= 1.0:
            app.ball_confidence_threshold.set(args.ball_confidence)
    
    # Start Tkinter main loop
    root.mainloop()

# Run the application if this module is executed directly
if __name__ == "__main__":
    main()