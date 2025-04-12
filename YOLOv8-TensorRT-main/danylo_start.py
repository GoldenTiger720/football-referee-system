#!/usr/bin/env python3
"""
Soccer Match Automatic Referee System
------------------------------------
A system for analyzing soccer videos and detecting players, ball, and events.

Usage:
    python main.py [--engine ENGINE_PATH] [--video VIDEO_PATH] [--rtsp RTSP_URL]
"""

import tkinter as tk
import os
import sys
import argparse

# Add parent directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import application
from danylo_soccer_referee_app import SoccerRefereeApp

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Soccer Match Automatic Referee System'
    )
    
    parser.add_argument(
        '--engine',
        type=str,
        help='Path to TensorRT engine file'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file for analysis'
    )
    
    parser.add_argument(
        '--rtsp',
        type=str,
        help='RTSP URL for camera stream'
    )
    
    parser.add_argument(
        '--pitch',
        type=str,
        default='pitch.jpg',
        help='Path to pitch image file'
    )
    
    parser.add_argument(
        '--min-ball-size',
        type=int,
        default=6,
        help='Minimum ball size in pixels'
    )
    
    parser.add_argument(
        '--ball-confidence',
        type=float,
        default=0.4,
        help='Ball detection confidence threshold (0.0-1.0)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create Tkinter root window
    root = tk.Tk()
    root.title("Soccer Match Automatic Referee System")
    
    # Create application instance
    app = SoccerRefereeApp(root)
    
    # Set initial values from command line arguments
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
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()