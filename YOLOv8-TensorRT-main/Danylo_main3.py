import time
from pathlib import Path
import numpy as np
import cv2
import csv
import logging
import os
import queue
import argparse
from models import TRTModule  # isort:skip
import proto_messages.players_pb2 as players_pb2

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import threading
import multiprocessing as mp
from src.danylo_aidetector import *
from src.danylo_detection_engine import DetectionEngine
from src.utils import *
from src.danylo_camera import *
from src.ball3 import *
from src.aisettings import *
from src.playeraction import *
from src.player_tracker import *
import requests
from datetime import datetime
from collections import deque
from src.player_analyzer.playeranalyzer import *

import psutil
#from playsound import playsound

# Load the WAV file
#filename = 'kick.wav'
#data, fs = sf.read(filename, dtype='float32')

#start_index = int(fs * 0.25)  # 0.5 seconds * sample rate


# Define a function to play the sound
#def play_sound():
    #playsound('kick.wav', block = False,)
    #sd.play(data[start_index:], fs)
    #sd.wait()  # Wait until the sound has finished playing

def get_performance_cores():
    # This is a placeholder function; replace with actual core identification logic
    # Example: Assume cores 0-7 are performance cores
    return range(8)

performance_cores = get_performance_cores()

# Set the current process to use performance cores
p = psutil.Process(os.getpid())
try:
    p.cpu_affinity(performance_cores)
except:
    print("Warning: Failed to set CPU affinity")

player_publisher = None

# Initialize the deque with a maximum length of 10
GOAL_HOLDBACK_FRAME_CNTR = 30
BALL_QUEUE_LEN=GOAL_HOLDBACK_FRAME_CNTR * 2
ball_pos_history = deque(maxlen=BALL_QUEUE_LEN)

FEED_FPS = 25
MAX_FPS = 25
PARALEL_MODELS = 3
VIDEO_SHIFT_SEC = 7

player_tracker = PlayerTracker()
analyzer = PlayerAnalyzer("yolov8m-seg.engine", "yolov8m-pose.engine", "tracker_imgs")
currentAction = CurrentAction()
displayQueue = mp.Queue()
feedbackQueue = mp.Queue()
thread_running = mp.Value('i', 1)
frame_start = mp.Value('i', 0)
cameras =[]
score_cntr_l=0
score_cntr_r=0
match_time="00:00"
previous_ball_speed_kmh=-1
kick_detected = False
kick_ttl=0
last_ball_direction = 0
#detector_live_cycle = mp.Value('i', -1)
last_valid_ball_x = 0
last_valid_ball_y = 0
prev_acceleration=-1

detector_cycle_live = 0
detector_cycle_processing = 0
overall_stat = OverallStats()
rpc_publisher = None
last_audio_notification=time.time()-15

ai_settings = None  # Will be initialized later in main()
CAMERA_ID="accueil"

seg_model = YOLO("yolov8m-seg.engine")
pose_model = YOLO("yolov8m-pose.engine")
conf_threshold=0.25


def send_camera_event(camera_id, event_type, duration):
    global last_audio_notification
    current_time = datetime.now().strftime("%H:%M:%S")
    print("["+ current_time + "] SENDING CAMERA EVENT", camera_id, event_type, duration)
    #return
    url = "https://app.backendsportunity2017.com/devices-hooks/camera-event"
    #url = "http://127.0.0.1:8080/devices-hooks/camera-event"
    query_params = {
        "camera_id": camera_id,
        "event_type": event_type,
        "duration": duration
    }

    if (time.time() - last_audio_notification > 10 or duration==0):
      response = requests.get(url, params=query_params)
      last_audio_notification = time.time()

      if response.status_code == 200:
          print("Event sent successfully.", query_params, url)
      else:
          print(f"Error sending event. Status code: {response.status_code}")
    else:
       print("  notification not sent due to another active notification from the last 10 sec")

def process_detection_results(publisher, detector, cycle):
    global score_cntr_l, score_cntr_r, match_time, previous_ball_speed_kmh, kick_detected, kick_ttl, last_ball_direction, currentAction, last_valid_ball_x, last_valid_ball_y, prev_acceleration, player_publisher, ai_settings
    detector_selected_cam = detector.best_camera
    detector_process_time = detector.last_proc_time
    detector_broadcast_cam = detector.broadcast_camera

    print("Last detector cycle:", detector.detector_cycle)
    print("Selected camera:", detector_selected_cam)
    print("Process time:", detector_process_time)

    # Initialize CSV writer for player positions - ADDED CODE
    from player_position_collector import collect_player_positions, save_player_position
    
    # Create the CSV filename with timestamp to ensure uniqueness
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    player_csv_path = os.path.join(ai_settings.video_out_folder, f'player_positions_{timestamp_str}.csv')
    csv_writer, timestamp = collect_player_positions(player_csv_path)
    
    print(f"Logging player positions to: {player_csv_path}")
    # END OF ADDED CODE

    other_cam = -1
    if detector_selected_cam==3:
        other_cam=2
    if detector_selected_cam==2:
        other_cam=3

    cam_selected=None
    cam_other = None

    ball_coords = BallCoords()
    multiple_balls = 0
    for cam in cameras:
        if cam.camera_id ==detector_selected_cam:
            #detector_objs = cam.get_detector_collection(detector_cycle_live-1, detection_type=DetectionType.GOAL)
            detector_objs = cam.get_detector_collection(cycle)
            for det_obj in detector_objs:
                if det_obj is not None:
                    if det_obj.ball>1:
                        multiple_balls+=1
                    if det_obj.x1==-1:
                        continue
                    cent_x, cent_y = (det_obj.x1 + det_obj.x2) // 2, (det_obj.y1 + det_obj.y2) // 2
                    radius = min(det_obj.x2 - det_obj.x1, det_obj.y2 - det_obj.y1) // 2                    
                    
                    moving_average_speed = calculate_moving_average(speed_queue, det_obj.ball_speed_kmh)

                    v1_ms = previous_ball_speed_kmh * (1000 / 3600)
                    v2_ms = moving_average_speed * (1000 / 3600)

                    # Calculate the change in velocity in m/s
                    delta_v_ms = v2_ms - v1_ms

                    # Define the time difference in seconds
                    delta_t_s = 0.04  # 40 milliseconds = 0.04 seconds

                    # Calculate the acceleration in m/s^2
                    acceleration = int(delta_v_ms / delta_t_s)
                    moving_average_acceleration = int(calculate_moving_average(acceleration_queue, acceleration))
                    if moving_average_acceleration>-1 and moving_average_acceleration<=1:
                        moving_average_acceleration=0

                    dy = det_obj.y1- last_valid_ball_y
                    dx = det_obj.x1 - last_valid_ball_x
                    angle_radians = math.atan2(dy, dx)
                    # Convert radians to degrees
                    angle_degrees = math.degrees(angle_radians)
                    angle_degrees = (angle_degrees + 360) % 360

                    det_obj.ball_direction = int(angle_degrees)
                    last_valid_ball_y = det_obj.y1
                    last_valid_ball_x = det_obj.x1

                    kicked=False
                    
                    previous_ball_speed_kmh = moving_average_speed                    
                    det_obj.ball_speed_kmh = moving_average_speed

                    det_obj.ball_direction_change=False
                    if det_obj.ball_direction>0:
                        direction_delta = abs(last_ball_direction - det_obj.ball_direction)

                        direction_delta = min(direction_delta, abs(360-direction_delta))
                        
                        if direction_delta>80 and moving_average_speed>5:
                            det_obj.ball_direction_change=True
                            kick_detected=False #reset previous kick as direction chnaged
                            print("KICK DETECTION SET TO FALSE - DIRECTION", cycle, det_obj.position, last_ball_direction,  det_obj.ball_direction, direction_delta)
                        
                        last_ball_direction = det_obj.ball_direction

                    if moving_average_acceleration<2 and moving_average_speed<2:
                        kick_detected=False
                        print("KICK DETECTION SET TO FALSE - ACCEL", cycle, det_obj.position, moving_average_acceleration,  moving_average_speed)
                    
                    if (moving_average_acceleration>3 and kick_detected==False) or (moving_average_acceleration>6 and (moving_average_acceleration-prev_acceleration)>5 and moving_average_speed<26):
                            kick_detected=True
                            kicked = True
                            currentAction.start_action(det_obj.x_2d, det_obj.y_2d, match_time, cycle, det_obj.position)
                            #ballkicked_cntr=3
                            #play_sound()

                    currentAction.add_speed(moving_average_speed)
                    currentAction.add_acceleration(moving_average_acceleration)
                    currentAction.add_score(score_cntr_l, score_cntr_r)
                    k_str='Off'
                    if kick_detected==True:
                        k_str=f'in Kick ->{kicked}|{currentAction.get_current_id()}'
                    else:
                        k_str=f'Not in kick ->{kicked}|{currentAction.get_current_id()}'
                        currentAction.stop_action(det_obj.x_2d, det_obj.y_2d, cycle, det_obj.position)

                    prev_acceleration = moving_average_acceleration

                    ball_data = BallCoordsObj(cam.camera_id, det_obj.x_2d, det_obj.y_2d,int(cent_x), int(cent_y), 
                                              int(radius),det_obj.ball_speed_kmh,det_obj.position, det_obj.ball_direction,
                                              moving_average_acceleration, kick_detected, kicked, k_str, det_obj.ball_direction_change)
                    ball_coords.data.append(ball_data)
                    currentAction.add_ball(det_obj.x_2d, det_obj.y_2d)    

    json_ball_coords = json.dumps([obj.to_dict() for obj in ball_coords.data])

    if publisher:
        publisher.send(camera_selection(cycle, detector_broadcast_cam, score_cntr_l, score_cntr_r, match_time, json_ball_coords))

    detector_selected_cam = detector.best_camera
    detector_process_time = detector.last_proc_time

    stat = DetectorStats()
    stat.detection_cycle = cycle
    stat.selected_cam = detector_selected_cam
    stat.process_time = detector_process_time

    overall_stat.cycle_cntr+=1

    for cam in cameras:
        if cam.capture_only_device ==False:
            cam_info = CamInfo()
            cam_info.id = cam.camera_id
            cam_info.name = cam.description
            cam_info.total_frames = 0
            cam_info.total_detections = 0
            detector_objs = cam.get_detector_collection(cycle, cam.detection_type)
            people = 0
            for det_obj in detector_objs:
                if det_obj is not None and det_obj.fake==False:
                    cam_info.total_frames += 1
                    stat.total_frames += 1
                    if det_obj.processed:
                        stat.total_detections +=1
                        cam_info.total_detections += 1
                        people += det_obj.people
                        if det_obj.confidence >0.2:
                            cam_info.ball_detected_frames += 1
                            stat.ball_detected_frames += 1
            
            if cam_info.total_detections>0:
                cam_info.players = round(people / cam_info.total_detections)
            else:
                cam_info.players = 0 
            stat.cam_info.append(cam_info)
    if stat.ball_detected_frames>1:
        overall_stat.ball_captured+=1

    overall_stat.detection_rate = round((overall_stat.ball_captured / overall_stat.cycle_cntr)*100)
    stat.overall_detection_rate = overall_stat.detection_rate

    players_collection = None
    players_collection_ttl=0

    goal_l_cam=None
    goal_r_cam=None
    for cam in cameras:
        if cam.camera_id ==11:
            goal_l_cam=cam
        if cam.camera_id ==10:
            goal_r_cam=cam

    gameplayer_collection = PlayerCollection()

#REMOVE IT!!!!!!!!!!!
    detector_selected_cam = 2
################    
    ################
    ################
    ################
    for cam in cameras:
        if cam.camera_id ==detector_selected_cam:
            #detector_objs = cam.get_detector_collection(detector_cycle_live-1, detection_type=DetectionType.GOAL)
            detector_objs = cam.get_detector_collection(cycle)
            for det_obj in detector_objs:
                if det_obj is not None:
                    #ball_data = BallCoordsObj(cam.camera_id, det_obj.x_2d, det_obj.y_2d, det_obj.position)
                    #ball_coords.data.append(ball_data)                                        
                    frame = det_obj.frame.copy()
                    #frame = cv2.resize(frame, (640, 384), interpolation=cv2.INTER_AREA)

                    frame_goal_l = goal_l_cam.get_goal_frame(cycle, det_obj.position) if goal_l_cam else None
                    frame_goal_r = goal_r_cam.get_goal_frame(cycle, det_obj.position) if goal_r_cam else None
                    
                    #frame_cpy = frame.copy()
                    txt1=f'{cycle} | CAMERA{detector_selected_cam}'
                    cv2.putText(frame,txt1, (50,55),cv2.FONT_HERSHEY_PLAIN,2, [0,0,100],thickness=2)
                    cv2.putText(frame,txt1, (51,56),cv2.FONT_HERSHEY_PLAIN,2, [0,255,0],thickness=2)
                    rad = min(det_obj.x2 - det_obj.x1, det_obj.y2, det_obj.y1)
                    txt=f'{det_obj.x1} | {det_obj.y1} | r={rad} | {det_obj.confidence} | {det_obj.position}'
                    cv2.rectangle(frame,(350,0),(950,50), (0,0,0), -1)
                    cv2.putText(frame,
                                    txt, (350,40),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    2, [0,255,255],
                                    thickness=2)
                                        
                    try:
                        print("::::::",cycle, cam.camera_id, det_obj.x1, det_obj.y1,det_obj.x2, det_obj.y2,det_obj.confidence, int(det_obj.x_2d), int(det_obj.y_2d), det_obj.processed, 
                                det_obj.skip, det_obj.position, "isfake:", det_obj.fake, "color:", int(det_obj.mean_saturation),int(det_obj.mean_value), "->speed:", det_obj.ball_speed_ms, "m/s", 
                                det_obj.ball_speed_kmh, "km/h", "dir:", det_obj.ball_direction)
                    except:
                        pass 
                    ball_point_x=-1
                    ball_point_y = -1
                    ball_point_x_2d=-1
                    ball_point_y_2d = -1                    
                    ball_radius = 0
                    if (det_obj.x1>0):
                        #cv2.rectangle(frame, (int(det_obj.x1), int(det_obj.y1)), (int(det_obj.x2), int(det_obj.y2)), (0,0,255), 2) 
                        center_x, center_y = (det_obj.x1 + det_obj.x2) // 2, (det_obj.y1 + det_obj.y2) // 2
                        radius = min(det_obj.x2 - det_obj.x1, det_obj.y2 - det_obj.y1) // 2
                        ball_radius = radius
                        # Draw a filled circle
                        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), -1)
                        cv2.circle(frame, (center_x, center_y), radius+1, (255, 0, 0), 1)
                        cv2.circle(frame, (center_x, center_y), radius+2, (0, 0, 0), 1)

                        ball_point_x =  (det_obj.x1+det_obj.x2)/2
                        ball_point_y =  (det_obj.y1+det_obj.y2)/2
                        ball_point_x_2d = det_obj.x_2d
                        ball_point_y_2d = det_obj.y_2d

                    if len(det_obj.players)>0:
                        players_collection = det_obj.players
                        players_collection_ttl = 3
                    else:
                        players_collection_ttl -=1

                    if players_collection_ttl>0:
                        closest_player=None
                        closest_distance=None
                        closet_player_width = None
                        distance=-1
                        player_collection = None
                        if player_publisher:
                            player_collection = players_pb2.FootPlayerCollection()
                            player_collection.cycle = cycle
                            player_collection.position = det_obj.position
                            player_collection.img = frame.tobytes()
                            height, width = frame.shape[:2]

                            player_collection.img_w = width
                            player_collection.img_h = height
                            
                            
                            player_collection.ball_x = int(ball_point_x)
                            player_collection.ball_y = int(ball_point_y)
                            player_collection.ball_radius = int(ball_radius)
                            player_collection.ball_x_2d = int(ball_point_x_2d)
                            player_collection.ball_y_2d = int(ball_point_y_2d)
                        
                        for player in players_collection:
                            if (player.x1!=-1):
                                player.x_2d, player.y_2d = cam.convert_player_to_2d(int((player.x1+player.x2)/2), int(player.y2)-15)
                                bounding_img = det_obj.frame[player.y1:player.y2,player.x1:player.x2]
                                player.player_id = player_tracker.register(bounding_img, player, cycle, det_obj.position)
                                
                                # Log player position - ADDED CODE
                                save_player_position(csv_writer, timestamp, cycle, det_obj.position, player)
                                # END OF ADDED CODE

                                if player_collection:
                                    height, width = bounding_img.shape[:2]
                                    foot_player = player_collection.players.add()
                                    foot_player.x1 = player.x1
                                    foot_player.x2 = player.x2
                                    foot_player.y1 = player.y1
                                    foot_player.y2 = player.y2
                                    foot_player.x_2d = player.x_2d
                                    foot_player.y_2d = player.y_2d
                                    foot_player.confidence = int(player.confidence*100)
                                    foot_player.img_w = width
                                    foot_player.img_h = height
                                    foot_player.img = bounding_img.tobytes()

                                if ball_point_x!=-1:
                                    player_point_x =  (player.x1+player.x2)/2
                                    player_point_y =  (player.y2+player.y2)/2
                                    distance_1 = distance_between_points(ball_point_x, ball_point_y, player_point_x, player_point_y)
                                    distance_2 = distance_between_points(ball_point_x, ball_point_y, player.x1, player_point_y)
                                    distance_3 = distance_between_points(ball_point_x, ball_point_y, player.x2, player_point_y)

                                    distance = min(distance_1, distance_2, distance_3)

                                    if (closest_distance is None or distance < closest_distance):
                                        closest_distance = distance
                                        closest_player = player
                                        closet_player_width = player.x2 - player.x1

                                cv2.rectangle(frame, (int(player.x1), int(player.y1)), (int(player.x2), int(player.y2)), (255,0,0), 2)
                                player.cam_id = cam.camera_id
                                #player.x_2d, player.y_2d = cam.convert_player_to_2d(int((player.x1+player.x2)/2), int(player.y2)-15)
                                player.ball_distance = distance
                                player.ball_x = ball_point_x
                                player.ball_y = ball_point_y

                                if (player.x_2d>0 and player.y_2d <0):
                                    player.y_2d=0

                                gameplayer_collection.add_to_collection(det_obj.position, player)

                        if player_publisher and player_collection:
                            player_publisher.send(player_collection)
                            
                        if closest_player is not None and closest_distance<(closest_player.x2-closest_player.x1)*1.5:
                            bounding_img = det_obj.frame[closest_player.y1:closest_player.y2,closest_player.x1:closest_player.x2]
                            currentAction.add_player(0,bounding_img, cycle, det_obj.position)
                            #currentAction.add_player(0,None, cycle, det_obj.position)
                            wwidth=2
                            display_rect=True
                            ccolor = (0,255,0)
                            ball_obj = ball_coords.get_ball_data_by_position(det_obj.position)
                            if ball_obj is not None and ball_obj.kicked==True:
                                wwidth=10
                                ccolor = (0,0,255)
                            elif ball_obj is not None and ball_obj.kick_detected==True:
                                display_rect=False
                            if display_rect==True:
                                closest_player.closest = True
                                cv2.rectangle(frame, (int(closest_player.x1), int(closest_player.y1)), (int(closest_player.x2), 
                                                                                                    int(closest_player.y2)), ccolor, wwidth)
                        else:
                            ball_obj = ball_coords.get_ball_data_by_position(det_obj.position)
                            if ball_obj is not None:
                                ball_obj.kicked=False

                    all_players = gameplayer_collection.get_collection(det_obj.position)
                    displayQueue.put({'frame':frame,'goal_l':frame_goal_l,'goal_r':frame_goal_r, 'stats':stat, 'ball':ball_coords,
                                      'position':det_obj.position, 'multiball':multiple_balls, 
                                      'players':all_players, 'currentAction':currentAction })
                    
                    cj=0
                    for pl in all_players:
                        cj+=1
                        print(f'[{cj}] 2dx:{pl.x_2d},2dy:{pl.y_2d}, cam:{pl.cam_id} = {pl.y_2d} -> {pl.x1}:{pl.y1} - {pl.x2}:{pl.y2} [w:{pl.x2-pl.x1},h:{pl.y2-pl.y1}] Ball Distance: {pl.ball_distance}, ball x,y: {pl.ball_x}, {pl.ball_y}')

def trigger_camera_event():
    even_type = f'AI_FIELD_{ai_settings.field_id}_ALARM1'
    send_camera_event(CAMERA_ID, even_type, 3)
    time.sleep(2)
    send_camera_event(CAMERA_ID, even_type, 0)

def display_frames(thread_running, displayQueue, feedbackQueue, ai_settings):
    score_cntr_l=0
    score_cntr_r=0 
    match_time="00:00"

    cntr = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Calculate the new height of the output video, considering the pitch image.
    # Here, we calculate the height of the pitch image after scaling it to match the 640 width.

    pitch_new_height = 310
    pitch_width = 538
    output_height = 384 + pitch_new_height+40+100
    #u_time = int(time.perf_counter() * 1000)
    if (ai_settings.video_disabled==False):
        out = cv2.VideoWriter(ai_settings.video_out_folder+f'/output_{ai_settings.recording_id}.avi', fourcc, MAX_FPS, (1100+340, output_height))
       
    
    # Load overlay and pitch images
    try:
        overlay = cv2.imread('score4.png', -1)  # -1 to load with alpha channel
        if overlay is None:
            print("Warning: Could not load score4.png, creating a blank overlay")
            # Create a blank overlay with alpha channel
            overlay = np.zeros((100, 400, 4), dtype=np.uint8)
    except Exception as e:
        print(f"Error loading overlay: {e}")
        # Create a blank overlay with alpha channel
        overlay = np.zeros((100, 400, 4), dtype=np.uint8)

    try:
        pitch = cv2.imread('pitch.jpg')
        if pitch is None:
            print("Warning: Could not load pitch.jpg, creating a blank pitch")
            # Create a blank pitch
            pitch = np.zeros((pitch_new_height, pitch_width, 3), dtype=np.uint8)
            # Draw field lines on blank pitch
            cv2.rectangle(pitch, (0, 0), (pitch_width, pitch_new_height), (0, 128, 0), -1)  # Green background
            cv2.rectangle(pitch, (50, 50), (pitch_width-50, pitch_new_height-50), (255, 255, 255), 2)  # Field outline
            cv2.line(pitch, (pitch_width//2, 50), (pitch_width//2, pitch_new_height-50), (255, 255, 255), 2)  # Center line
    except Exception as e:
        print(f"Error loading pitch: {e}")
        # Create a blank pitch
        pitch = np.zeros((pitch_new_height, pitch_width, 3), dtype=np.uint8)
        # Draw field lines on blank pitch
        cv2.rectangle(pitch, (0, 0), (pitch_width, pitch_new_height), (0, 128, 0), -1)  # Green background
        cv2.rectangle(pitch, (50, 50), (pitch_width-50, pitch_new_height-50), (255, 255, 255), 2)  # Field outline
        cv2.line(pitch, (pitch_width//2, 50), (pitch_width//2, pitch_new_height-50), (255, 255, 255), 2)  # Center line

    # Resize pitch to fit the width of the frame while maintaining its aspect ratio
    pitch = cv2.resize(pitch, (pitch_width, pitch_new_height), interpolation=cv2.INTER_AREA)

    total_frames_captures=0
    start_time = time.perf_counter()
    goal_l = None
    goal_r = None
    
    # Initialize goal checkers
    try:
        goal_config_path = os.path.join(os.path.dirname(ai_settings.feed_config_path), "goal_config.json")
        if not os.path.exists(goal_config_path):
            # Create a basic goal config file
            goal_config = {
                f"Field_{ai_settings.field_id}_Goal_L": {
                    "blur": 7,
                    "ref_img_folder": "ref_img_l/",
                    "ref_frame_folder": "ref_frame_l/",
                    "max_ball_bouncback_speed": 14,
                    "max_ball_negative_acceleration": -30,
                    "goal_in": [[157, 213], [120, 86], [230, 0], [263, 111]],
                    "frame_crop": [[0, 220], [0, 300]],
                    "compare_images_absdiff_lower_threshold": 100,
                    "contour_smoothing_epsilon": 1,
                    "ball_circularity_threshold": 0.85,
                    "ball_color_threshold": 120,
                    "max_difference_between_color_channels": 120,
                    "max_height_width_ratio": 1.2,
                    "max_ball_width": 18,
                    "max_ball_height": 18,
                    "min_ball_area": 90,
                    "max_ball_area": 200,
                    "ball_ssim_threshold": 0.8,
                    "ball_surrounding_ssim_threshold": 0.8,
                    "surrounding_difference_threshold": 20,
                    "surrounding_pixels_x": 15,
                    "surrounding_pixels_y": 15,
                    "skip_goal_check_after_scoring_for_frames": 60,
                    "debug_mode": True
                },
                f"Field_{ai_settings.field_id}_Goal_R": {
                    "blur": 7,
                    "ref_img_folder": "ref_img_r/",
                    "ref_frame_folder": "ref_frame_r/",
                    "max_ball_bouncback_speed": 14,
                    "max_ball_negative_acceleration": -30,
                    "goal_in": [[157, 213], [120, 86], [230, 0], [263, 111]],
                    "frame_crop": [[0, 220], [0, 300]],
                    "compare_images_absdiff_lower_threshold": 100,
                    "contour_smoothing_epsilon": 1,
                    "ball_circularity_threshold": 0.85,
                    "ball_color_threshold": 120,
                    "max_difference_between_color_channels": 120,
                    "max_height_width_ratio": 1.2,
                    "max_ball_width": 18,
                    "max_ball_height": 18,
                    "min_ball_area": 90,
                    "max_ball_area": 200,
                    "ball_ssim_threshold": 0.8,
                    "ball_surrounding_ssim_threshold": 0.8,
                    "surrounding_difference_threshold": 20,
                    "surrounding_pixels_x": 15,
                    "surrounding_pixels_y": 15,
                    "skip_goal_check_after_scoring_for_frames": 60,
                    "debug_mode": True
                }
            }
            os.makedirs(os.path.dirname(goal_config_path), exist_ok=True)
            with open(goal_config_path, 'w') as f:
                json.dump(goal_config, f, indent=4)
            print(f"Created example goal configuration file at {goal_config_path}")
            
        goalcheck_l = GoalChecker(f'Field_{ai_settings.field_id}_Goal_L', goal_config_path, "L")
        goalcheck_r = GoalChecker(f'Field_{ai_settings.field_id}_Goal_R', goal_config_path, "R")
        
        # Create necessary folders for goal reference images
        os.makedirs("ref_img_l", exist_ok=True)
        os.makedirs("ref_img_r", exist_ok=True)
        os.makedirs("ref_frame_l", exist_ok=True)
        os.makedirs("ref_frame_r", exist_ok=True)
        
    except Exception as e:
        print(f"Error initializing goal checkers: {e}")
        # Create dummy goal checkers
        goalcheck_l = None
        goalcheck_r = None
    
    last_ball_x = 0
    last_ball_y = 0
    last_ball_x_real = 0
    last_ball_y_real = 0
    last_ball_ttl = 0
    take_ref_img_l=False
    take_ref_img_r=False

    goal_hold_back_l=-1
    goal_hold_back_r=-1
    goal_holdback_frames = GOAL_HOLDBACK_FRAME_CNTR
    multiball_alert=False
    multiball_alert_cntr=0
    previous_ball_speed_kmh=-1
    previous_ball_acceleration=-1
    previous_ball_acceleration2=-1

    ballkicked_cntr=0
    players_collection = None
    last_ball_direction = 0
    #kick_detected=False

    ball_time_on_left=0
    ball_time_on_right=0
    currentAction = None
    
    # Flag to indicate if we've displayed a frame yet
    frame_displayed = False
    last_frame_time = time.time()
    
    # Create a blank frame to display if no frames are available yet
    blank_frame = np.zeros((output_height, 1100+340, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Waiting for frames...", (450, output_height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Frame', blank_frame)
    cv2.waitKey(1)
    
    while thread_running.value == 1:
        if goal_hold_back_l!=-1:
            goal_hold_back_l-=1
        if goal_hold_back_r!=-1:
            goal_hold_back_r-=1

        overall_elapsed = (time.perf_counter() - start_time) * 1000
        expedcted_elapsed = total_frames_captures * (1000/MAX_FPS)
        wait = expedcted_elapsed  - overall_elapsed
        if wait>0:
            time.sleep(wait / 1000)

        total_frames_captures+=1
        
        # Check if we haven't received any frames for a while
        current_time = time.time()
        if not frame_displayed and current_time - last_frame_time > 5:
            # Display a message to the user
            cv2.putText(blank_frame, f"No frames received for {int(current_time - last_frame_time)} seconds", 
                       (350, output_height//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(blank_frame, "Check if the video file exists and is valid", 
                       (350, output_height//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Frame', blank_frame)
            cv2.waitKey(1)
            last_frame_time = current_time
        
        if not displayQueue.empty():
            frame_displayed = True
            last_frame_time = current_time
            multiball_alert=False
            frame_msg = displayQueue.get()
            frame = frame_msg['frame']
            
            if frame_msg['goal_l'] is not None:
                goal_l = frame_msg['goal_l']
            if frame_msg['goal_r'] is not None:
                goal_r = frame_msg['goal_r']

            if frame_msg['players'] is not None:
                players_collection = frame_msg['players']

            if frame_msg['currentAction'] is not None:
                currentAction = frame_msg['currentAction']

            if frame_msg['multiball'] is not None:
                multiball = frame_msg['multiball']
                if multiball>10:
                    multiball_alert=True
                    multiball_alert_cntr+=1

            stat = frame_msg['stats']
            ball_coords = frame_msg['ball']
            position = frame_msg['position']
            cntr += 1

            if last_ball_x_real<250 and last_ball_x_real>0:
                ball_time_on_left+=1
            elif last_ball_x_real>=250:
                ball_time_on_right+=1

            if frame is not None:
                frame = cv2.resize(frame, (640+340, 384), interpolation=cv2.INTER_AREA)
                try:
                    frame = overlay_transparent(frame, overlay, 10, 354)
                except Exception as e:
                    print(f"Error overlaying score: {e}")
                    
                combined_frame = np.zeros((output_height, 1100+340, 3), dtype=np.uint8)
                
                # Place the pitch image below the original frame
                combined_frame[404:404+pitch_new_height, 281:281+pitch_width] = pitch
                
                if ball_time_on_right+ball_time_on_left>0:
                    left_perc = int(ball_time_on_left / (ball_time_on_right+ball_time_on_left) *100)
                    right_perc = 100-left_perc
                else:
                    left_perc=0
                    right_perc=0
                
                cv2.putText(combined_frame,f'{left_perc}%', (280+100, 690),cv2.FONT_HERSHEY_SIMPLEX,2, (80,140,80),thickness=5) 
                cv2.putText(combined_frame,f'{right_perc}%', (280+120+200, 690),cv2.FONT_HERSHEY_SIMPLEX,2, (80,140,80),thickness=5) 

                printed=False
                max_ttl = 40
                ball_rad= 6
                step = (255-100)/max_ttl
                step_ball = ball_rad/ max_ttl
                pitch_shift_x = 281
                pitch_shift_y = 423
                
                cv2.line(combined_frame, (20+3 + pitch_shift_x,17 + pitch_shift_y+96),(20 + 3+pitch_shift_x,17+pitch_shift_y+149), (0,0,255),2)
                cv2.line(combined_frame,(520 - 4 + pitch_shift_x,17 + pitch_shift_y+96),(520 - 4 +pitch_shift_x,17+pitch_shift_y+149), (0,0,255),2)
                distance_to_L_goal_line = None
                distance_to_R_goal_line = None

                current_distance_to_l= -1
                current_distance_to_r= -1
                total_nbr_of_displayed_players = 0
                player_box_size=20
                # Draw players and ball on pitch.jpg
                if players_collection is not None:
                    for player in players_collection:
                        if player.x_2d>0 and player.y_2d>0:
                            xx = int(20 + player.x_2d+pitch_shift_x - player_box_size / 2)
                            yy = int(pitch_shift_y+17+player.y_2d - player_box_size / 2 )
                            total_nbr_of_displayed_players+=1
                            ccol=(250,160,0)
                            if player.closest:
                                ccol =  (0,255,0)

                            cv2.rectangle(combined_frame,(xx,yy),(xx+player_box_size,yy+player_box_size),ccol, -1)
                            cv2.putText(combined_frame,  f"{player.player_id:02}", (xx+2, yy+12), cv2.FONT_HERSHEY_PLAIN, 0.9, [0,0,255], thickness=1) 

                
                for ball in ball_coords.data:
                    try:
                        ball.x=int(ball.x)
                        ball.y=int(ball.y)
                    except:
                        ball.x=-1
                        ball.y=-1

                    ball_processed=False
                    if (ball.x!=-1):
                        if position==ball.position:
                            ball_processed=True
                            ccolor=(100,100,255)
                            xx = int(20 + ball.x)+pitch_shift_x
                            yy = int(pitch_shift_y+17+ball.y)
                            last_ball_x = xx
                            last_ball_y = yy
                            last_ball_ttl = max_ttl
                            
                            last_ball_x_real = ball.x
                            last_ball_y_real = ball.y
                            
                            cv2.circle(combined_frame, (xx, yy), ball_rad, ccolor, -1)
                            printed=True
                            if ball.kicked==True:
                                 ballkicked_cntr=3
                                 #play_sound()

                            print_txt2(combined_frame, 320,770,f'Kick: {ball.kick_text} - players: {total_nbr_of_displayed_players}' )

                            
                            #previous_ball_speed_kmh = moving_average_speed
                            print_txt2(combined_frame, 20,570,f'SPEED: {ball.ball_speed_kmh} km/h')
                            print_txt2(combined_frame, 20,630,f'Ball x,y: {int(ball.x)}, {int(ball.y)}')

                            dir_change_mark=""
                            if ball.ball_direction_change:
                                    dir_change_mark="*"

                            print_txt2(combined_frame, 20,750,f'Ball Direction: {ball.ball_direction} {dir_change_mark}' )
                            print_txt2(combined_frame, 20,770,f'Acceleration: {ball.acceleration} m/s2' )

                            if ballkicked_cntr>0:
                                ballkicked_cntr-=1
                                cv2.rectangle(combined_frame,(210,0),(890,400), (0,255,0), -1)

                            # Check if goal checkers are initialized
                            if goalcheck_l is not None and goalcheck_r is not None:
                                if goal_hold_back_l<goal_holdback_frames-3 and ball.ball_speed_kmh>=goalcheck_l.settings.max_ball_bouncback_speed:
                                    goal_hold_back_l=-1
                                    print("Left speed reset", ball.ball_speed_kmh, goalcheck_l.settings.max_ball_bouncback_speed)
                                if goal_hold_back_r<goal_holdback_frames-3 and ball.ball_speed_kmh>=goalcheck_r.settings.max_ball_bouncback_speed:
                                    goal_hold_back_r=-1
                                    print("Right speed reset", ball.ball_speed_kmh, goalcheck_r.settings.max_ball_bouncback_speed)
                                    
                                if ball.acceleration<goalcheck_l.settings.max_ball_negative_acceleration:
                                    goal_hold_back_l=-1
                                    print("Left acceleration reset", ball.acceleration,"m/s2")                                
                                if ball.acceleration<goalcheck_r.settings.max_ball_negative_acceleration:
                                    goal_hold_back_r=-1
                                    print("Right acceleration reset", ball.acceleration,"m/s2")

                                distance_to_L_goal_line = round(distance_point_to_line(3,96,3,149,ball.x, ball.y)/500*25,1)
                                if (ball.x <3 and ball.x>-12 and ball.y<142 and ball.y>98):
                                    distance_to_L_goal_line = 0
                                distance_to_R_goal_line = round(distance_point_to_line(496,96,496,149,ball.x, ball.y)/500*25,1)
                                if (ball.x >496 and ball.x<514 and ball.y<142 and ball.y>98):
                                    distance_to_R_goal_line = 0

                                if goal_hold_back_l<goal_holdback_frames-5 and distance_to_L_goal_line>2.2:
                                    goal_hold_back_l=-1
                                    #print("Left distance reset", distance_to_L_goal_line)
                                if goal_hold_back_r<goal_holdback_frames-5 and distance_to_R_goal_line>2.2:
                                    goal_hold_back_r=-1
                                    #print("Right distance reset", distance_to_R_goal_line)

                                current_distance_to_l = distance_to_L_goal_line
                                current_distance_to_r = distance_to_R_goal_line
                                
                                print_txt2(combined_frame, 20,650,f'L Goal Dist: {distance_to_L_goal_line} m')
                                print_txt2(combined_frame, 20,670,f'R Goal Dist: {distance_to_R_goal_line} m')
                    if (ball.x==-1 or ball.ball_speed_kmh==-1):
                        print_txt2(combined_frame, 20,570,f'SPEED:')
                
                if multiball_alert:
                    print_txt2(combined_frame, 20,720,f'MULTIBALL ALERT!!!!')
                print_txt2(combined_frame, 20,700,f'MULTIBALL ALERT CNTR: {multiball_alert_cntr}' )
                
                ball_pos_history.append((last_ball_x_real, last_ball_y_real, last_ball_ttl, distance_to_L_goal_line, distance_to_R_goal_line))
                
                if printed==False and last_ball_ttl>0:
                        ttl_left = (max_ttl-last_ball_ttl)
                        ballshrink = ttl_left*step_ball
                        shift = ttl_left*step
                        ccolor_lost=(100+shift,100+shift,255)
                        last_ball_ttl-=1
                        if (ballshrink<ball_rad):
                            cv2.circle(combined_frame, (last_ball_x, last_ball_y), int(ball_rad - ballshrink), ccolor_lost, -1)

                goal_check_frame_l = None
                goal_check_frame_r = None
                
                # Handle goal checking if goal checkers are initialized
                if goalcheck_l is not None and goalcheck_r is not None and goal_l is not None and goal_r is not None:
                    if take_ref_img_l==True:
                        take_ref_img_l=False
                        cv2.imwrite(f'{goalcheck_l.settings.ref_frame_folder}ref_{total_frames_captures}_goal.png', goal_l)
                        goalcheck_l.reference_frame = goal_l

                    #start =time.perf_counter()*1000
                    goal_scored, goal_l_frame, goal_check_l_h, goal_check_l_w, goal_check_frame_l = goalcheck_l.check(goal_l, stat.detection_cycle, 0)
                    if goal_scored: 
                        print("GOAL scored[L] -> checking distance", current_distance_to_l)
                        if (current_distance_to_l<0.2 and current_distance_to_l>=0):
                        #if (current_distance_to_l==-1 or current_distance_to_l<0.5):
                            print("GOOOOL - B1", goal_holdback_frames)
                            #score_cntr_l+=1
                            if goal_hold_back_l==-1:
                                goal_hold_back_l = goal_holdback_frames   
                    now =time.perf_counter()*1000
                    #time_diff_ms = now - start
                    #print("Process time:", time_diff_ms)                    
                    goal_l_res = cv2.resize(goal_l_frame, (200, 200), interpolation=cv2.INTER_LINEAR)
                    combined_frame[40:240, 10:210] = goal_l_res
                    
                    if take_ref_img_r==True:
                        take_ref_img_r=False
                        cv2.imwrite(f'{goalcheck_r.settings.ref_frame_folder}ref_{total_frames_captures}_goal.png', goal_r)
                        goalcheck_r.reference_frame = goal_r

                    goal_scored, goal_r_frame, goal_check_r_h, goal_check_r_w, goal_check_frame_r = goalcheck_r.check(goal_r, stat.detection_cycle, 0)
                    if goal_scored:
                        print("GOAL scored[R] -> checking distance", current_distance_to_r)
                        if (current_distance_to_r<0.2 and current_distance_to_r>=0):
                        #if (current_distance_to_r==-1 or current_distance_to_r<0.5):
                            print("GOOOOL -A1", goal_holdback_frames)
                            #score_cntr_r+=1
                            if goal_hold_back_r==-1:
                                goal_hold_back_r = goal_holdback_frames
                    goal_r_res = cv2.resize(goal_r_frame, (200, 200), interpolation=cv2.INTER_LINEAR)
                    combined_frame[40:240, 890+340:1090+340] = goal_r_res

                    if goal_check_frame_l is not None:
                        combined_frame[240:240+goal_check_l_h, 10:10+200] = goal_check_frame_l[0:goal_check_l_h, 0:200]

                    if goal_check_frame_r is not None:
                        combined_frame[240:240+goal_check_r_h, 890+340:890+200+340] = goal_check_frame_r[0:goal_check_r_h,goal_check_r_w-200:300 ]


                    if current_distance_to_l>3:
                        goal_hold_back_l=-1
                    if current_distance_to_r>3:
                        goal_hold_back_r=-1

                    if goal_hold_back_l ==0:
                        print("Is goal confirmed?")
                        is_confirmed = goalcheck_l.do_last_look(ball_pos_history, GOAL_HOLDBACK_FRAME_CNTR)
                        if (is_confirmed):
                            score_cntr_l+=1
                            goal_hold_back_l=-1
                            print("GOOOOL confirmed")
                            
                            
                            goalcheck_l.init_goal_stop()
                            with open(ai_settings.video_out_folder+f'/highlights_{ai_settings.recording_id}.txt', 'a') as file_highlights:
                                rtime = stat.detection_cycle-VIDEO_SHIFT_SEC
                                file_highlights.write(f'GOAL,{rtime}, {score_cntr_l}:{score_cntr_r}\n')
                        
                            # Skip sportunity update for video processing
                            # update_sportunity_scores(ai_settings.sportunity_id, score_cntr_l, score_cntr_r)

                    if goal_hold_back_r ==0:
                        print("Is goal confirmed?")
                        is_confirmed = goalcheck_r.do_last_look(ball_pos_history, GOAL_HOLDBACK_FRAME_CNTR)
                        if (is_confirmed):
                            score_cntr_r+=1
                            goal_hold_back_r=-1
                            print("GOOOOL confirmed")
                            
                            goalcheck_r.init_goal_stop()
                            with open(ai_settings.video_out_folder+f'/highlights_{ai_settings.recording_id}.txt', 'a') as file_highlights:
                                rtime = stat.detection_cycle-VIDEO_SHIFT_SEC
                                file_highlights.write(f'GOAL,{rtime}, {score_cntr_l}:{score_cntr_r}\n')
                            # Skip sportunity update for video processing
                            # update_sportunity_scores(ai_settings.sportunity_id, score_cntr_l, score_cntr_r)

                minutes = stat.detection_cycle // 60
                seconds = stat.detection_cycle % 60

                # Formatting the time string as mm:ss
                time_string = f"{minutes:02}:{seconds:02}"


                print_txt_dyn(combined_frame,100,30,str(score_cntr_l),1,2)
                print_txt_dyn(combined_frame,970+340,30,str(score_cntr_r),1,2)

                match_time = time_string
                cv2.putText(frame, time_string, (24, 373), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], thickness=1)      
                cv2.putText(frame, str(score_cntr_l)+" : "+str(score_cntr_r), (200, 373), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,255], thickness=2)

                # Place the original frame at the top
                combined_frame[0:384, 230:870+340] = frame

                cv2.rectangle(combined_frame,(830,400),(1215,800), (80,40,40), -1)
                
                ic=0
                idx =stat.detection_cycle*25+position
                img_displayed=False
                extra_sh=0
                if currentAction is not None:
                    for i in range(1,12):
                        last_action = currentAction.get_action(i, idx)
                        if last_action is not None:
                            txt=last_action.get_short_string()
                            if img_displayed==False:

                                last_xx=0
                                last_yy=0
                                for (bx, by) in last_action.ball_data:
                                    xx = int(20 + bx)+pitch_shift_x
                                    yy = int(pitch_shift_y+17+by)
                                    if last_xx>0:
                                        cv2.line(combined_frame, (xx,yy), (last_xx, last_yy),(0,0,255),1,cv2.LINE_AA)
                                    last_xx = xx
                                    last_yy = yy
                            
                            thi=1
                            ccc=[0,255,255]
                            if "LONG" in txt:
                                ccc=[255,220,220]
                            if "MEDIUM" in txt:
                                ccc=[180,255,180]
                            if "GOAL" in txt:
                                ccc=[180,180,255]
                            cv2.putText(combined_frame, f'{txt}', (830,420+ic*20+extra_sh), cv2.FONT_HERSHEY_PLAIN, 0.8, 
                                        ccc, thickness=thi)
                            ic+=1
                            extra_sh=15

                print_txt2(combined_frame, 20,530,f'DET. RATE: {stat.overall_detection_rate}%')
                print_txt2(combined_frame, 20,550,f'CYCLE: {stat.detection_cycle}')

                print_txt2(combined_frame, 20,590, f'Best Cam.: CAM{stat.selected_cam}')
                print_txt2(combined_frame, 20,610, f'FPS: {stat.total_frames}/sec')

                start_pos = 440
                print_txt(combined_frame, 890+340, start_pos, f'Camera feeds: {len(stat.cam_info)}')
                print_txt(combined_frame, 890+340, start_pos+15, f'Total Process Time: {stat.process_time} ms')
                print_txt(combined_frame, 890+340, start_pos+30, f'AI processed frames: {stat.total_detections}/sec')
                print_txt(combined_frame, 890+340, start_pos+45, f'Ball detected frames: {stat.ball_detected_frames}',[255,180,180])
                row=0
                for info in stat.cam_info:
                    row+=1
                    color = [255,255,0]
                    if info.id == stat.selected_cam:
                        color = [0,255,255]
                    print_txt(combined_frame, 890+340, start_pos+20+row*50, f'[{info.id}] {info.name}', color)
                    print_txt(combined_frame, 910+340, start_pos+20+row*50+15, f'FPS: [{info.total_frames}] Det: {info.total_detections} Ball: {info.ball_detected_frames}')
                    print_txt(combined_frame, 910+340, start_pos+20+row*50+30, f'Players detected: {info.players}')



                feedbackQueue.put({'score_cntr_l':score_cntr_l,'score_cntr_r':score_cntr_r, 'match_time':match_time})
                cv2.imshow('Frame', combined_frame)
                if (ai_settings.video_disabled==False):
                    out.write(combined_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    thread_running.value = 0
                    break
                #if key == ord('k'):
                #    play_sound()
                elif key == ord('s'):
                    score_cntr_l=0
                    score_cntr_r=0
                elif key == ord('1'):
                    score_cntr_l+=1
                elif key == ord('2'):
                    score_cntr_r+=1
                elif key == ord('l'):
                    take_ref_img_l=True
                elif key == ord('r'):
                    take_ref_img_r=True

        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    thread_running.value = 0
                    break
            time.sleep(0.005)
    
    # Clean up resources
    if (ai_settings.video_disabled==False) and 'out' in locals():
        out.release()
    cv2.destroyAllWindows()
    print("Display thread stopped")
    
def print_txt(frame, x, y, txt, color = [255,255,255]):
    cv2.putText(frame,
                txt, (x, y),
                cv2.FONT_ITALIC,
                0.4, color,
                thickness=1)
    
def print_txt2(frame, x, y, txt, color = [255,255,255]):
    cv2.putText(frame,
                txt, (x, y),
                cv2.FONT_ITALIC,
                0.5, color,
                thickness=1)      

def print_txt_dyn(frame, x, y, txt, size, thickn, color = [255,255,255]):
    cv2.putText(frame,
                txt, (x, y),
                cv2.FONT_ITALIC,
                size, color,
                thickness=thickn)      

def distance_point_to_line(x1, y1, x2, y2, x, y):
    # If the line's points coincide, it's just a distance to a point
    if (x2 - x1 == 0) and (y2 - y1 == 0):
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    # Line direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Project the point (x, y) onto the line, finding the projection point
    t = ((x - x1) * dx + (y - y1) * dy) / (dx ** 2 + dy ** 2)
    
    # Find the nearest point on the line segment to (x, y)
    t = max(0, min(1, t))
    
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    # Return the Euclidean distance from the point to the nearest point on the line
    return math.sqrt((x - nearest_x) ** 2 + (y - nearest_y) ** 2)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    Overlay a transparent image `img_to_overlay_t` onto another image `background_img` at position `x`, `y`.
    """
    if img_to_overlay_t is None or background_img is None:
        return background_img
        
    try:
        bg_img = background_img.copy()
        
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        # Extract the alpha mask of the RGBA image, convert to RGB 
        b, g, r, a = cv2.split(img_to_overlay_t)
        overlay_color = cv2.merge((b, g, r))
        
        # Apply some simple filtering to remove edge noise
        mask = cv2.medianBlur(a, 15)

        h, w, _ = overlay_color.shape
        roi = bg_img[y:y+h, x:x+w]

        # Black-out the area behind the logo in our original ROI
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        
        # Mask out the logo from the logo image.
        img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

        # Update the original image with our new ROI
        bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

        return bg_img
    except Exception as e:
        print(f"Error in overlay_transparent: {e}")
        return background_img

def distance_between_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# For moving average calculation of speed
MAX_SPEED_MVG_AVRG = 6
speed_queue = deque(maxlen=MAX_SPEED_MVG_AVRG)
acceleration_queue = deque(maxlen=MAX_SPEED_MVG_AVRG)

def calculate_moving_average(queue, new_speed, queue_size=MAX_SPEED_MVG_AVRG):
    def round_speed(speed):
        return round(speed, 2)
    
    if new_speed != -1:
        queue.append(new_speed)
    
    # Ensure the queue size does not exceed the limit
    if len(queue) > queue_size:
        queue.popleft()
    
    if len(queue) < queue_size and new_speed != -1:
        return round_speed(sum(queue) / len(queue))
    
    if queue.count(-1) == len(queue):
        return -1
    
    # Assign weights to the elements in the queue, giving more weight to recent values
    weights = range(1, len(queue) + 1)
    weighted_sum = sum(weight * speed for weight, speed in zip(weights, queue) if speed != -1)
    total_weights = sum(weights)
    
    moving_average = weighted_sum / total_weights
    return round_speed(moving_average)

# def extract_frames_from_video(video_path, max_frames, output_queue, thread_running):
#     """
#     Extract frames from video file and feed them to the system
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file: {video_path}")
#         return False
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"Video loaded: {video_path}")
#     print(f"FPS: {fps}, Total frames: {frame_count}")
    
#     # Adjust frame extraction rate based on video fps vs desired fps
#     skip_frames = max(1, round(fps / FEED_FPS))
    
#     frame_idx = 0
#     frames_read = 0
    
#     while cap.isOpened() and frames_read < max_frames and thread_running.value == 1:
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video reached")
#             break
            
#         # Skip frames to match desired FPS if needed
#         if frame_idx % skip_frames == 0:
#             output_queue.put((frame, frames_read))
#             frames_read += 1
            
#         frame_idx += 1
        
#         # Sleep to prevent queue overflow
#         if output_queue.qsize() > 100:
#             time.sleep(0.1)
    
#     cap.release()
#     return True

def process_single_frame(detector, frame, cycle, position, csv_writer=None, team_identifier=None):
    """
    Process a single video frame directly with the detector and save player positions to CSV
    
    Args:
        detector: AI detector object
        frame: Video frame to process
        cycle: Current detection cycle
        position: Position within cycle
        csv_writer: CSV writer object for logging player positions
        team_identifier: TeamIdentifier object for team classification
    """
    # Create a copy of the frame for processing
    w, h = 2300, 896
    if frame.shape[1] > 1000:
        w, h = 2300, 896
    else:
        w, h = 960, 576
    resized_frame = cv2.resize(frame, (w, h))
    frame_copy = resized_frame.copy()
    
    # Create a detection object
    det_obj = Detector(0, 0, position, cycle, DetectionType.NORMAL)
    det_obj.frame = resized_frame
    
    # Generate timestamp for logging
    timestamp = f"{cycle//60:02d}:{cycle%60:02d}.{position:02d}"
    
    # Track ball position
    ball_x = -1
    ball_y = -1
    ball_x_2d = -1
    ball_y_2d = -1
    
    # Run detection on this frame
    try:
        # Get tensor representation
        tensor = create_tensor(resized_frame, detector.device, w, h)
        if tensor is None:
            print("Failed to create tensor from frame")
            return None
            
        # Select engine based on frame size
        if frame.shape[1] > 1000:
            engine = detector.engines_1280[0]
        else:
            engine = detector.engines_960[0]
            
        # Run detection
        data = engine(tensor)
        
        # Post-process results
        bboxes, scores, labels = det_postprocess(data)
        
        # Process detection results
        results = {
            'frame': frame_copy,
            'stats': DetectorStats(),
            'ball': BallCoords(),
            'players': []
        }
        
        # Check if any objects were detected
        if bboxes.numel() > 0:
            # First, find the ball (needed to calculate player-ball distance)
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                if cls_id == 0 and score >= 0.4:  # Ball detection
                    x1, y1 = bbox[:2]
                    x2, y2 = bbox[2:]
                    ball_x = (x1 + x2) / 2
                    ball_y = (y1 + y2) / 2
                    # Convert to 2D field coordinates (simplified mapping)
                    ball_x_2d = int(ball_x / w * 500)  # Map to 500-unit field width
                    ball_y_2d = int(ball_y / h * 250)  # Map to 250-unit field height
                    break
            
            # Now process all detections (balls, players)
            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                x1, y1 = bbox[:2]
                x2, y2 = bbox[2:]
                
                # Handle player detection
                if cls_id == 1 and score >= 0.3:
                    player = Player()
                    player.x1 = int(x1)
                    player.x2 = int(x2)
                    player.y1 = int(y1)
                    player.y2 = int(y2)
                    player.confidence = float(score)
                    
                    # Calculate player center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Convert to 2D field coordinates (simplified mapping)
                    player_x_2d = int(center_x / w * 500)  # Map to 500-unit field width
                    player_y_2d = int(center_y / h * 250)  # Map to 250-unit field height
                    
                    # Set 2D coordinates
                    player.x_2d = player_x_2d
                    player.y_2d = player_y_2d
                    
                    # Calculate distance to ball if ball is detected
                    distance_to_ball = -1
                    if ball_x != -1 and ball_y != -1:
                        distance_to_ball = distance_between_points(center_x, center_y, ball_x, ball_y)
                        player.ball_distance = distance_to_ball
                        player.ball_x = ball_x
                        player.ball_y = ball_y
                    
                    # Assign a unique ID (simplified - in a real system would use tracking)
                    player.player_id = hash((x1, y1, x2, y2)) % 1000  # Simple hash-based ID
                    
                    # Extract player image for team identification
                    player_img = frame_copy[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Determine team if team_identifier is provided
                    team_id = -1
                    display_color = (255, 255, 255)  # Default: white
                    
                    if team_identifier is not None and player_img.size > 0:
                        team_id = team_identifier.get_team_for_player(player.player_id, player_img)
                        display_color = team_identifier.get_display_color_for_player(player.player_id)
                    
                    # Store team info
                    player.team_id = team_id
                    
                    # Add player to results
                    results['players'].append(player)
                    
                    # Draw player box with team color
                    cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), display_color, 2)
                    
                    # Draw player ID
                    cv2.putText(frame_copy, f"ID:{player.player_id}", (int(x1), int(y1)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 1)
                    
                    # Save player data to CSV if writer is provided
                    if csv_writer:
                        csv_writer.writerow([
                            cycle,                    # detection cycle
                            position,                 # position within cycle
                            player.player_id,         # player ID
                            player.x1,                # bounding box x1
                            player.y1,                # bounding box y1
                            player.x2,                # bounding box x2
                            player.y2,                # bounding box y2
                            player.x_2d,              # field x coordinate
                            player.y_2d,              # field y coordinate
                            player.confidence,        # detection confidence
                            ball_x_2d,                # ball x coordinate
                            ball_y_2d,                # ball y coordinate
                            distance_to_ball,         # distance to ball
                            team_id,                  # team ID
                            timestamp                 # timestamp
                        ])
                
                # Handle ball detection
                elif cls_id == 0 and score >= 0.4:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = min(x2 - x1, y2 - y1) // 2
                    
                    # Draw ball
                    cv2.circle(frame_copy, (center_x, center_y), radius, (0, 0, 255), -1)
                    
                    # Create ball data
                    ball_data = BallCoordsObj(
                        0, ball_x_2d, ball_y_2d, center_x, center_y, 
                        radius, 0, position, 0, 0, False, False, "", False
                    )
                    results['ball'].data.append(ball_data)
        
        # Update statistics
        results['stats'].detection_cycle = cycle
        results['stats'].selected_cam = 0
        results['stats'].total_frames = 1
        results['stats'].total_detections = 1
        results['stats'].ball_detected_frames = 1 if len(results['ball'].data) > 0 else 0
        
        return results
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def main() -> None:
    global thread_running, score_cntr_l, score_cntr_r, match_time, ai_settings

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Football Video Analysis with Team Identification')
    parser.add_argument('--video', type=str, default='input1.mp4', help='Path to the input video file')
    parser.add_argument('--config_path', type=str, default='c:/Develop/Configuration', help='Path to configuration directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--duration', type=int, default=90, help='Duration of analysis in minutes')
    parser.add_argument('--csv', type=str, help='Path to output CSV file for player positions (defaults to output_dir/player_positions_timestamp.csv)')
    parser.add_argument('--yolo_seg', type=str, default='yolov8m-seg.pt', help='Path to YOLOv8 segmentation model')
    parser.add_argument('--yolo_pose', type=str, default='yolov8m-pose.pt', help='Path to YOLOv8 pose model')
    parser.add_argument('--num_teams', type=int, default=2, help='Number of teams to identify (default: 2)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup CSV for player positions
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(args.output_dir, f'player_positions_{timestamp_str}.csv')
    
    # Open CSV file for writing player positions
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write CSV header - now includes team_id
    csv_writer.writerow([
        'cycle', 'position', 'player_id', 'x1', 'y1', 'x2', 'y2',
        'field_x', 'field_y', 'confidence', 'ball_x', 'ball_y',
        'distance_to_ball', 'team_id', 'timestamp'
    ])
    
    print(f"Player positions will be saved to: {csv_path}")
    
    # Load AI settings
    config_path = args.config_path
    feed_config = os.path.join(config_path, "feed_config.json")
    ai_config = os.path.join(config_path, "ai_config.json")
    
    # Check if configuration files exist and create if needed
    if not os.path.exists(feed_config) or not os.path.exists(ai_config):
        print(f"Error: Configuration files not found at {config_path}")
        print("Creating example configuration files...")
        
        # Create example configuration directory
        os.makedirs(config_path, exist_ok=True)
        
        # Create example feed_config.json
        feed_config_example = {
            "field_id": 1,
            "detection_first_stage_frames": 6,
            "detection_last_stage_on_best_frames": 24,
            "detection_last_stage_on_both_centers_frames": 12,
            "video_disabled": False,
            "video_out_folder": args.output_dir,
            "recording_id": int(time.time()),
            "duration_min": args.duration,
            "sportunity_id": "example_id"
        }
        
        with open(feed_config, 'w') as f:
            json.dump(feed_config_example, f, indent=4)
        
        # Create example ai_config.json
        ai_config_example = {
            "ball_confidence": 0.4,
            "people_confidence": 0.3,
            "min_ball_size": 8,
            "ball_do_deep_check": True,
            "ball_mean_saturation": 50,
            "ball_mean_value": 150
        }
        
        with open(ai_config, 'w') as f:
            json.dump(ai_config_example, f, indent=4)
        
        print(f"Example configuration files created in {config_path}")
    
    # Initialize AI settings
    ai_settings = AISettings(feed_config, ai_config)
    
    # Override some settings
    ai_settings.video_out_folder = args.output_dir
    ai_settings.recording_id = int(time.time())
    ai_settings.duration_min = args.duration
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        csv_file.close()
        return
    
    print(f"Processing video: {args.video}")
    
    # Check if YOLO models exist (for team identification)
    seg_model_path = args.yolo_seg
    pose_model_path = args.yolo_pose
    
    # Import team identification module
    try:
        from team_identification import TeamIdentifier
        team_identifier = TeamIdentifier(num_teams=args.num_teams)
        print(f"Team identification initialized for {args.num_teams} teams")
    except ImportError as e:
        print(f"Warning: Team identification module could not be imported: {e}")
        print("Players will not be identified by team")
        team_identifier = None
    
    # Open the video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {args.video}")
        csv_file.close()
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video loaded: {args.video}")
    print(f"FPS: {fps}, Resolution: {frame_width}x{frame_height}, Total Frames: {total_frames}")
    
    # Initialize detector directly (no cameras needed with this approach)
    detector = AIDetector(ai_settings, [], PARALEL_MODELS, MAX_FPS)

    # Start the display thread
    display_thread = mp.Process(target=display_frames, args=(thread_running, displayQueue, feedbackQueue, ai_settings))
    display_thread.start()

    # Create a log file for highlights
    with open(os.path.join(ai_settings.video_out_folder, f'highlights_{ai_settings.recording_id}.txt'), 'w') as file_highlights:
        file_highlights.write(f'Event,Timestamp,Score\n')

    # Process video frames
    print("Starting video analysis...")
    start_time = time.time()
    frame_count = 0
    
    # Processing rate control
    target_fps = min(fps, MAX_FPS)
    skip_frames = max(1, int(fps / target_fps))
    print(f"Processing at {target_fps} FPS (skipping every {skip_frames-1} frames)")
    
    # Initialize stats
    current_stats = DetectorStats()
    current_stats.detection_cycle = 0
    current_stats.selected_cam = 0
    
    # For team identification initialization
    if team_identifier is not None:
        print("Collecting initial player samples for team identification...")
        player_samples = {}
        initial_frames_needed = min(100, total_frames // 2)  # Use up to 100 frames or half the video
        
        # Collect player images from a subset of frames for team identification
        temp_frame_count = 0
        while temp_frame_count < initial_frames_needed and len(player_samples) < 20:  # Limit to 20 player samples
            ret, frame = cap.read()
            if not ret:
                break
                
            temp_frame_count += 1
            if temp_frame_count % 10 != 0:  # Only process every 10th frame for efficiency
                continue
                
            # Process frame to detect players
            results = process_single_frame(detector, frame, 0, 0)
            if results and 'players' in results and len(results['players']) > 0:
                for player in results['players']:
                    if player.player_id not in player_samples and player.x2 > player.x1 and player.y2 > player.y1:
                        # Extract player image
                        player_img = frame[max(0, int(player.y1)):min(frame.shape[0], int(player.y2)), 
                                           max(0, int(player.x1)):min(frame.shape[1], int(player.x2))]
                        if player_img.size > 0:
                            player_samples[player.player_id] = (player_img, None)  # No keypoints yet
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Initialize team identification if we have enough samples
        if len(player_samples) >= args.num_teams * 2:  # Need at least 2 players per team
            print(f"Initializing team identification with {len(player_samples)} player samples")
            team_identifier.initialize_teams(player_samples)
        else:
            print(f"Warning: Not enough player samples ({len(player_samples)}) for reliable team identification")
    
    # Process frames until thread_running is set to 0 or duration is exceeded
    try:
        while cap.isOpened() and thread_running.value == 1:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Check if we've reached the specified duration
            if elapsed_time >= ai_settings.duration_min * 60:
                print(f"Reached specified duration of {ai_settings.duration_min} minutes. Exiting.")
                break

            # Process feedback from display thread (like score updates)
            if not feedbackQueue.empty():
                feedback_msg = feedbackQueue.get()
                score_cntr_l = feedback_msg['score_cntr_l']
                score_cntr_r = feedback_msg['score_cntr_r']
                match_time = feedback_msg['match_time']

            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            # Process at target FPS by skipping frames if needed
            frame_count += 1
            if (frame_count - 1) % skip_frames != 0:
                continue
            
            # Calculate current cycle and position
            position = (frame_count // skip_frames) % MAX_FPS
            current_cycle = (frame_count // skip_frames) // MAX_FPS + 1
            
            # Update match time
            minutes = current_cycle // 60
            seconds = current_cycle % 60
            match_time = f"{minutes:02}:{seconds:02}"
            
            # Process frame directly using our helper function with the CSV writer and team identifier
            results = process_single_frame(detector, frame, current_cycle, position, csv_writer, team_identifier)
            
            if results:
                # Create goal camera views from the frame
                h, w = frame.shape[:2]
                goal_l = frame[:, :w//3].copy()  # Left third of the frame
                goal_r = frame[:, 2*w//3:].copy()  # Right third of the frame
                
                goal_l = cv2.resize(goal_l, (200, 200))
                goal_r = cv2.resize(goal_r, (200, 200))
                
                # Update current stats
                current_stats.detection_cycle = current_cycle
                current_stats.total_frames += 1
                current_stats.total_detections += 1
                current_stats.ball_detected_frames += 1 if len(results['ball'].data) > 0 else 0
                
                # Send results to display queue
                displayQueue.put({
                    'frame': results['frame'],
                    'goal_l': goal_l,
                    'goal_r': goal_r,
                    'stats': current_stats,
                    'ball': results['ball'],
                    'position': position,
                    'multiball': 0,
                    'players': results['players'],
                    'currentAction': currentAction  # Use global currentAction
                })
            
            # Control processing rate
            frame_time = time.time() - current_time
            target_time = 1.0 / target_fps
            if frame_time < target_time:
                time.sleep(target_time - frame_time)

            # Print progress every 100 frames
            if (frame_count // skip_frames) % 100 == 0:
                progress = frame_count / total_frames * 100
                elapsed = time.time() - start_time
                remaining = (total_frames - frame_count) / (frame_count / elapsed) if frame_count > 0 else 0
                print(f"Progress: {progress:.1f}% | ETA: {int(remaining/60):02d}:{int(remaining%60):02d} | Frame: {frame_count}/{total_frames}")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        print("Video processing complete.")
        
        # Close the CSV file
        csv_file.close()
        print(f"Player positions saved to: {csv_path}")
        
        # Allow time for threads to finish
        time.sleep(1)

        # Release video capture
        cap.release()
        
        # Terminate display thread
        try:
            display_thread.terminate()
        except:
            pass
        
        print("Analysis completed")
        print(f"Output saved to {ai_settings.video_out_folder}")
        print(f"Final score: {score_cntr_l} - {score_cntr_r}")

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    print("Stopping...")
    os._exit(0)