import os
import time
import cv2
import psutil
import requests
import threading
import numpy as np
import multiprocessing as mp
from datetime import datetime
from collections import deque

from src.utils import *
from src.ball3 import *
from src.camera import *
from src.aidetector import *
from src.aisettings import *
from src.playeraction import *
from src.player_tracker import *
import proto_messages.players_pb2 as players_pb2

def get_performance_cores():
    return range(8)

performance_cores = get_performance_cores()
p = psutil.Process(os.getpid())
p.cpu_affinity(performance_cores)
player_publisher = None
GOAL_HOLDBACK_FRAME_CNTR = 30
BALL_QUEUE_LEN = GOAL_HOLDBACK_FRAME_CNTR * 2
ball_pos_history = deque(maxlen=BALL_QUEUE_LEN)

FEED_FPS = 25
MAX_FPS = 25
PARALEL_MODELS = 3
VIDEO_SHIFT_SEC = 7

player_tracker = PlayerTracker()
currentAction = CurrentAction()
displayQueue = mp.Queue()
feedbackQueue = mp.Queue()
thread_running = mp.Value('i', 1)
frame_start = mp.Value('i', 0)
cameras =[]
score_cntr_l = 0
score_cntr_r = 0
match_time = "00:00"
previous_ball_speed_kmh = -1
kick_detected = False
kick_ttl = 0
last_ball_direction = 0
last_valid_ball_x = 0
last_valid_ball_y = 0
prev_acceleration = -1

detector_cycle_live = 0
detector_cycle_processing = 0
overall_stat = OverallStats()
rpc_publisher = None
last_audio_notification=time.time()-15

ai_settings = AISettings("c:/Develop/Configuration/feed_config.json", "c:/Develop/Configuration/ai_config.json")
CAMERA_ID = "accueil"

def send_camera_event(camera_id, event_type, duration):
    global last_audio_notification
    current_time = datetime.now().strftime("%H:%M:%S")
    print("["+ current_time + "] SENDING CAMERA EVENT", camera_id, event_type, duration)
    url = "https://app.backendsportunity2017.com/devices-hooks/camera-event"
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
    global score_cntr_l, score_cntr_r, match_time, previous_ball_speed_kmh, kick_detected, kick_ttl, last_ball_direction, currentAction, last_valid_ball_x, last_valid_ball_y, prev_acceleration, player_publisher
    detector_selected_cam = detector.best_camera
    detector_process_time = detector.last_proc_time
    detector_broadcast_cam = detector.broadcast_camera

    ball_coords = BallCoords()
    multiple_balls = 0
    for cam in cameras:
        if cam.camera_id == detector_selected_cam:
            detector_objs = cam.get_detector_collection(cycle)
            for det_obj in detector_objs:
                if det_obj is not None:
                    if det_obj.ball > 1:
                        multiple_balls += 1
                    if det_obj.x1==-1:
                        continue
                    cent_x, cent_y = (det_obj.x1 + det_obj.x2) // 2, (det_obj.y1 + det_obj.y2) // 2
                    radius = min(det_obj.x2 - det_obj.x1, det_obj.y2 - det_obj.y1) // 2
                    moving_average_speed = calculate_moving_average(speed_queue, det_obj.ball_speed_kmh)
                    v1_ms = previous_ball_speed_kmh * (1000 / 3600)
                    v2_ms = moving_average_speed * (1000 / 3600)
                    delta_v_ms = v2_ms - v1_ms
                    delta_t_s = 0.04  
                    acceleration = int(delta_v_ms / delta_t_s)
                    moving_average_acceleration = int(calculate_moving_average(acceleration_queue, acceleration))
                    if moving_average_acceleration>-1 and moving_average_acceleration<=1:
                        moving_average_acceleration=0
                    dy = det_obj.y1- last_valid_ball_y
                    dx = det_obj.x1 - last_valid_ball_x
                    angle_radians = math.atan2(dy, dx)
                    angle_degrees = math.degrees(angle_radians)
                    angle_degrees = (angle_degrees + 360) % 360

                    det_obj.ball_direction = int(angle_degrees)
                    last_valid_ball_y = det_obj.y1
                    last_valid_ball_x = det_obj.x1
                    kicked=False
                    
                    previous_ball_speed_kmh = moving_average_speed                    
                    det_obj.ball_speed_kmh = moving_average_speed
                    det_obj.ball_direction_change = False
                    if det_obj.ball_direction > 0:
                        direction_delta = abs(last_ball_direction - det_obj.ball_direction)
                        direction_delta = min(direction_delta, abs(360-direction_delta))
                        
                        if direction_delta>80 and moving_average_speed > 5:
                            det_obj.ball_direction_change = True
                            kick_detected = False #reset previous kick as direction chnaged
                        
                        last_ball_direction = det_obj.ball_direction
                    if moving_average_acceleration<2 and moving_average_speed<2:
                        kick_detected=False
                    
                    if (moving_average_acceleration>3 and kick_detected==False) or (moving_average_acceleration>6 and (moving_average_acceleration-prev_acceleration)>5 and moving_average_speed<26):
                            kick_detected=True
                            kicked = True
                            currentAction.start_action(det_obj.x_2d, det_obj.y_2d, match_time, cycle, det_obj.position)
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
    detector_selected_cam = 2
    width, height = 2300, 896
    for cam in cameras:
        if cam.camera_id == detector_selected_cam:
            detector_objs = cam.get_detector_collection(cycle)
            for det_obj in detector_objs:
                if det_obj is not None:
                    if det_obj.frame.shape[1] > 1000:
                        width, height = 2300, 896
                    else:
                        width, height = 960, 576
                    frame = cv2.resize(det_obj.frame, (width, height))
                    frame_goal_l = goal_l_cam.get_goal_frame(cycle, det_obj.position)
                    frame_goal_r = goal_r_cam.get_goal_frame(cycle, det_obj.position)
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
                    if (det_obj.x1 > 0):
                        center_x, center_y = (det_obj.x1 + det_obj.x2) // 2, (det_obj.y1 + det_obj.y2) // 2
                        radius = min(det_obj.x2 - det_obj.x1, det_obj.y2 - det_obj.y1) // 2
                        ball_radius = radius
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
                        players_collection_ttl -= 1
                    if players_collection_ttl>0:
                        closest_player=None
                        closest_distance=None
                        closet_player_width = None
                        distance=-1
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
                                cv2.rectangle(frame, (int(player.x1), int(player.y1)), (int(player.x2), int(player.y2)), (0,255,0), 2)
                                player.cam_id = cam.camera_id
                                player.ball_distance = distance
                                player.ball_x = ball_point_x
                                player.ball_y = ball_point_y

                                if (player.x_2d>0 and player.y_2d <0):
                                    player.y_2d=0
                                gameplayer_collection.add_to_collection(det_obj.position, player)

                        player_publisher.send(player_collection)
                        if closest_player is not None and closest_distance<(closest_player.x2-closest_player.x1)*1.5:
                            bounding_img = det_obj.frame[closest_player.y1:closest_player.y2,closest_player.x1:closest_player.x2]
                            currentAction.add_player(0,bounding_img, cycle, det_obj.position)
                            wwidth = 2
                            display_rect = True
                            ccolor = (0, 255, 0)
                            ball_obj = ball_coords.get_ball_data_by_position(det_obj.position)
                            if ball_obj is not None and ball_obj.kicked==True:
                                wwidth = 10
                                ccolor = (0, 0, 255)
                            elif ball_obj is not None and ball_obj.kick_detected==True:
                                display_rect = False
                            if display_rect == True:
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
                    
                    cj = 0
                    for pl in all_players:
                        cj+=1
                        print(f'[{cj}] 2dx:{pl.x_2d},2dy:{pl.y_2d}, cam:{pl.cam_id} = {pl.y_2d} -> {pl.x1}:{pl.y1} - {pl.x2}:{pl.y2} [w:{pl.x2-pl.x1},h:{pl.y2-pl.y1}] Ball Distance: {pl.ball_distance}, ball x,y: {pl.ball_x}, {pl.ball_y}')

def trigger_camera_event():
    even_type = f'AI_FIELD_{ai_settings.field_id}_ALARM1'
    send_camera_event(CAMERA_ID, even_type, 3)
    time.sleep(2)
    send_camera_event(CAMERA_ID, even_type, 0)

def main() -> None:
    global thread_running, detector_cycle_live, detector_cycle_processing, cameras, rpc_publisher, score_cntr_l, score_cntr_r, match_time, currentAction, player_publisher
    detector_stage=0
    ecal_core.initialize([], "BallTracker AI - PYTHON")
    rpc_publisher = ProtoPublisher("RPC2", rpc_pb2.RPCData)
    player_publisher = ProtoPublisher("Players", players_pb2.FootPlayer)

    cameras.append(Camera(ai_settings.field_id, 0, "c:/forex/yolo/cam3.mp4",thread_running, frame_start, MAX_FPS, FEED_FPS, "Side A"))
    cameras.append(Camera(ai_settings.field_id, 1, "c:/forex/yolo/cam4.mp4",thread_running, frame_start, MAX_FPS, FEED_FPS, "Side B"))
    cameras.append(Camera(ai_settings.field_id, 2, "",thread_running, frame_start, MAX_FPS, FEED_FPS, "Center LEFT"))
    cameras.append(Camera(ai_settings.field_id, 3, "",thread_running, frame_start, MAX_FPS, FEED_FPS, "Center RIGHT"))
    cameras.append(Camera(ai_settings.field_id, 10, "",thread_running, frame_start, MAX_FPS, FEED_FPS, "Goal R", detection_type = DetectionType.GOAL))
    cameras.append(Camera(ai_settings.field_id, 11, "",thread_running, frame_start, MAX_FPS, FEED_FPS, "Goal L", detection_type = DetectionType.GOAL))
    detector = AIDetector(ai_settings, cameras, PARALEL_MODELS, MAX_FPS)

    display_thread = mp.Process(target=display_frames, args=(thread_running, displayQueue, feedbackQueue))
    display_thread.start()

    overall_cntr=0
    overall_start_time = None
    time_segment = 1
    total_detect_frames = 0
    sub_ball = 0
    sub_ball_sum = 0
    sub_sum = 0
    fps_cntr =0
    fps_start = None
    detector_selected_cam = -1
    detector_cycle_live = 1
    detector_cycle_processing = 1
    detector_process_time = 0
    detector.start_cycle(5)
    idx=0
    frame_start.value = round(time.perf_counter()*1000)
    last_processed_detector_cycle =0

    if ecal_core.ok():
        print("Waiting for topic registration")
        time.sleep(1.1) 
        rpc_publisher.send(restart_session_rpc())
        
    req_snt = False
    detector_cycle_live=-1
    passed_cycles = 0
    detector_started = False
    start_time = time.time()
    while (thread_running.value==1 and ecal_core.ok()):
        time.sleep(0.005)
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= ai_settings.duration_min*60:
            break  # Exit the loop after the duration is met
        if not feedbackQueue.empty():
            feedback_msg = feedbackQueue.get()
            score_cntr_l = feedback_msg['score_cntr_l']
            score_cntr_r = feedback_msg['score_cntr_r']
            match_time = feedback_msg['match_time']
        for cam in cameras:
            if cam.current_live_cycle  >detector_cycle_live:
                last_cycle = detector_cycle_live
                post_proc_start = time.time()
                detector_cycle_live = cam.current_live_cycle
                for cam in cameras:
                    cam.finalize_detector_array(last_cycle)        
                if detector_started: 
                    process_detection_results(rpc_publisher,detector, last_cycle-1)
                for cam in cameras:
                    cam.reset_segment(detector_cycle_live)
                current_time = time.time()
                passed_cycles += 1
                break

        if passed_cycles>2 and detector_started==False:
            detector_started = True
            detector.start_cycle(detector_cycle_live-1)
        
        if detector_started:
            detector.process(detector_cycle_live)

        continue
    print("Exiting....")
    time.sleep(1)
    for cam in cameras:
        try:
            cam.thread.terminate()
        except:
            pass
    try:
        display_thread.terminate()
    except:
        pass

def distance_between_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.medianBlur(a, 15)
    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    return bg_img

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
    if (x2 - x1 == 0) and (y2 - y1 == 0):
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    dx = x2 - x1
    dy = y2 - y1
    t = ((x - x1) * dx + (y - y1) * dy) / (dx ** 2 + dy ** 2)
    t = max(0, min(1, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.sqrt((x - nearest_x) ** 2 + (y - nearest_y) ** 2)

def update_sportunity_scores(sportunity_id, score1, score2):
    thread = threading.Thread(target=publish_sportunity_scores, args=(sportunity_id, score1, score2))
    thread.start()

def publish_sportunity_scores(sportunity_id, score1, score2):
    try:
        url = 'https://backendsportunity2017.com/graphql'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # Revised mutation to match the new requirements
        mutation = """
        mutation UpdateSportunityScores($id: String!, $score1: Int!, $score2: Int!) {
            updateSportunityScores(input: {
                sportunityId: $id,
                score1: $score1,
                score2: $score2
            }) {
                clientMutationId
            }
        }
        """
        
        # Variables to be sent with the mutation
        variables = {
            'id': sportunity_id,
            'score1': score1,
            'score2': score2
        }
        
        # Send the mutation request
        response = requests.post(url, json={'query': mutation, 'variables': variables}, headers=headers)
        
        # Check the response status
        if response.status_code == 200:
            print("Mutation successful. Response:", response.json())
        else:
            print("Failed to execute mutation. Status Code:", response.status_code, "Response:", response.json())
    except:
        print("Exception while reporting score...")

MAX_SPEED_MVG_AVRG=6
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

def round_speed(speed):
    if speed < 10:
        return round(speed)
    else:
        return round(speed / 5) * 5

def check_acceleration_trend(accelerations):
    if len(accelerations) < MAX_SPEED_MVG_AVRG:
        return
    
    # Check if the ball is continuously accelerating
    if all(x < y for x, y in zip(accelerations, list(accelerations)[1:])):
        if accelerations[0] < 10 and accelerations[-1] > 20:
            return True
    return False

def check_speed_accelerate_trend(speeds):
    if len(speeds) < MAX_SPEED_MVG_AVRG:
        return False
    
    # Check if the ball is continuously accelerating
    if all(x < y for x, y in zip(speeds, list(speeds)[1:])):
        if speeds[-1] >= 8:
            return True
    return False

def check_speed_decelerate_trend(speeds):
    if len(speeds) < MAX_SPEED_MVG_AVRG:
        return False
    
    # Check if the ball is continuously decelerating
    if all(x > y for x, y in zip(speeds, list(speeds)[1:])):
        if len(speeds) >= 2 and speeds[-1] < 6 and speeds[-2] < 6:
            return True
    return 

def display_frames(thread_running, displayQueue, feedbackQueue):
    score_cntr_l=0
    score_cntr_r=0 
    match_time="00:00"

    cntr = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    pitch_new_height = 310
    pitch_width = 538
    output_height = 384 + pitch_new_height+40+100
    if (ai_settings.video_disabled==False):
        out = cv2.VideoWriter(ai_settings.video_out_folder+f'/output_{ai_settings.recording_id}.avi', fourcc, MAX_FPS, (1100+340, output_height))
    overlay = cv2.imread('score4.png', -1)  # -1 to load with alpha channel
    pitch = cv2.imread('pitch.jpg')
    # Resize pitch to fit the width of the frame while maintaining its aspect ratio
    pitch = cv2.resize(pitch, (pitch_width, pitch_new_height), interpolation=cv2.INTER_AREA)

    total_frames_captures=0
    start_time = time.perf_counter()
    goal_l = None
    goal_r = None
    goalcheck_l = GoalChecker(f'Field_{ai_settings.field_id}_Goal_L', "c:\Develop\Configuration\goal_config.json", "L")
    goalcheck_r = GoalChecker(f'Field_{ai_settings.field_id}_Goal_R', "c:\Develop\Configuration\goal_config.json", "R")
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
    ball_time_on_left=0
    ball_time_on_right=0
    currentAction = None
    while thread_running.value == 1:
        if goal_hold_back_l != -1:
            goal_hold_back_l -= 1
        if goal_hold_back_r != -1:
            goal_hold_back_r -= 1
        overall_elapsed = (time.perf_counter() - start_time) * 1000
        expedcted_elapsed = total_frames_captures * (1000/MAX_FPS)
        wait = expedcted_elapsed  - overall_elapsed
        if wait > 0:
            time.sleep(wait / 1000)

        total_frames_captures+=1
        if not displayQueue.empty():
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
                frame = overlay_transparent(frame,overlay,10,354)
                combined_frame = np.zeros((output_height, 1100+340, 3), dtype=np.uint8)
                
                # Place the pitch image below the original frame
                combined_frame[404:404+pitch_new_height, 281:281+pitch_width] = pitch
                
                if ball_time_on_right + ball_time_on_left>0:
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
                pitch_shift_x=281
                pitch_shift_y=423
                
                cv2.line(combined_frame,(20+3+pitch_shift_x,17+pitch_shift_y+96),(20 + 3+pitch_shift_x,17+pitch_shift_y+149), (0,0,255),2)
                cv2.line(combined_frame,(520-4+pitch_shift_x,17+pitch_shift_y+96),(520 - 4 +pitch_shift_x,17+pitch_shift_y+149), (0,0,255),2)
                distance_to_L_goal_line = None
                distance_to_R_goal_line = None

                current_distance_to_l= -1
                current_distance_to_r= -1
                total_nbr_of_displayed_players = 0
                player_box_size=20
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
                            printed=True
                            if ball.kicked==True:
                                 ballkicked_cntr = 3
                            
                            print_txt2(combined_frame, 320,770,f'Kick: {ball.kick_text} - players: {total_nbr_of_displayed_players}' )
                            print_txt2(combined_frame, 20,570,f'SPEED: {ball.ball_speed_kmh} km/h')
                            print_txt2(combined_frame, 20,630,f'Ball x,y: {int(ball.x)}, {int(ball.y)}')

                            dir_change_mark=""
                            if ball.ball_direction_change:
                                    dir_change_mark="*"

                            print_txt2(combined_frame, 20,750,f'Ball Direction: {ball.ball_direction} {dir_change_mark}' )
                            print_txt2(combined_frame, 20,770,f'Acceleration: {ball.acceleration} m/s2' )

                            if ballkicked_cntr > 0:
                                ballkicked_cntr -= 1
                                cv2.rectangle(combined_frame,(210,0),(890,400), (0,255,0), -1)

                            if  goal_hold_back_l<goal_holdback_frames-3 and ball.ball_speed_kmh>=goalcheck_l.settings.max_ball_bouncback_speed:
                                goal_hold_back_l = -1
                            if  goal_hold_back_r<goal_holdback_frames-3 and ball.ball_speed_kmh>=goalcheck_r.settings.max_ball_bouncback_speed:
                                goal_hold_back_r = -1
                                
                            if  ball.acceleration<goalcheck_l.settings.max_ball_negative_acceleration:
                                goal_hold_back_l = -1     
                            if  ball.acceleration<goalcheck_r.settings.max_ball_negative_acceleration:
                                goal_hold_back_r = -1
                            distance_to_L_goal_line = round(distance_point_to_line(3,96,3,149,ball.x, ball.y)/500*25,1)
                            if (ball.x <3 and ball.x>-12 and ball.y<142 and ball.y>98):
                                distance_to_L_goal_line = 0
                            distance_to_R_goal_line = round(distance_point_to_line(496,96,496,149,ball.x, ball.y)/500*25,1)
                            if (ball.x >496 and ball.x<514 and ball.y<142 and ball.y>98):
                                distance_to_R_goal_line = 0

                            if  goal_hold_back_l<goal_holdback_frames-5 and distance_to_L_goal_line>2.2:
                                goal_hold_back_l = -1
                            if  goal_hold_back_r<goal_holdback_frames-5 and distance_to_R_goal_line>2.2:
                                goal_hold_back_r = -1
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
                if goal_l is not None:
                    if take_ref_img_l==True:
                        take_ref_img_l=False
                        cv2.imwrite(f'{goalcheck_l.settings.ref_frame_folder}ref_{total_frames_captures}_goal.png', goal_l)
                        goalcheck_l.reference_frame = goal_l
                    goal_scored, goal_l_frame, goal_check_l_h, goal_check_l_w, goal_check_frame_l = goalcheck_l.check(goal_l, stat.detection_cycle, 0)
                    if goal_scored: 
                        if (current_distance_to_l<0.2 and current_distance_to_l>=0):
                            if goal_hold_back_l==-1:
                                goal_hold_back_l = goal_holdback_frames
                    goal_l_res = cv2.resize(goal_l_frame, (200, 200), interpolation=cv2.INTER_LINEAR)
                    combined_frame[10:210, 10:210] = goal_l_res
                   
                if goal_r is not None:
                    if take_ref_img_r==True:
                        take_ref_img_r=False
                        cv2.imwrite(f'{goalcheck_r.settings.ref_frame_folder}ref_{total_frames_captures}_goal.png', goal_r)
                        goalcheck_r.reference_frame = goal_r

                    goal_scored, goal_r_frame, goal_check_r_h, goal_check_r_w, goal_check_frame_r = goalcheck_r.check(goal_r, stat.detection_cycle, 0)
                    if goal_scored:
                        if (current_distance_to_r < 0.2 and current_distance_to_r >= 0):
                            if goal_hold_back_r == -1:
                                goal_hold_back_r = goal_holdback_frames
                    goal_r_res = cv2.resize(goal_r_frame, (200, 200), interpolation=cv2.INTER_LINEAR)
                    combined_frame[40:240, 890+340:1090+340] = goal_r_res

                if goal_check_frame_l is not None:
                    combined_frame[240:240+goal_check_l_h, 10:10+200] = goal_check_frame_l[0:goal_check_l_h, 0:200]

                if goal_check_frame_r is not None:
                    combined_frame[240:240+goal_check_r_h, 890+340:890+200+340] = goal_check_frame_r[0:goal_check_r_h,goal_check_r_w-200:300 ]

                if current_distance_to_l > 3:
                    goal_hold_back_l = -1
                if current_distance_to_r > 3:
                    goal_hold_back_r = -1

                if goal_hold_back_l == 0:
                    is_confirmed = goalcheck_l.do_last_look(ball_pos_history, GOAL_HOLDBACK_FRAME_CNTR)
                    if (is_confirmed):
                        score_cntr_l += 1
                        goal_hold_back_l =- 1
                        goalcheck_l.init_goal_stop()
                        with open(ai_settings.video_out_folder+f'/highlights_{ai_settings.recording_id}.txt', 'a') as file_highlights:
                            rtime = stat.detection_cycle-VIDEO_SHIFT_SEC
                            file_highlights.write(f'GOAL,{rtime}, {score_cntr_l}:{score_cntr_r}\n')
                        update_sportunity_scores(ai_settings.sportunity_id, score_cntr_l, score_cntr_r)

                if goal_hold_back_r ==0:
                    is_confirmed = goalcheck_r.do_last_look(ball_pos_history, GOAL_HOLDBACK_FRAME_CNTR)
                    if (is_confirmed):
                        score_cntr_r+=1
                        goal_hold_back_r=-1
                        goalcheck_r.init_goal_stop()
                        with open(ai_settings.video_out_folder+f'/highlights_{ai_settings.recording_id}.txt', 'a') as file_highlights:
                            rtime = stat.detection_cycle-VIDEO_SHIFT_SEC
                            file_highlights.write(f'GOAL,{rtime}, {score_cntr_l}:{score_cntr_r}\n')
                        update_sportunity_scores(ai_settings.sportunity_id, score_cntr_l, score_cntr_r)                        

                minutes = stat.detection_cycle // 60
                seconds = stat.detection_cycle % 60
                time_string = f"{minutes:02}:{seconds:02}"
                print_txt_dyn(combined_frame,100,30,str(score_cntr_l),1,2)
                print_txt_dyn(combined_frame,970+340,30,str(score_cntr_r),1,2)

                match_time = time_string
                cv2.putText(frame, time_string, (24, 373), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], thickness=1)      
                cv2.putText(frame, str(score_cntr_l)+" : "+str(score_cntr_r), (200, 373), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,255], thickness=2)
                combined_frame[0:384, 230:870+340] = frame

                cv2.rectangle(combined_frame,(830,400),(1215,800), (80,40,40), -1)
                ic=0
                idx =stat.detection_cycle*25+position
                img_displayed=False
                extra_sh=0
                for i in range(1,12):
                    last_action = currentAction.get_action(i, idx)
                    if last_action is not None:
                        txt=last_action.get_short_string()
                        if img_displayed==False:

                            last_xx = 0
                            last_yy = 0
                            for (bx, by) in last_action.ball_data:
                                xx = int(20 + bx) + pitch_shift_x
                                yy = int(pitch_shift_y + 17 + by)
                                if last_xx > 0:
                                    cv2.line(combined_frame, (xx,yy), (last_xx, last_yy),(0,0,255),1,cv2.LINE_AA)
                                last_xx = xx
                                last_yy = yy
                        
                        thi=1
                        ccc = [0,255,255]
                        if "LONG" in txt:
                            ccc = [255,220,220]
                        if "MEDIUM" in txt:
                            ccc = [180,255,180]
                        if "GOAL" in txt:
                            ccc = [180,180,255]
                        cv2.putText(combined_frame, f'{txt}', (830,420 + ic * 20 + extra_sh), cv2.FONT_HERSHEY_PLAIN, 0.8, 
                                    ccc, thickness=thi)
                        ic += 1
                        extra_sh = 15
                    
                print_txt2(combined_frame, 20,530,f'DET. RATE: {stat.overall_detection_rate}%')
                print_txt2(combined_frame, 20,550,f'CYCLE: {stat.detection_cycle}')
                print_txt2(combined_frame, 20,590, f'Best Cam.: CAM{stat.selected_cam}')
                print_txt2(combined_frame, 20,610, f'FPS: {stat.total_frames}/sec')

                start_pos = 440
                print_txt(combined_frame, 890+340, start_pos, f'Camera feeds: {len(stat.cam_info)}')
                print_txt(combined_frame, 890+340, start_pos+15, f'Total Process Time: {stat.process_time} ms')
                print_txt(combined_frame, 890+340, start_pos+30, f'AI processed frames: {stat.total_detections}/sec')
                print_txt(combined_frame, 890+340, start_pos+45, f'Ball detected frames: {stat.ball_detected_frames}',[255,180,180])
                row = 0
                for info in stat.cam_info:
                    row += 1
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
                    break
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
                    break
            time.sleep(0.005)
    thread_running.value = 0
    print("Display thread stopped")
    
if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    print("Stopping...")
    os._exit(0)
