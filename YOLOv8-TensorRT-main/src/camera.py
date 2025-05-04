import threading
import numpy as np
import multiprocessing as mp
import cv2
import time
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
import torch
from collections import defaultdict
from enum import Enum
import time
from src.utils import *
from queue import Queue
from src.gpu import *
from src.cameramapper import *
import math

class Quality(Enum):
    HIGH = 1
    LOW = 2
    MEDIUM = 3

class BallCoordsObj2:
    def __init__ (self, cam_id, x, y,  x2d, y2d, position, confidence):
        self.x =x
        self.y =y
        self.x_2d =x2d
        self.y_2d =y2d
        self.camera_id =cam_id
        self.position = position
        self.confidence = confidence

class BallCoordsObj:
    def __init__ (self, cam_id, x, y, x_3d, y_3d, radius, ball_speed_kmh, position, direction, acceleration,kick_detected, kicked, kick_text, ball_direction_change):
        self.x =x
        self.y =y
        self.x_3d =x_3d
        self.y_3d =y_3d
        self.radius = radius
        self.camera_id =cam_id
        self.position = position
        self.ball_speed_kmh = ball_speed_kmh
        self.ball_direction = direction
        self.acceleration = acceleration
        self.kick_detected = kick_detected
        self.kicked = kicked
        self.kick_text= kick_text
        self.ball_direction_change = ball_direction_change
    
    def to_dict(self):
        try:
            xx = int(self.x)
            yy = int(self.y)
            x3d = int(self.x_3d)
            y3d = int(self.y_3d)
            rr = int(self.radius)
        except:
            xx=-1
            yy=-1
            x3d=-1
            y3d=-1
            rr=-1

        return {
            'x': xx,
            'y': yy,
            'x_3d': x3d,
            'y_3d': y3d,
            'radius': rr,            
            'cam_id': int(self.camera_id),
            'pos': int(self.position)
        }        
    
class BallCoords:
    def __init__ (self):
        self.data = []

    def get_ball_data_by_position(self, position):
        for ball_data in self.data:
            if ball_data.position == position:
                return ball_data
        return None        

class CamInfo:
    def __init__ (self):
        self.name = ""
        self.id = 0
        self.total_frames = 0
        self.total_detections = 0
        self.ball_detected_frames = 0
        self.players = 0

class OverallStats:
    def __init__ (self):
        self.cycle_cntr = 0
        self.ball_captured = 0
        self.detection_rate = 0

class DetectorStats:
    def __init__ (self):
        self.total_frames = 0
        self.ball_detected_frames = 0
        self.total_detections = 0
        self.detection_cycle = 0
        self.process_time = 0
        self.cam_info = []
        self.selected_cam = 0
        self.overall_detection_rate = 0

class Frame:
    def __init__(self, frame, camera_id, frame_id, timestamp, slave = False):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.slave = slave
        self.frame = frame

class PlayerCollection:
    def __init__(self):
        self.data={}
        self.last_good_collection = None
        self.last_good_collection_ttl=-1
    def add_to_collection(self, key, value):
        if key not in self.data:
            self.data[key] = []  # Initialize an empty list for this key if it doesn't exist
        self.data[key].append(value)        

    def get_collection(self, key):
        players =  self.data.get(key, [])  # Return the collection if key exists, otherwise return an empty list
        
        if (len(players) > 0):
            self.last_good_collection = players
            self.last_good_collection_ttl = 3
        else:
            if self.last_good_collection_ttl>0:
                self.last_good_collection_ttl-=1
                players = self.last_good_collection
        return players
    
    def are_duplicates(self, player1, player2):
        return False

class Player:
    def __init__(self):
        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.confidence = 0
        self.x_2d = -1
        self.y_2d = -1
        self.player_id = -1
        self.closest = False
        self.ball_distance = -1
        self.ball_x=-1
        self.ball_y=-1
        self.cam_id = -1
        self.centerx=-1
        self.centery=-1
        self.last_used_idx=-1
        self.img = None

class Detector:
    def __init__(self,camera_id, frame, frame_id, timestamp, detection_type=DetectionType.NORMAL):
        self.x1 = -1
        self.y1 = -1
        self.x2 = -1
        self.y2 = -1
        self.confidence = 0
        self.x_2d = -1
        self.y_2d = -1
        self.ball_radius = -1
        self.ball_speed_ms = -1
        self.ball_speed_kmh = -1
        self.ball_direction = -1

        self.players = []
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.processed = False
        self.detection_type = detection_type
        self.skip = True
        self.quality = Quality.MEDIUM
        self.region = ""
        self.tensor = None
        self.detect = True
        self.frame = frame
        self.position = 0
        self.ball = 0
        self.people = 0
        self.mean_saturation = -1
        self.mean_value = -1
        self.white_gray = -1
        self.in_progress=False
        self.timestamp = timestamp
        self.fake = False

class Camera:
    def __init__(self, field_id, id, path, thread_running,frame_start, max_fps, feed_fps, description, detection_type = DetectionType.NORMAL , capture_only_device = False, slaves=[]):
        self.next_frame_please = mp.Value('i', 7)
        self.output_queue =  mp.Queue()
        self.slaves = slaves
        self.frame_id = 0
        self.detection_type = detection_type
        self.description = description
        self.capture_only_device = capture_only_device
        self.path = path
        self.camera_id = id
        self.fake = False
        self.detector_objects = defaultdict(list)
        self.thread_running = thread_running
        self.feed_fps = feed_fps
        self.max_fps = max_fps
        self.coll  =[]
        self.blank_image = np.zeros((384,640,3), np.uint8)
        self.executor_futures = None
        self.executor_cycle = -1
        self.engine = None
        self.device = None
        self.W = 0
        self.H = 0
        self.frame_w = 0
        self.frame_h = 0
        self.callback_cntr = 0
        self.transformation_matrix = None
        self.reference_frame_saved = False
        self.current_live_cycle = -1

        self.camera_mapper = CameraMapper(f'mapping_field{field_id}_cam{self.camera_id}.json')
        sub = ProtoSubscriber(f'video_topic_{self.camera_id}'
                            , frame_pb2.FrameData)
        sub.set_callback(self.callback)
        if path != "":
            pass
        else:
            print(f'*** WARNING: CAM {self.camera_id} is a Slave stream only! Used for storage only.')
        if capture_only_device:
            print(f'*** WARNING: CAM {self.camera_id} is a capture only device providing frames for slaves!')
        self.detector_cycle_current = 1

    def set_executor_cycle(self, cycle):
        self.executor_cycle = cycle

    def callback(self, topic_name, frame_proto_msg, time):
            channels = 3
            if self.current_live_cycle<frame_proto_msg.cycle:
                self.current_live_cycle = frame_proto_msg.cycle
            frame_array = np.frombuffer(frame_proto_msg.frame, dtype=np.uint8)
            frame = frame_array.reshape((frame_proto_msg.height, frame_proto_msg.width, channels))
            if (self.camera_id == 10 or self.camera_id == 11):
                frame = self.enhance_contrast_and_whiten(frame).copy()
                if frame is None:
                    return
            self.frame_w = frame_proto_msg.width
            self.frame_h = frame_proto_msg.height
            frm = Frame(frame, frame_proto_msg.camera_id, frame_proto_msg.frame_id, frame_proto_msg.unix_timestamp)
            self.register_frame(frm, frame_proto_msg.cycle, frame_proto_msg.position)
            self.callback_cntr += 1
            if self.reference_frame_saved == False:
                self.reference_frame_saved = True
                cv2.imwrite(f'XXX{self.camera_id}_{self.callback_cntr}.png', frame)

    def apply_white_balance(self,image):
        # Convert the image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the channels
        l, a, b = cv2.split(lab)
        
        # Equalize the L channel (lightness)
        l = cv2.equalizeHist(l)
        
        # Merge the channels back
        lab = cv2.merge((l, a, b))
        
        # Convert back to BGR color space
        balanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return balanced_image

    def increase_contrast(self, image, factor=1.15):
        # Convert the image to float32 for better precision
        image = np.float32(image)
        
        # Increase contrast by the given factor
        contrasted_image = image * factor
        
        # Clip the values to [0, 255] range
        contrasted_image = np.clip(contrasted_image, 0, 255)
        
        # Convert back to uint8
        contrasted_image = np.uint8(contrasted_image)
        
        return contrasted_image

            #global_executor.submit(self.check,self.executor_cycle)
    def enhance_contrast_and_whiten(self, image):
        if image is None:
            return None

        # Convert to YUV color space
        yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Stretch the histogram of the Y channel to increase contrast
        y_channel = yuv_img[:, :, 0]

        # Find the 1st and 99th percentile to avoid outliers
        min_val, max_val = np.percentile(y_channel, (1, 99))
        y_channel = np.clip((y_channel - min_val) * (255 / (max_val - min_val)), 0, 255)
        yuv_img[:, :, 0] = y_channel.astype(np.uint8)

        # Convert back to BGR color space
        enhanced_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

        # Scale the image to make the brightest point white (255, 255, 255)
        min_val = np.min(enhanced_img)
        max_val = np.max(enhanced_img)
        enhanced_img = ((enhanced_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Ensure true white (255, 255, 255)
        white_pixel_indices = np.all(enhanced_img == [255, 255, 255], axis=-1)
        enhanced_img[white_pixel_indices] = [255, 255, 255]

        return enhanced_img

    def convert_to_3d(self, x_2d, y_2d):

        x,y = self.camera_mapper.estimate_3d_coordinate(x_2d, y_2d)
        return x,y

    def convert_to_2d(self, det_obj):
        if (det_obj.confidence>0):
            x = (det_obj.x1+det_obj.x2)/2
            y = (det_obj.y1+det_obj.y2)/2
            
            #if self.camera_id==3:
            #    x=x-150
            
            det_obj.ball_radius = min((det_obj.x2 - det_obj.x1), (det_obj.y2 - det_obj.y1)) / 2

            det_obj.x_2d, det_obj.y_2d = self.camera_mapper.estimate_2d_coordinate(x, y)

    def convert_player_to_2d(self, x, y):
        x_2d, y_2d = self.camera_mapper.estimate_2d_coordinate(x, y)
        return x_2d, y_2d

    def print_detector_obj(self, obj):
        if obj is not None:
            isfake=""
            if obj.fake==True:
                isfake=", FAKE"
        else:
            print("None")

    def finalize_detector_array(self, cycle):
        sorted_detectors = sorted(self.detector_objects[cycle], key=lambda det: det.position)
        final_array = [None] * (self.max_fps)
        for det in sorted_detectors:
            final_array[det.position] = det
        self.detector_objects[cycle] = final_array    

    def get_cycle_min_max_times(self):
        if len(self.detector_objects[self.detector_cycle_current])==0:
            return None, None
        min_timestamp = min(self.detector_objects[self.detector_cycle_current], key=lambda x: x.timestamp).timestamp
        max_timestamp = max(self.detector_objects[self.detector_cycle_current], key=lambda x: x.timestamp).timestamp
        return min_timestamp, max_timestamp

    def register_frame(self, frame_object, cycle, position):
        detector = Detector(self.camera_id, frame_object.frame, frame_object.frame_id, frame_object.timestamp, self.detection_type)
        detector.position = position
        self.detector_objects[cycle].append(detector)

    def get_goal_frame(self, detector_cycle, pos):
        try:
            if self.detector_objects[detector_cycle][pos] is None or self.detector_objects[detector_cycle][pos].frame is None:
                return None

            det_obj = self.detector_objects[detector_cycle][pos]
            frame = det_obj.frame
            if (det_obj.x1!=-1):
                x = int((det_obj.x1+det_obj.x2)/2)
                y = int((det_obj.y1+det_obj.y2)/2)
                rad = int(abs(det_obj.x2-det_obj.x1)/2)
                cv2.circle(frame, center=(x, y), radius=rad, color= (0, 0, 255), thickness=-1)
                
            return frame
        except:
            return None

    def detector_cycle_completed(self, detector_cycle):
        cycle_to_delete = detector_cycle #we delete only older once as we still need the data from the one that just completed
        try:
            del self.detector_objects[cycle_to_delete]
        except KeyError:
            print(f'Key {detector_cycle} does not exist in the collection.')

    def reset_segment(self, detector_cycle):
        try:
            lowest_key = min(int(key) for key in self.detector_objects.keys())
            highest_key = max(int(key) for key in self.detector_objects.keys())
            if (lowest_key>0 and highest_key-lowest_key>6):
                self.detector_cycle_completed(lowest_key)
        except:
            print("Error while reetting segment....")
        self.detector_cycle_current = detector_cycle

    def get_frame(self, detector_cycle, frame_pos):
        if len(self.detector_objects[detector_cycle])<=frame_pos:
            return None
        obj= self.detector_objects[detector_cycle][frame_pos]
        if obj is not None:
            return obj.frame
        else:
            return None
        
    def set_detector_obj(self, cycle, pos, value):
        for obj in self.detector_objects[cycle]:
            if obj is not None and obj.position==pos:
                obj.x2=value

    def initial_gap_fill(self, cycle):
        cntr=0
        frame = None
        for obj in self.detector_objects[cycle]:
            if obj is not None:
                frame = obj.frame
            if obj is None and cntr<self.max_fps:
                if frame is None:
                    last, _=self.get_last_value(cycle-1)
                    if last is not None:
                        frame = last.frame

                if frame is None:
                    frame = self.blank_image
                self.detector_objects[cycle][cntr]=Detector(self.camera_id,frame,cntr,0)
                self.detector_objects[cycle][cntr].processed=True
                self.detector_objects[cycle][cntr].fake =True
                self.detector_objects[cycle][cntr].position = cntr
            cntr+=1
        try:
            first = self.detector_objects[cycle][0]
            if first.x1==-1:
                last, last_width_det=self.get_last_value(cycle-1)
                
                if last_width_det is not None:
                    if (last_width_det.position>self.feed_fps-5):
                        self.detector_objects[cycle][0]=Detector(self.camera_id,last.frame,0,0)
                        self.detector_objects[cycle][0].x1 = last_width_det.x1
                        self.detector_objects[cycle][0].x2 = last_width_det.x2
                        self.detector_objects[cycle][0].y1 = last_width_det.y1
                        self.detector_objects[cycle][0].y2 = last_width_det.y2
                        self.detector_objects[cycle][0].confidence = 0.11
                        self.detector_objects[cycle][0].processed=True
                        self.detector_objects[cycle][0].fake =True
                        self.detector_objects[cycle][0].position = 0
                else:
                    print("LD: NONE")
        except:
            print("Error - failed to complete init gap fill...")
    
    def get_last_value(self, cycle):
        last = None
        last_det = None
        for obj in self.detector_objects[cycle]:
            if obj is not None and obj.fake==False:
                last=obj
                if (obj.x1!=-1):
                    last_det=obj

        return last, last_det
    
    def get_detector_collection(self, detector_cycle, detection_type = DetectionType.NORMAL):
        if self.detection_type != detection_type:
            return []
        return self.detector_objects[detector_cycle]
    
    def get_non_none_detector_collection(self, detector_cycle, detection_type=DetectionType.NORMAL):
        if self.detection_type!= detection_type:
            return []
        return [obj for obj in self.detector_objects[detector_cycle] if obj is not None]

    def enhance_frame(self,frame):
        return frame

    def capture(self, frame_start):
        device = torch.device("cuda:0")
        ecal_core.initialize([], f'CAM{self.camera_id} Publisher')
        pubs=[]
        pubs.append(ProtoPublisher(f'video_topic_{self.camera_id}', frame_pb2.FrameData))
        if self.capture_only_device:
            for slave in self.slaves:
                pubs.append(ProtoPublisher(f'video_topic_{slave}', frame_pb2.FrameData))

        temp_storage=[]
        err_cntr=99
        cap=None
        rtsp=False
        fps_step = self.max_fps /  self.feed_fps
        fps_step_cntr = 0
        last_step_nbr = 0
        start_time = None
        last_frame_sync_val = 0
        if self.path.startswith("rtsp"):
            rtsp=True

        first_capture_time = 0
        total_frames_captures = 0
        expected_frame_captures = 0
        frame_time = 1000/self.feed_fps
        while self.thread_running.value==1:
            if self.path=="DUMMY":
                time.sleep(0.1)
                continue
            if (err_cntr>3):
                print("Connecting to source ", self.path)
                cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_POS_MSEC, 2250 * 1000)
                err_cntr=0

            if frame_start.value==0:
                continue            

            if (rtsp==False):
                if start_time is not None:
                    overall_elapsed = (time.perf_counter() - start_time) * 1000
                    expedcted_elapsed = total_frames_captures * (1000/self.feed_fps)
                    wait = expedcted_elapsed  - overall_elapsed
                    if wait>0:
                        time.sleep(wait / 1000)
                if (start_time is None):
                    start_time = time.perf_counter()
            
            success, frame = cap.read()
            total_frames_captures+=1
            if first_capture_time==0:
                first_capture_time = time.perf_counter()*1000
            else:
                expected_frame_captures = round((time.perf_counter()*1000-first_capture_time)/(frame_time))

                if (expected_frame_captures>total_frames_captures):
                    success, frame = cap.read()
                    total_frames_captures+=1

            #-------- skip frames if needed ------
            fps_step_cntr += 1
            curr_fps_nbr = round(fps_step_cntr*fps_step)
            if (curr_fps_nbr == last_step_nbr):
                success, frame = cap.read()
                total_frames_captures += 1
                fps_step_cntr += 1
            last_step_nbr = curr_fps_nbr
            self.frame_id = int((time.perf_counter()*1000 - frame_start.value)/(frame_time))
            if success:
                err_cntr = 0
                timestamp = int(time.perf_counter() * 1000)
                if (self.camera_id==4):
                    res_start = time.perf_counter()
                    original_frame = frame
                    cropped_frame = frame[468:1368, 244:2544]
                    width = 1200
                    height = 384
                    fwidth = 640
                    fheight = 384
                    frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    frame = self.enhance_frame(frame)
                    width_half, height_half = int(width/2), int(height/2)
                    slices = []
                    goal_size = 256
                    slices.append(frame[0:fheight, 0:fwidth].copy())
                    slices.append(frame[0:fheight, width - fwidth:width].copy())
                    slices.append(original_frame[570:(570+goal_size), 370:(370+goal_size)].copy())
                    slices.append(original_frame[630:(630+goal_size), 2080:(2080+goal_size)].copy())
                    self.print_frame(slices[0],"C-Left", self.frame_id, timestamp, 380)
                    self.print_frame(slices[1],"C-Right", self.frame_id, timestamp, 380)
                    both_goals = np.concatenate((slices[2], slices[3]), axis=1)
                    pubs[1].send(create_protobuf(self.frame_id, self.slaves[0], slices[0], timestamp, fwidth, fheight))
                    pubs[2].send(create_protobuf(self.frame_id, self.slaves[1], slices[1], timestamp, fwidth, fheight))
                    pubs[3].send(create_protobuf(self.frame_id, self.slaves[2], both_goals, timestamp, goal_size * 2, goal_size))
                    res_elapsed = (time.perf_counter() - res_start) * 1000
                else:
                    frame = cv2.resize(frame, (640, 384), interpolation=cv2.INTER_LINEAR)
                    frame = self.enhance_frame(frame)
                    self.print_frame(frame,self.description, self.frame_id, timestamp, 380)
                    pubs[0].send(create_protobuf(self.frame_id, self.camera_id, frame, timestamp, 640, 384))
            else:
                err_cntr+=1
                print("Frame capture error. Cntr=", err_cntr)
        
        print("ENDING THREAD")
        if (cap is not None):
            cap.release()
        print("Cap released")
    
    def print_frame(self, frame, desc, frame_id, position, y):
        
        cv2.rectangle(frame, (y-50,y),(640, y-20), (0,0,0), -1)
        cv2.putText(frame,
                    f'{frame_id} | {round(position)} | {desc}', (390,y-5),
                    cv2.FONT_HERSHEY_PLAIN,
                    1, [0,255,0],
                    thickness=1)
