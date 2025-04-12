import sys
import time
import numpy as np
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
import cv2
import queue
import numpy as np
from scipy import stats
import logging
from ultralytics import YOLO
import glob
import math
import os
import copy
#from playeranalyzer import *
from playeranalyzer import *

# Import the "hello_world_pb2.py" file that we have just generated from the
# proto_messages directory 
import proto_messages.players_pb2 as players_pb2


cv=0
cycle=0
object_queue = queue.Queue()

class QueueObj:
    def __init__(self):
        self.img=None
        self.players = None
        self.position = None
        self.cycle = None
        self.images = None

class QueuePlayer:
    def __init__(self):
        self.foot_x = -1
        self.foot_y = -1
        self.is_within = False


ccck=0

analyzer = PlayerAnalyzer("yolov8l-seg.engine", "yolov8m-pose.engine", "tracker_imgs")


REGISTER_RECTANGLE=[(150,230),(350,230),(350,300),(150,300)]

class PlayerObj:
    def __init__(self):
        self.x1=-1
        self.y1=-1
        self.x2=-1
        self.y2=-1
        self.x_2d=-1
        self.y_2d=-1
        self.conf = -1
        self.img = None
        self.img_w = -1
        self.img_h = -1
        self.id = -1
        self.is_within=False
        self.features = None

class TackedPlayer:
    def __init__(self):
        self.player_id=-1
        self.current_x=-1
        self.current_y=-1
        self.current_height=-1
        self.curent_width=-1
        self.features = None
        self.current_pos = -1
        self.image = None
        self.id = None

class Tracker:
    def __init__(self):
        self.dict=[]
        self.last_cycle = -1
        self.tracked_players=[]
        self.measure_time = ProcTimer()
        
    def in_register_rect(self, x, y):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = REGISTER_RECTANGLE

        # Since this is a rectangle, we can just use the opposite corners
        # (x1, y1) and (x3, y3) or (x2, y2) and (x4, y4)

        # Calculate the minimum and maximum x and y coordinates
        min_x = min(x1, x3)
        max_x = max(x1, x3)
        min_y = min(y1, y3)
        max_y = max(y1, y3)

        # Check if the point (x, y) is within the rectangle bounds
        return min_x <= x+20 <= max_x and min_y <= y+32 <= max_y

    def match(self, cycle, pos, x1, y1, x2, y2, x_2d, y_2d):
        for tracked in self.tracked_players:
            dist=min(abs(x_2d-tracked.current_x),abs(y_2d-tracked.current_y))

    def get_tracked_player(self, id):
        if len(self.tracked_players)<id:
            return None
        return self.tracked_players[id-1]

    def rgb_to_hsv(self, r, g, b):
        #print("rgb_to_hsv()", r, g, b)
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        diff = mx - mn
        
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / diff) + 240) % 360
            
        if mx == 0:
            s = 0
        else:
            s = (diff / mx) * 100
            
        v = mx * 100
        return h, s, v
    
    def get_color_diff(self,col1, col2):
        if (col1 is None or col2 is None):
            return 255
        p_b, p_g, p_r = col1
        t_b, t_g, t_r = col2
        #p_h, p_s, p_v = self.rgb_to_hsv(p_r, p_g, p_b)
        #t_h, t_s, t_v = self.rgb_to_hsv(t_r, t_g, t_b)
        #return int((abs(p_h- t_h)+abs(p_s- t_s)+abs(p_v- t_v))/3)
        return int((abs(p_r- t_r)+abs(p_g- t_g)+abs(p_b- t_b))/3)

    def validate_colors(self, player_feat, tracked_feat):
        #print("validate_colors",player_feat.shirt_color1, player_feat.shirt_color2, tracked_feat.shirt_color1, tracked_feat.shirt_color2 )
        if player_feat.shirt_color1 is None or tracked_feat.shirt_color1 is None:
            return 255

        shirt1 = self.get_color_diff(player_feat.shirt_color1, tracked_feat.shirt_color1)
        shirt2 = self.get_color_diff(player_feat.shirt_color2, tracked_feat.shirt_color2)
        shirt3 = self.get_color_diff(player_feat.shirt_color1, tracked_feat.shirt_color2)
        shirt_diff = min(shirt1, shirt2, shirt3)
        
        pants1 = self.get_color_diff(player_feat.pants_color1, tracked_feat.pants_color1)
        pants2 = self.get_color_diff(player_feat.pants_color2, tracked_feat.pants_color2)
        pants3 = self.get_color_diff(player_feat.pants_color1, tracked_feat.pants_color2)
        pants_diff = min(pants1, pants2, pants3)

        #print(f'S diff:{shirt_diff}, p_diff:{pants_diff}')

        if pants_diff==255:
            return 255
        
        return int((pants_diff+shirt_diff)/2)

    def track(self, cycle, pos, AIProcessed):
        for tracked in self.tracked_players:
            coll=[]
            for player in self.dict[-1]:
                dist_x = (player.x_2d - tracked.current_x)
                dist_y = (player.y_2d - tracked.current_y)
                dist = int(math.sqrt((dist_x)**2 + (dist_y)**2))

                coll.append((dist, player))

            coll.sort(key=lambda x: x[0])
            if len(coll)==0:
                print("NO VISIBLE PLAYERS!")
                continue

            dist1, pl1 = coll[0]
            dist2= None
            pl2 = None
            if (len(coll)>1):
                dist2, pl2 = coll[1]

            #print(f"{cycle}, {[pos]} Distances", dist1, dist2)
            MAX_DISTANCE=-15
            SECOND_MAX_DISTANCE = MAX_DISTANCE*2
            MAX_COLOR_DIFF=40

            if (dist1<MAX_DISTANCE and (dist2 is None or dist2>SECOND_MAX_DISTANCE)):
                tracked.current_x = pl1.x_2d
                tracked.current_y = pl1.y_2d

                pl1.id = tracked.id
                print(f"{cycle}, {[pos]} tracker updated", dist1, dist2)
            elif AIProcessed and (dist1<MAX_DISTANCE and (dist2 is None or dist2>SECOND_MAX_DISTANCE)):
                color_diff = self.validate_colors(pl1.features, tracked.features)
                if color_diff<MAX_COLOR_DIFF:
                    tracked.current_x = pl1.x_2d
                    tracked.current_y = pl1.y_2d

                    pl1.id = tracked.id
                    print(f"{cycle}, {[pos]} tracker updated[AI]", color_diff, dist1, dist2)

            elif AIProcessed==True:
                print(f"{cycle}, {[pos]}...discovery...", len(coll))
                #tracked.features.print()

                closest_few = coll#coll[:6]
                closest_color = None
                closest_color_player=None
                closest_color_distance = None
                for dist, player in closest_few:
                    
                    color_diff = self.validate_colors(player.features, tracked.features)
                    print(f" Distance: {dist}, ColorDiff:{color_diff}")
                    if (closest_color is None or closest_color>color_diff):
                        closest_color = color_diff
                        closest_color_player = player
                        closest_color_distance = dist

                if closest_color is not None and closest_color<MAX_COLOR_DIFF:
                    tracked.current_x = closest_color_player.x_2d
                    tracked.current_y = closest_color_player.y_2d
                    closest_color_player.id = tracked.id
                    print("DISCOVERED - tracker updated. Color diff:", closest_color, "distance:", closest_color_distance)
                else:
                    print("Closest color", closest_color)
            else:
                print("Skipped processing.", dist1, dist2)


    def capture(self, images, cycle, pos, AIProcessed=False):
        #idx = cycle*25 + pos
        this_collection=[]
        for imageObj in images:
            plObj = PlayerObj()
            plObj.x1=imageObj.playerObj.x1
            plObj.y1=imageObj.playerObj.y1
            plObj.x2=imageObj.playerObj.x2
            plObj.y2=imageObj.playerObj.y2
            plObj.conf=imageObj.playerObj.confidence
            plObj.x_2d=imageObj.playerObj.x_2d
            plObj.y_2d=imageObj.playerObj.y_2d

            plObj.features = HumanFeatures()
            plObj.features.shirt_color1 = imageObj.features.shirt_color1
            plObj.features.shirt_color2 = imageObj.features.shirt_color2
            plObj.features.pants_color1 = imageObj.features.pants_color1
            plObj.features.pants_color2 = imageObj.features.pants_color2
            plObj.features.socks_color1 = imageObj.features.socks_color1
            plObj.features.socks_color2 = imageObj.features.socks_color2

            plObj.id = 0
            #plObj.features.print()

            if AIProcessed==True and len(self.tracked_players)==0 and self.in_register_rect(plObj.x_2d, plObj.y_2d):
                tracked_player=TackedPlayer()
                tracked_player.current_x=plObj.x_2d
                tracked_player.current_y=plObj.y_2d
                tracked_player.curent_width = plObj.x2-plObj.x1
                tracked_player.current_height = plObj.y2-plObj.y1
                tracked_player.features = HumanFeatures()
                tracked_player.features.shirt_color1 = plObj.features.shirt_color1
                tracked_player.features.shirt_color2 = plObj.features.shirt_color2
                tracked_player.features.pants_color1 = plObj.features.pants_color1
                tracked_player.features.pants_color2 = plObj.features.pants_color2
                tracked_player.features.socks_color1 = plObj.features.socks_color1
                tracked_player.features.socks_color2 = plObj.features.socks_color2
                tracked_player.id = len(self.tracked_players) + 1
                tracked_player.current_pos = pos
                tracked_player.image = imageObj.resized

                self.tracked_players.append(tracked_player)
                print("Player registered")
                tracked_player.features.print()



            this_collection.append(plObj)
        self.dict.append(this_collection)
        if len(self.dict) > 10:
            self.dict.pop(0)

        self.track(cycle, pos, AIProcessed)

        return this_collection


tracker = Tracker()

# Callback for receiving messages
def callback(topic_name, msg, arr_time):
    global cv, cycle, pose_analyzer, object_queue
    
    cv+=1
    if cycle!=msg.cycle:
        cycle=msg.cycle
        print("RX", msg.cycle, msg.position)

    frame_array = np.frombuffer(msg.img, dtype=np.uint8)
    m_frame = frame_array.reshape((msg.img_h, msg.img_w, 3))

    q_obj = QueueObj()
    q_obj.img = m_frame.copy()
    q_obj.cycle = msg.cycle
    q_obj.position = msg.position
    q_obj.players = []



    images=[]
    for player in msg.players:
        if player.x_2d>=0 and player.y_2d>=0:
            
            #print(f"Coordinates: ({player.x1}, {player.y1}) to ({player.x2}, {player.y2})")
            #print(f"2D Coordinates: ({player.x_2d}, {player.y_2d})")
            #print(f"Confidence: {player.confidence}")
            #print(f"Image Size: {player.img_w}x{player.img_h}")
            frame_array = np.frombuffer(player.img, dtype=np.uint8)
            #frame_array.setflags(write=1)
            frame = frame_array.reshape((player.img_h, player.img_w, 3))
            #cv2.imwrite(f'cycle_{cv}.png',frame)

            '''qPlayer=QueuePlayer()
            qPlayer.foot_x = player.x_2d
            qPlayer.foot_y = player.y_2d
            plObj = tracker.add(msg.cycle, msg.position, player.x1, player.y1, player.x2, player.y2, player.x_2d, player.y_2d,
                        player.confidence,player.img_w, player.img_h, frame)

            qPlayer.is_within = plObj.is_within
            q_obj.players.append(qPlayer)'''
            image_obj = ImageObject(frame)  # Pass the image and filename
            image_obj.filename = f'C{msg.cycle}F{msg.position}P{len(images)}.png'
            image_obj.playerObj = player
            images.append(image_obj)

            #cv2.rectangle(q_obj.img, (player.x1, player.y1),(player.x2, player.y2) ,(0,0,255), 2)

    analyzer.analyze(images, debug=True)
    '''if msg.position %3==0:
        timer =  ProcTimer()
        if analyzer.analyze(images, debug=True, segment_enabled=False, cycle=msg.cycle, position=msg.position):
            players = tracker.capture(images, msg.cycle, msg.position, AIProcessed=True)
        timer.stop("---->>>>>AI Colors")
    else:
        players = tracker.capture(images, msg.cycle, msg.position, AIProcessed=False)'''

    '''q_obj.players = players
    q_obj.images = copy.deepcopy(images)
    object_queue.put(q_obj)'''

if __name__ == "__main__":
    #init AI
    black = np.zeros((100,100, 3), dtype=np.uint8)
    image_obj = ImageObject(black)  # Pass the image and filename
    image_obj.filename = ""
    images=[]
    images.append(image_obj)                
    #analyzer.analyze(images, debug=False, segment_enabled=False, cycle=0, position=0)
    analyzer.analyze(images, debug=False,  segment_enabled=False)
    

    # initialize eCAL API. The name of our Process will be
    # "Python Protobuf Subscriber"
    ecal_core.initialize(sys.argv, "Python Protobuf Subscriber")

    # Create a Protobuf Publisher that publishes on the topic
    # "hello_world_python_protobuf_topic". The second parameter tells eCAL which
    # datatype we are expecting to receive on that topic.
    sub = ProtoSubscriber("Players"
                        , players_pb2.FootPlayerCollection)

    # Set the Callback
    sub.set_callback(callback)
    

    pitch_new_height = 310
    pitch_width = 538
    pitch = cv2.imread('pitch.jpg')
    # Resize pitch to fit the width of the frame while maintaining its aspect ratio
    pitch = cv2.resize(pitch, (pitch_width, pitch_new_height), interpolation=cv2.INTER_AREA)
    points = np.array(REGISTER_RECTANGLE, np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(pitch, [points], True, (0, 255, 0), 2)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('DEBUG.avi', fourcc, 25, (640+340, 800))


    # Just don't exit
    cntr=0
    while ecal_core.ok():
        time.sleep(0.025)
        try:
            #
            if not object_queue.empty():
                # Get the next frame from the queue
                q_obj = object_queue.get()
                if object_queue.qsize()>26:
                    print("Drop")
                    q_obj = object_queue.get()

                player_len = len(q_obj.players)
                frame = cv2.resize(q_obj.img, (640+340, 384), interpolation=cv2.INTER_LINEAR)
                combined_frame = np.zeros((800, 640+340, 3), dtype=np.uint8)
                combined_frame[0:384,0:640+340]=frame

                cv2.putText(combined_frame,  f"{q_obj.cycle}|{q_obj.position}", (30,30), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,255], thickness=3)

                combined_frame[400:400+pitch_new_height,0:pitch_width]=pitch
                #cv2.putText(frame,f"{player_len}", (100,100),cv2.FONT_HERSHEY_PLAIN,3, [0,255,100],thickness=2)

                player_box_size=20
                pitch_shift_x=0
                pitch_shift_y=400+15
                for player in q_obj.players:
                    if player.x_2d>0 and player.y_2d>0:
                        xx = int(20 + player.x_2d+pitch_shift_x - player_box_size / 2)
                        yy = int(pitch_shift_y+17+player.y_2d - player_box_size / 2 )
                        ccol=(250,160,0)
                        if (player.is_within):
                            ccol=(100,100,255)
                        
                        cv2.rectangle(combined_frame,(xx,yy),(xx+player_box_size,yy+player_box_size),ccol, -1)
                        cv2.putText(combined_frame,  f"{player.id:02}", (xx+2, yy+12), cv2.FONT_HERSHEY_PLAIN, 0.9, [0,0,255], thickness=1)

                
                player1 = tracker.get_tracked_player(1)
                if player1 is not None:
                    h,w = player1.image.shape[:2]
                    combined_frame[800-h:800, 980-w:980]=player1.image


                if len(q_obj.images)>0:
                    h,w = q_obj.images[0].resized.shape[:2]
                    combined_frame[800-h:800, 700-w:700]=q_obj.images[0].resized
                out.write(combined_frame)
                cv2.imshow("X", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        except:
            pass
        cntr+=1
        if cntr>24:
            cntr=0

    out.release()
    # finalize eCAL API
    ecal_core.finalize()