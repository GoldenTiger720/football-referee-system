import math
import time


global action_counter
action_counter=0

class PlayerAction:
    def __init__(self):
        self.distance = -1
        self.match_time = None
        self.max_speed = -1
        self.max_acceleration=-1
        self.min_acceleration=None
        self.start_player_id=None
        self.end_player_id=None
        self.start_player_img=None
        self.end_player_img=None
        self.type = "PASS"
        self.processed = False
        self.ttl=0
        self.goal_detected=False
        self.end_time = None
        self.score_l=0
        self.score_r=0
        self.action_id = -1
        self.index=-1
        self.ball_data=[]
    def age_in_ms(self):
        if self.end_time is None:
            return 0
        
        return (time.time() - self.end_time)*1000

    def get_string(self):
        return f'{self.match_time} - {self.type}. Max Speed: {int(self.max_speed)} km/h, distance: {(self.distance)} m, acc.: [{self.min_acceleration} / {self.max_acceleration} m/s2] Goal: {self.goal_detected}'
    def get_short_string(self):
        txt=""
        #if self.end_player_id is None:
        #    txt=" - MISS"

        if (self.goal_detected):
            txt="-GOAL"
        elif self.start_player_id is None:
            txt = "-BOUNCE"
        return f'[{self.action_id }]{self.match_time} |{len(self.ball_data)}| {self.type}{txt} | {int(self.max_speed)} km/h | {(self.distance)} m '#| {self.goal_detected}|{self.score_l}:{self.score_r}'
class CurrentAction:
    def __init__(self):
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.max_speed = -1
        self.max_acceleration=-1
        self.min_acceleration=None
        self.start_time = None
        self.end_time = None
        self.start_player_id=None
        self.start_player_img=None
        self.match_time = None
        self.action_collection=[]
        self.is_open = False
        self.score_l = 0
        self.score_r = 0
        self.goal_detected=False
        self.action_id = -1
        self.index=-1
        self.ball_data=[]

    def get_last_action(self):
        if len(self.action_collection)>0:
            return self.action_collection[-1]
        return None
    def maintain_size(self):
        if len(self.action_collection) > 20:
            item = self.action_collection.pop(0)
            item.ball_data=[]
            item.start_player_img=None
            item.end_player_img=None      

    def get_action(self, pos, idx):
        
        if len(self.action_collection)>=pos:
            if self.action_collection[-1*pos].index <=idx:
                #if (self.action_collection[-1*pos].start_player_id==None):
                #    return None
                return self.action_collection[-1*pos]
        return None
    def get_current_id(self):
        return self.action_id
    
    def start_action(self, x, y, match_time, cycle, pos):
        global action_counter
        print(f'start_action[{self.action_id}]. cycle',cycle,'pos:', pos, "ball xy:", x, y)
        self.maintain_size()
        if (self.is_open==True):
            self.stop_action(x, y, cycle, pos)
        self.start_x = x
        self.start_y = y
        self.match_time = match_time
        self.end_x = -1
        self.end_y = -1
        self.max_speed = -1
        self.max_acceleration=-1
        self.min_acceleration=None
        #self.start_time = time.now()
        self.end_time = None
        self.is_open=True
        self.goal_detected=False
        action_counter+=1
        self.action_id = action_counter#len(self.action_collection)+1
        self.index = cycle*25+pos
        self.start_player_id=None
        self.start_player_img=None
        self.ball_data=[]
        print("ACTIONSTARTED - opened ", self.action_id)

    def stop_action(self, x, y, cycle, pos):
        print(f'stop_action[{self.action_id}]. cycle:', cycle, 'pos:', pos, "ball xy:", x, y)
        if self.is_open==False:
            print("action not open. Can't close it.")
            return
        self.end_x = x
        self.end_y = y
        #self.end_time = time.now()
        player_action = PlayerAction()

        player_action.distance = self.calculate_distance() #in meters
        player_action.distance = player_action.distance / 500
        player_action.distance = round(player_action.distance * 50, 1)
        if player_action.distance>0.5 and self.max_speed<140: # ignore passes under 1m        
            player_action.end_player_id=None
            player_action.end_player_img=None
            player_action.start_player_id = self.start_player_id
            player_action.start_player_img = self.start_player_img
            player_action.max_speed = self.max_speed
            player_action.max_acceleration = self.max_acceleration
            player_action.min_acceleration = self.min_acceleration
            player_action.match_time = self.match_time
            #player_action.goal_detected = self.goal_detected
            player_action.end_time = time.time()
            player_action.action_id = self.action_id
            player_action.index = cycle*25+pos
            player_action.ball_data = self.ball_data.copy()

            player_action.type="SHORT/DRIBLE"
            if player_action.distance>15:
                player_action.type="MEDIUM PASS"
            if player_action.distance>25:
                player_action.type="LONG PASS"

            self.action_collection.append(player_action)
        print("ACTIONADDED - closed", self.action_id, "dist=",  player_action.distance, f' [{self.start_x},{self.start_y}, {self.end_x}, {self.end_y} ]')#len(self.action_collection))
        self.is_open=False

    def add_player(self, player_id, player_img, cycle, pos):
        print(f'add_player([{self.action_id},cycle: {cycle}, pos: {pos}])')
        idx = cycle*25+pos
        if self.start_player_id is None and abs(self.index-idx)<3:
            self.start_player_id=player_id
            self.start_player_img=player_img

        if len(self.action_collection)>0 and self.action_collection[-1].end_player_id==None and abs(self.action_collection[-1].index -idx)<3:
            self.action_collection[-1].end_player_id=player_id
            self.action_collection[-1].end_player_img=player_img
            print(f'add_player - end [id:{self.action_collection[-1].action_id}]')            

        return
        #if self.is_open==True:
        if self.start_player_id is None:
            self.start_player_id=player_id
            self.start_player_img=player_img
            print(f'add_player - start [id:{self.action_id}]')
            if len(self.action_collection)>0 and self.action_collection[-1].age_in_ms()<300 and self.action_collection[-1].end_player_id==None:
                self.action_collection[-1].end_player_id=player_id
                self.action_collection[-1].end_player_img=player_img
                print(f'add_player - end [id:{self.action_collection[-1].action_id}]')
        else:
            self.end_player_id=player_id
            self.end_player_img=player_img
            
    def add_score(self, score_l, score_r):
        print(f'add_score({score_l}, {score_r})')
       
        #self.score_l = score_l
        #self.score_r = score_r
        if len(self.action_collection)>0:
            self.action_collection[-1].score_l=score_l
            self.action_collection[-1].score_r=score_r
        if len(self.action_collection)>1:
            if (self.action_collection[-1].score_l>self.action_collection[-2].score_l):
                self.action_collection[-1].goal_detected=True
            if (self.action_collection[-1].score_r>self.action_collection[-2].score_r):
                self.action_collection[-1].goal_detected=True

    def add_ball(self, x, y):
        if x>0 and y>0:
            self.ball_data.append((x,y))
    def add_speed(self, speed):
        if self.max_speed<speed:
            self.max_speed=speed
    def add_acceleration(self, acceleration):
        if self.max_acceleration<acceleration:
            self.max_acceleration=acceleration
        if (self.min_acceleration is None or self.min_acceleration>acceleration):
            self.min_acceleration=acceleration
    def calculate_distance(self):
        # Calculate the Euclidean distance
        distance = math.sqrt((self.end_x - self.start_x) ** 2 + (self.end_y - self.start_y) ** 2)
        return distance
