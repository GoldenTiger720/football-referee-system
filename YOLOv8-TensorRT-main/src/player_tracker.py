import copy
import math
import numpy as np
import cv2
from skimage import metrics


class FootPlayer:
    def __init__(self):
        self.cycle=-1
        self.position=-1
        self.x1 =-1
        self.x2 =-1
        self.y1=-1
        self.y2=-1
        self.img = None

class PlayerRef:
    def __init__(self):
        self.id=-1
        self.width =-1
        self.height =-1
        self.x=-1
        self.y=-1
        self.img = None
        self.last_used_idx = None

class PlayerStore:
    def __init__(self):
        self.players = {}
    
    def register_player(self, player, id, img):
        new_player = PlayerRef()
        new_player.id = id
        new_player.width = player.x2-player.x1
        new_player.height = player.y2-player.y1
        new_player.x = int(player.x1+ new_player.width/2)
        new_player.y = int(player.y1+ new_player.height/2)
        new_player.img = img
        if id not in self.players:
            self.players[id] = None

        self.players[id] = new_player

    def lookup_player_id(self, player, dict_key):
        new_player = PlayerRef()
        new_player.id = id
        w = player.x2-player.x1
        h = player.y2-player.y1
        x = int(player.x1+ new_player.width/2)
        y = int(player.y1+ new_player.height/2)
       
        min_dist=None
        min_dist_player = None
        for plr in self.players:
            dist = max(abs(plr.x-x),abs(plr.y-y))
            if min_dist is None or min_dist>dist and plr.last_used_idx!=dict_key:
                min_dist=dist
                min_dist_player = plr
        
        if min_dist is not None and min_dist<40:
            min_dist_player.last_used_idx!=dict_key
            return min_dist_player.id
        
        return None

class PlayerTracker:
    def __init__(self):
        # Initialize a dictionary to store players
        self.players = {}
        self.allocated_ids = {}
        self.max_alloc_id = 0
        self.player_ref = PlayerStore()
    
    def get_max_id(self,key):
        players = self.players[key]
        if players is None:
            return 1
        max_id=0
        for player in players:
            if player.player_id>max_id:
                max_id=player.player_id
        return max_id
    def register(self, img, player, cycle, position):
        return 0

        if cycle<10:
            return 0
        #hist_img1 = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        #hist_img1[255, 255, 255] = 0 #ignore all white pixels
        #cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        #image1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Calculate SSIM
        #ssim_score = metrics.structural_similarity(image1_gray, image1_gray, full=True)


        print("register", cycle, position)
        player.centerx=(player.x1+player.x2)/2
        player.centery = player.y1
        image1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        player.img = image1_gray.copy()
        # Create a key for the dictionary based on cycle and position
        dict_key = cycle*25+ position
        # If the key is not in the dictionary, add it with an empty list
        if dict_key not in self.players:
            self.players[dict_key] = []
        if dict_key not in self.allocated_ids:
            self.allocated_ids[dict_key] = []

        #prev_player = self.get_previous_instance(dict_key, image1_gray, player, cycle, position)
        id = self.player_ref.lookup_player_id(player, dict_key)
        if id is not None:
            print("using prev id", id)
            player.player_id = id
        else:
            id = self.max_alloc_id +1
            self.max_alloc_id+=1
            print("Allocating new ID of", id)
            player.player_id = id
            self.player_ref.register_player(player, id, img)
        # Append the player to the list for the given key
        self.players[dict_key].append(copy.deepcopy(player))
        self.allocated_ids[dict_key].append(player.player_id)

        return player.player_id
    
    def get_players(self, cycle, position):
        print("get_players", cycle, position)
        # Create a key for the dictionary based on cycle and position
        key = cycle*25+ position
        
        # Return the list of players for the given key, or an empty list if the key is not found
        return self.players.get(key, [])
    
    '''def get_previous_instance(self, dict_key, img, player, cycle, position):
        print("get_previous_instance ", cycle, position)
        pos=position
        cyc = cycle
        for i in range(30):
            print("turn", i)
            min_distance = None
            min_distance_player = None            
            pos-=1
            if pos<0:
                pos=24
                cyc-=1
            if cyc>0:
                players = self.get_players(cyc, pos)
                if players is not None:
                    for plr in players:
                        if plr.last_used_idx!=dict_key:
                            dist = math.sqrt((plr.centerx - player.centerx) ** 2 + (plr.centery - player.centery) ** 2)
                            print("min_distance", dist, plr.player_id, plr.last_used_idx)
                            if (min_distance==None or min_distance>dist and self.compare(plr, player)==True and 
                                (self.allocated_ids[dict_key] is None or plr.player_id not in self.allocated_ids[dict_key])):
                                min_distance = dist
                                min_distance_player = plr

                    if min_distance is not None and min_distance<60:
                        ssim_score=-1
                        if img is not None and min_distance_player.img is not None:
                            try:
                                orig = min_distance_player.img.copy()
                                image2 = cv2.resize(img, (orig.shape[1], orig.shape[0]), interpolation = cv2.INTER_AREA)
                                print("Ssize", min_distance_player.img.shape[1], min_distance_player.img.shape[0], 
                                  image2.shape[1], image2.shape[0])
                                ssim_score = metrics.structural_similarity(image2, orig, full=True)
                            except:
                                pass
                        print("ssim_score", ssim_score[0])
                        if ssim_score[0]>0.4 or ssim_score==-1:
                            return min_distance_player
        print("No prev. instance...")
        return None
    
    def compare(self, pl1, pl2):
        height_delta= abs(abs(pl1.y2-pl1.y1) - abs(pl2.y2-pl2.y1))
        if height_delta>15:
            return False
        return True'''