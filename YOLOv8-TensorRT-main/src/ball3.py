import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage import metrics
import os
import json
import collections


class Settings:
    def __init__(self, config_file, section_id) -> None:
        with open(config_file, 'r') as file:
            self.config_data = json.load(file)
        self.blur=0

        if section_id not in self.config_data:
            raise ValueError(f"No configuration found for ID '{section_id}'")
        data = self.config_data[section_id]
        self.blur = data.get('blur', 7)
        self.ref_img_folder = data.get('ref_img_folder', '')
        self.ref_frame_folder = data.get('ref_frame_folder', '')
        self.max_ball_bouncback_speed=data.get('max_ball_bouncback_speed', 14)

        self.max_ball_negative_acceleration=data.get('max_ball_negative_acceleration', -30)

        
        #self.ref_frame_file = data.get('ref_frame', '')
        
        self.goal_in = data['goal_in']
        self.frame_crop = data['frame_crop']
        self.compare_images_absdiff_lower_threshold = data.get('compare_images_absdiff_lower_threshold', 100)
        self.contour_smoothing_epsilon = data.get('contour_smoothing_epsilon', 1)
        self.ball_circularity_threshold = data.get('ball_circularity_threshold', 0.85)
        self.ball_color_threshold = data.get('ball_color_threshold', 120)
        self.max_difference_between_color_channels=data.get('max_difference_between_color_channels', 120)
        self.max_height_width_ratio = data.get('max_height_width_ratio', 1.2)
        self.max_ball_width = data.get('max_ball_width', 18)
        self.max_ball_height = data.get('max_ball_height', 18)

        self.min_ball_area = data.get('min_ball_area', 90)
        self.max_ball_area = data.get('max_ball_area', 200)

        self.ball_ssim_threshold = data.get('ball_ssim_threshold', 0.8)
        self.ball_surrounding_ssim_threshold = data.get('ball_surrounding_ssim_threshold', 0.8)
        self.surrounding_difference_threshold = data.get('surrounding_difference_threshold', 20)
        self.surrounding_pixels_x = data.get('surrounding_pixels_x', 15)
        self.surrounding_pixels_y = data.get('surrounding_pixels_y', 15)

        self.skip_goal_check_after_scoring_for_frames = data.get('skip_goal_check_after_scoring_for_frames', 60)
        self.debug_mode = data.get('debug_mode', True)
        


class GoalChecker:
    def __init__(self, section_id, config_file, side) -> None:
        #self.balltracker= BallTracker()
        self.settings = Settings(config_file, section_id)
        print(self.settings.blur)
        self.side = side        
        self.reference_frame = None
        self.blurred_edges = None
        self.edges = None
        self.cntr = 0
        self.section_id = section_id
        
        self.fg_mask =None
        self.foreground = None
        self.backSub = cv2.createBackgroundSubtractorKNN(history=0)# 25*60 , dist2Threshold=180)
        #self.backSub_bg =cv2.createBackgroundSubtractorMOG2(history=5000)
        self.ref_imgs = []
        self.ref_goalpost_imgs = []
        self.ref_bg = None

        #self.goal_in =[(157,213),(120,86), (230,0), (263,111) ]
        self.crop_frame = None
        self.video_out_frame = None
        self.frame_id = 0
        self.goal_stop = 0
        self.score_cntr=0
        self.kernel = np.ones((5,5),np.float32)/25
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))


        if (self.settings.debug_mode):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #self.out = cv2.VideoWriter('balltracker.avi', fourcc, 25, (1500, 220))        

        self.load_ref_ball_images(self.settings.ref_img_folder)
        self.load_ref_goalpost_images(self.settings.ref_frame_folder)
        pass

    
    def is_point_in_goals(self, point):
        # Convert list of points to a numpy array of shape (-1, 1, 2)
        goal_in = np.array(self.settings.goal_in, dtype=np.int32).reshape((-1, 1, 2))
        # Check if point is in goal_in
        in_in = cv2.pointPolygonTest(goal_in, point, False) >= 0
        return in_in 

    def load_ref_ball_images(self, folder):
        for filename in os.listdir(folder):
            # Check if the file is an image based on common image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    self.ref_imgs.append(img)
                else:
                    print(f"Failed to load image: {filename}")

    def load_ref_goalpost_images(self, folder):
        for filename in os.listdir(folder):
            # Check if the file is an image based on common image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    self.ref_goalpost_imgs.append(img)
                else:
                    print(f"Failed to load image: {filename}")

    def crop(self, frame):
        return frame
        #return frame[642:642+220,2052:2052+300]
        
        y_start, y_end = self.settings.frame_crop[0]
        x_start, x_end = self.settings.frame_crop[1]
        return frame[y_start:y_end, x_start:x_end]
        #return frame[self.settings.frame_crop[0],self.settings.frame_crop[1]]
        #return frame[580:800,300:600]
    
    def compare_images4(self, ref_img, image2, id):
        image2 = cv2.resize(image2, (ref_img.shape[1], ref_img.shape[0]), interpolation = cv2.INTER_AREA)
        #print(image1.shape, image2.shape)
        # Convert images to grayscale
        image1_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Calculate SSIM
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)

        line_width = 1
        height = ref_img.shape[0]
        red_line = np.full((height, line_width, 3), [0, 0, 255], dtype=np.uint8)
        #combined_image = np.concatenate((ref_img, red_line, image2), axis=1)
        #cv2.imwrite(f'img_sv/o{self.frame_id}_{id}.png', combined_image)
        #cv2.imwrite(f'img_sv/r1_{self.frame_id}_{id}.png', image2)
        #if id==0:
        #    cv2.imwrite(f'img_sv/{self.section_id}_{self.frame_id}_{id}.png', image2)
            
        
        diffImage =cv2.absdiff(ref_img, image2)
        _, diffImage = cv2.threshold(diffImage, self.settings.compare_images_absdiff_lower_threshold, 255, cv2.THRESH_BINARY)
        #cv2.imwrite(f'img_sv/d1_{self.frame_id}.png', diffImage)

        non_black_pixels_count = np.sum(diffImage == 255)
        full_size = ref_img.shape[1]* ref_img.shape[0]
        return round(ssim_score[0],2), non_black_pixels_count, full_size

    def average_images(self, image1, image2):
        # Ensure both images are the same size
        if image1.shape != image2.shape:
            raise ValueError("Images must be of the same size to compute average")

        # Convert images to float32 to prevent data type overflow/underflow
        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)
        
        # Calculate the average
        average_image = (image1 + image2) / 2.0
        
        # Convert back to uint8
        average_image = np.uint8(average_image)

        return average_image

    def get_best_goalpost_reference(self, cropped_img):
        print("***********************")
        print("***********************")
        curr = cv2.filter2D(cropped_img,-1,self.kernel)
        best=None
        best_non_black=-1
        for frame in self.ref_goalpost_imgs:
            ref = cv2.filter2D(frame,-1,self.kernel)
            ssim_surr, non_black, full_size =self.compare_images4(ref, curr, 1)
            print(ssim_surr, non_black, full_size)
            if (non_black<best_non_black or best_non_black==-1):
                best_non_black = non_black
                best = frame
        print("***********************")
        print("***********************")

        #if best_non_black>0:
        #    best=None
        return best


    def report_distance(self, distance_to_goal_line):
        goal_scored = False
        if self.goal_stop==0 and distance_to_goal_line<0.1:
            goal_scored = True
            if (self.settings.debug_mode):
                print("GOOOOL - Non black: - (AI", )
                #cv2.imwrite(f'goal_{self.cntr}.png', self.crop_frame)
                #cv2.imwrite(f'ball_{int(video_pos)}_{self.cntr}.png', region)
            self.score_cntr+=1
            #self.goal_stop=self.settings.skip_goal_check_after_scoring_for_frames
        
        return goal_scored
    def init_goal_stop(self):
        self.goal_stop=self.settings.skip_goal_check_after_scoring_for_frames

    def check_ball(self, center_x, item):
        is_confirmed=True
        last_x, last_y = item[0], item[1]
        if abs(center_x-last_x)<6:
            if self.side == 'L' and  (last_y<98 or last_y>142):
                print("Ball almost last is outside!!!!!")
                is_confirmed=False
            if self.side == 'R' and  (last_y<98 or last_y>142):
                print("Ball almost last outside!!!!!")
                is_confirmed=False
        return is_confirmed

    def do_last_look(self, ball_pos_history, holdback_frame_cntr):
        headers = ["last_ball_x", "last_ball_y", "last_ball_ttl", "distance_to_L_goal_line", "distance_to_R_goal_line"]

        # Print the headers
        print(f"{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<25} {headers[4]:<25}")

        # Calculate the number of items to print
        num_items_to_print = holdback_frame_cntr

        # Get the items to print from the back of the deque
        items_to_print = list(ball_pos_history)[-num_items_to_print:]

        # Print the items in reverse order
        for item in reversed(items_to_print):
            # Replace None values with an empty string or any other default value
            item = tuple("" if v is None else v for v in item)
            print(f"{item[0]:<15} {item[1]:<15} {item[2]:<15} {item[3]:<25} {item[4]:<25}")

        # Determine direction if there are at least two points
        if len(items_to_print) < 2:
            return 'X'

        first_x, first_y = items_to_print[0][0], items_to_print[0][1]
        last_x, last_y = items_to_print[-1][0], items_to_print[-1][1]

        delta_x = abs(first_x - last_x)
        delta_y = abs(first_y - last_y)

        #Left goal line y -> 98 (top) - 142 (bottom)
        if self.side == 'L':
            center_x, center_y = 2, 125
        else:
            center_x, center_y = 498, 125

        is_confirmed=True

        #ball travels up and down
        if (delta_x<8):
            if self.side == 'L' and  (last_y<98 or last_y>142):
                print("Ball last pos is outside!!!!!")
                is_confirmed=False
            if self.side == 'R' and  (last_y<98 or last_y>142):
                print("Ball last pos is outside!!!!!")
                is_confirmed=False

        for i in range(2, 8):
            if is_confirmed:
                is_confirmed = self.check_ball(center_x, items_to_print[-i])
                return is_confirmed
     

    def check(self, frame, cycle, video_pos=0):
        self.cntr+=1
        goal_scored = False
        #cv2.imwrite("frm.png", frame)

        cropped_img = self.crop(frame)

        if self.reference_frame is None:
            self.reference_frame = self.get_best_goalpost_reference(cropped_img)
            '''# Check if the file exists
            if os.path.exists(self.settings.ref_frame_file):
                print("File exists.")
                self.reference_frame = cv2.imread(self.settings.ref_frame_file)
            else:
                print("File does not exist. Creating")
                self.reference_frame = cropped_img
                cv2.imwrite(self.settings.ref_frame_file, self.reference_frame)

            self.reference_frame = cv2.GaussianBlur(self.reference_frame, (self.settings.blur,self.settings.blur), 100)'''
        
        #if self.cntr<2:
        #    self.reference_frame = self.average_images(self.reference_frame, cv2.GaussianBlur(cropped_img, (self.settings.blur,self.settings.blur), 100))

        '''ORIGINAL
        curr = cv2.filter2D(cropped_img,-1,self.kernel)

        if self.ref_bg is None:
            self.ref_bg = curr.copy()

        #if (self.backSub_bg is not None):
        #    self.backSub_bg.apply(curr,learningRate=-1)

        self.fg_mask = self.backSub.apply(curr, learningRate=0)#,learningRate=-1)        
        _, self.fg_mask = cv2.threshold(self.fg_mask, 254, 255, cv2.THRESH_BINARY)
        
        self.fg_mask = cv2.morphologyEx(self.fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        #self.fg_mask = cv2.morphologyEx(self.fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        '''
        curr = cv2.filter2D(cropped_img,-1,self.kernel)

        if self.ref_bg is None:
            self.ref_bg = curr.copy()


        self.fg_mask = self.backSub.apply(curr, learningRate=0)#,learningRate=-1) 
        
        # Convert the image to HSV
        hsv_img = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for white-ish colors in HSV
        lower_bound = np.array([0, 0, 140], dtype=np.uint8)
        upper_bound = np.array([180, 50, 255], dtype=np.uint8)

        # Create a mask for white-ish colors
        white_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        # Ensure fg_mask is single-channel and of type uint8
        if len(self.fg_mask.shape) == 3:
            self.fg_mask = cv2.cvtColor(self.fg_mask, cv2.COLOR_BGR2GRAY)
        self.fg_mask = self.fg_mask.astype(np.uint8)

        # Combine the white mask with the foreground mask
        self.fg_mask = cv2.bitwise_and(self.fg_mask, self.fg_mask, mask=white_mask)

        _, self.fg_mask = cv2.threshold(self.fg_mask, 254, 255, cv2.THRESH_BINARY)

        self.fg_mask = cv2.morphologyEx(self.fg_mask, cv2.MORPH_OPEN, self.morph_kernel)      

        height, width = curr.shape[:2]  # Get dimensions from blurred_edges
        # Create a blank image (black) with the same size and same type as blurred_edges
        self.contour_img = np.zeros((height, width, 3), dtype=curr.dtype)
        if (self.settings.debug_mode):
            self.video_out_frame = np.zeros((height, width, 3), dtype=curr.dtype)
            self.video_out_frame[0:height, 0:width]=cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
        
        contours, hierarchy = cv2.findContours(self.fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.crop_frame = cropped_img#.copy()        
        
        self.frame_id+=1

        #print("-----", self.frame_id)
        
        if self.goal_stop>0:
            self.goal_stop-=1

        for idx, contour in enumerate(contours):
            contour = cv2.approxPolyDP(contour, self.settings.contour_smoothing_epsilon, True)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            x, y, w, h = cv2.boundingRect(contour)

            if perimeter == 0:
                continue  # Skip this contour as it can't be circular (also avoids division by zero)

              #it can not be elliptic that way that it's height is much longer then the width
            if h>w*self.settings.max_height_width_ratio:
                continue
            
            #if w>50 or h>50:
            if w>self.settings.max_ball_width or h>self.settings.max_ball_height:
                continue

            #if area<10 or area>800:
            if area<self.settings.min_ball_area or area>self.settings.max_ball_area:
                continue

            circularity = round((4 * np.pi * area) / (perimeter ** 2),2)

            #must be either circular or elliptic
            if circularity<self.settings.ball_circularity_threshold:
                continue

            # Calculate the bounding rectangle for each contour
            avrg_color = self.calculate_average_color(curr,contour)
  
            obj_red = avrg_color[2]
            obj_green = avrg_color[1]
            obj_blue = avrg_color[0]
            
            #must be light color
            if obj_red<self.settings.ball_color_threshold or obj_blue<self.settings.ball_color_threshold or obj_green<self.settings.ball_color_threshold:
                continue

            #must be close to white/gray as much as possible
            avrg_ball_color = (obj_red+obj_green+obj_blue)/3
            if abs(obj_red-avrg_ball_color)>self.settings.max_difference_between_color_channels or abs(obj_green-avrg_ball_color)>self.settings.max_difference_between_color_channels or abs(obj_blue-avrg_ball_color)>self.settings.max_difference_between_color_channels:
                continue
  
            # Calculate the center position of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2


            found=False
            region = curr[y:y+h, x:x+w]
            for ref_img in self.ref_imgs:
                ssim, _ , _ = self.compare_images4(ref_img, region,0)
                if (ssim>self.settings.ball_ssim_threshold and x>self.settings.surrounding_pixels_x and y>self.settings.surrounding_pixels_y ):
                    edge_x=self.settings.surrounding_pixels_x
                    edge_y=self.settings.surrounding_pixels_y

                    ref = self.ref_bg[y-edge_y:y+h+edge_y, x-edge_x:x+w+edge_x].copy()
                    current  = curr[y-edge_y:y+h+edge_y, x-edge_x:x+w+edge_x].copy()
                    #erase the ball so we can compare the background                    
                    cv2.rectangle(ref,(edge_x-1,edge_y-1),(edge_x+w+2, edge_y+h+2), (0,0,0), -1)
                    cv2.rectangle(current,(edge_x-1,edge_y-1),(edge_x+w+2, edge_y+h+2), (0,0,0), -1)
                    
                    ssim_surr, non_black, full_size =self.compare_images4(ref, current, 1)
                    calculated_surr_sim = 1- non_black/full_size
                    #cv2.imwrite(f'sim_ref_{cycle}_{idx}.png', ref)
                    #cv2.imwrite(f'sim_curr_{cycle}_{idx}.png', current)
                    
                    #if (self.settings.debug_mode):
                    print(self.frame_id, f'[{cycle}]', "SIMILARITY:", ssim_surr, self.goal_stop, "circularity:", circularity, "INITIAL SIMILARITY:", ssim, "None black:", 
                              non_black, full_size, calculated_surr_sim, f'Center: {center_x}, {center_y}')
                    color = (0, 0, 255)
                    if calculated_surr_sim>self.settings.ball_surrounding_ssim_threshold:
                        if non_black<self.settings.surrounding_difference_threshold:
                            color = (0,255,0)
                            is_within = self.is_point_in_goals((center_x, center_y))==True
                            print("is_within:", is_within)
                            print("GOAL STOP:", self.goal_stop)
                            if is_within and self.goal_stop==0:
                                goal_scored = True
                                if (self.settings.debug_mode):
                                    print("GOOOOL - TP - Non black:", non_black)
                                    #cv2.imwrite(f'goal_{int(video_pos)}_{self.cntr}.png', self.crop_frame)
                                    #cv2.imwrite(f'ball_{int(video_pos)}_{self.cntr}.png', region)
                                self.score_cntr+=1
                                #self.goal_stop=self.settings.skip_goal_check_after_scoring_for_frames
                            else:
                                print("SKIP SCORE_ NO GOAL!")
                        else:
                            if (self.settings.debug_mode):
                                pass
                                #cv2.imwrite(f'NONgoal_{int(video_pos)}_{self.cntr}.png', self.crop_frame)
                    if (self.settings.debug_mode):
                        cv2.drawContours(self.crop_frame, [contour], -1,color, -1)
                    found = True
                    break
            if found == False:
                continue


            if (self.settings.debug_mode):
                cv2.rectangle(self.contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding boxes
                
                #obj_id = self.balltracker.register_object(self.frame_id, contour, area, circularity, 0, center_x, center_y, w, h, "", avrg_ball_color)
                obj_id=idx+1
                print(obj_id, "cntr len:",len(contour), "area:", area, "circ:",circularity, "w:",w, "h:",h, center_x, center_y, "color:", avrg_color, "elliptic:", 0, "dir:", "")

                # Draw the index number at the center of the bounding box
                cv2.putText(self.contour_img, str(obj_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        

        
        if (self.settings.debug_mode):
            cv2.putText(self.contour_img, str(self.frame_id), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            goal2 = np.array(self.settings.goal_in, dtype=np.int32)
            goal2 = goal2.reshape((-1, 1, 2))

            # Step 1: Create a copy of the image to draw the polygon
            overlay = self.crop_frame.copy()

            # Step 2: Draw the polygon on the copy
            cv2.fillPoly(overlay, [goal2], (100, 255, 255))

            # Step 3: Blend the original image with the overlay using the desired alpha (transparency level)
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, self.crop_frame, 1 - alpha, 0, self.crop_frame)
            
            #video_out_frame[0:height, 0:width]=self.contour_img
            #video_out_frame[0:height, width*2:width*3]=self.crop_frame
            #video_out_frame[0:height, width*3:width*4]=self.reference_frame
            #self.out.write(video_out_frame)

            cv2.putText(self.crop_frame, str(self.score_cntr), (12,32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(self.crop_frame, str(self.score_cntr), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        #if self.backSub_bg is not None:
        #    self.reference_frame = self.backSub_bg.getBackgroundImage()

            
        return goal_scored, self.crop_frame, height, width, self.video_out_frame

    def calculate_average_color(self, img, contour):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        indices = np.where(mask != 0)
        pixels = img[indices[0], indices[1], :]
        
        average_color = np.mean(pixels, axis=0)

        return average_color
    
    def save_ball(self, frame, x1, y1, x2, y2):
            region = frame[y1:y2, x1:x2]
            cv2.imwrite(f'frma_{self.cntr}.png', region)

'''def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    gc = GoalChecker("Field_2_Goal_L", "goal.json")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    pos = 11#1820#710#3005
    cap.set(cv2.CAP_PROP_POS_MSEC, pos*1000)  # 120000 milliseconds = 2 minutes

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read the video file or end of file")
            break


        start =time.perf_counter()*1000
        ppos = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        goal_scored = gc.check(frame, ppos)
        if goal_scored:
            print("GOOOOL")
        now =time.perf_counter()*1000
        time_diff_ms = now - start
        #print("Process time:", time_diff_ms)'''

