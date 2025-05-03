import os
import cv2
import time
import json
import numpy as np
import collections
from skimage import metrics
from skimage.metrics import structural_similarity as ssim

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
        self.backSub = cv2.createBackgroundSubtractorKNN(history=0)
        self.ref_imgs = []
        self.ref_goalpost_imgs = []
        self.ref_bg = None

        self.crop_frame = None
        self.video_out_frame = None
        self.frame_id = 0
        self.goal_stop = 0
        self.score_cntr=0
        self.kernel = np.ones((5,5),np.float32)/25
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        if (self.settings.debug_mode):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.load_ref_ball_images(self.settings.ref_img_folder)
        self.load_ref_goalpost_images(self.settings.ref_frame_folder)
        pass

    
    def is_point_in_goals(self, point):
        goal_in = np.array(self.settings.goal_in, dtype=np.int32).reshape((-1, 1, 2))
        in_in = cv2.pointPolygonTest(goal_in, point, False) >= 0
        return in_in 

    def load_ref_ball_images(self, folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    self.ref_imgs.append(img)
                else:
                    print(f"Failed to load image: {filename}")

    def load_ref_goalpost_images(self, folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    self.ref_goalpost_imgs.append(img)
                else:
                    print(f"Failed to load image: {filename}")

    def crop(self, frame):
        return frame
    
    def compare_images4(self, ref_img, image2, id):
        image2 = cv2.resize(image2, (ref_img.shape[1], ref_img.shape[0]), interpolation = cv2.INTER_AREA)
        image1_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)

        line_width = 1
        height = ref_img.shape[0]
        red_line = np.full((height, line_width, 3), [0, 0, 255], dtype=np.uint8)
        diffImage =cv2.absdiff(ref_img, image2)
        _, diffImage = cv2.threshold(diffImage, self.settings.compare_images_absdiff_lower_threshold, 255, cv2.THRESH_BINARY)

        non_black_pixels_count = np.sum(diffImage == 255)
        full_size = ref_img.shape[1]* ref_img.shape[0]
        return round(ssim_score[0],2), non_black_pixels_count, full_size

    def average_images(self, image1, image2):
        if image1.shape != image2.shape:
            raise ValueError("Images must be of the same size to compute average")
        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)
        average_image = (image1 + image2) / 2.0
        average_image = np.uint8(average_image)
        return average_image

    def get_best_goalpost_reference(self, cropped_img):
        curr = cv2.filter2D(cropped_img,-1,self.kernel)
        best=None
        best_non_black=-1
        for frame in self.ref_goalpost_imgs:
            ref = cv2.filter2D(frame,-1,self.kernel)
            ssim_surr, non_black, full_size =self.compare_images4(ref, curr, 1)
            if (non_black<best_non_black or best_non_black==-1):
                best_non_black = non_black
                best = frame
        return best

    def report_distance(self, distance_to_goal_line):
        goal_scored = False
        if self.goal_stop==0 and distance_to_goal_line<0.1:
            goal_scored = True
            if (self.settings.debug_mode):
                print("GOOOOL - Non black: - (AI", )
            self.score_cntr+=1
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
        print(f"{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<25} {headers[4]:<25}")
        num_items_to_print = holdback_frame_cntr
        items_to_print = list(ball_pos_history)[-num_items_to_print:]
        for item in reversed(items_to_print):
            item = tuple("" if v is None else v for v in item)
            print(f"{item[0]:<15} {item[1]:<15} {item[2]:<15} {item[3]:<25} {item[4]:<25}")
        if len(items_to_print) < 2:
            return 'X'

        first_x, first_y = items_to_print[0][0], items_to_print[0][1]
        last_x, last_y = items_to_print[-1][0], items_to_print[-1][1]

        delta_x = abs(first_x - last_x)
        delta_y = abs(first_y - last_y)

        if self.side == 'L':
            center_x, center_y = 2, 125
        else:
            center_x, center_y = 498, 125
        is_confirmed=True
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
        height, width = curr.shape[:2]
        self.contour_img = np.zeros((height, width, 3), dtype=curr.dtype)
        if (self.settings.debug_mode):
            self.video_out_frame = np.zeros((height, width, 3), dtype=curr.dtype)
            self.video_out_frame[0:height, 0:width]=cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
        contours, hierarchy = cv2.findContours(self.fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.crop_frame = cropped_img  #.copy()        
        self.frame_id+=1

        if self.goal_stop>0:
            self.goal_stop-=1

        for idx, contour in enumerate(contours):
            contour = cv2.approxPolyDP(contour, self.settings.contour_smoothing_epsilon, True)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            if perimeter == 0:
                continue 
            if h>w*self.settings.max_height_width_ratio:
                continue
            
            #if w>50 or h>50:
            if w>self.settings.max_ball_width or h>self.settings.max_ball_height:
                continue

            #if area<10 or area>800:
            if area<self.settings.min_ball_area or area>self.settings.max_ball_area:
                continue

            circularity = round((4 * np.pi * area) / (perimeter ** 2),2)
            if circularity<self.settings.ball_circularity_threshold:
                continue

            avrg_color = self.calculate_average_color(curr,contour)
  
            obj_red = avrg_color[2]
            obj_green = avrg_color[1]
            obj_blue = avrg_color[0]
            
            #must be light color
            if obj_red<self.settings.ball_color_threshold or obj_blue<self.settings.ball_color_threshold or obj_green<self.settings.ball_color_threshold:
                continue
            avrg_ball_color = (obj_red+obj_green+obj_blue)/3
            if abs(obj_red-avrg_ball_color)>self.settings.max_difference_between_color_channels or abs(obj_green-avrg_ball_color)>self.settings.max_difference_between_color_channels or abs(obj_blue-avrg_ball_color)>self.settings.max_difference_between_color_channels:
                continue
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
                                   
                                self.score_cntr+=1
                            else:
                                print("SKIP SCORE_ NO GOAL!")
                        else:
                            if (self.settings.debug_mode):
                                pass
                    if (self.settings.debug_mode):
                        cv2.drawContours(self.crop_frame, [contour], -1,color, -1)
                    found = True
                    break
            if found == False:
                continue

            if (self.settings.debug_mode):
                cv2.rectangle(self.contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding boxes
                obj_id=idx+1
                print(obj_id, "cntr len:",len(contour), "area:", area, "circ:",circularity, "w:",w, "h:",h, center_x, center_y, "color:", avrg_color, "elliptic:", 0, "dir:", "")

                # Draw the index number at the center of the bounding box
                cv2.putText(self.contour_img, str(obj_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if (self.settings.debug_mode):
            cv2.putText(self.contour_img, str(self.frame_id), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            goal2 = np.array(self.settings.goal_in, dtype=np.int32)
            goal2 = goal2.reshape((-1, 1, 2))
            overlay = self.crop_frame.copy()
            cv2.fillPoly(overlay, [goal2], (100, 255, 255))
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, self.crop_frame, 1 - alpha, 0, self.crop_frame)
            cv2.putText(self.crop_frame, str(self.score_cntr), (12,32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(self.crop_frame, str(self.score_cntr), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
