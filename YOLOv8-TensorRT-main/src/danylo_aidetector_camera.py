import time
import math
import logging
from src.danylo_gpu import *
from src.detector_utils import *
from src.camera import *
from src.aisettings import *
from src.ballradiustracker import *

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

#50-77%
CLASSID_BALL=0#32
CLASSID_FASTBALL=1
CLASSID_PLAYER=2#0



#CLASSID_BALL=32
#CLASSID_FASTBALL=32
#CLASSID_PLAYER=0

ballRadiusChecker = BallRadiusTracker()


class AIDetector:
    def __init__(self, ai_settings, cameras, paralell_models, max_fps) -> None:

        logging.basicConfig(level=logging.DEBUG, filename='detector.log', filemode='w',
                    format='%(message)s')        
        self.cameras = cameras
        self.ai_settings = ai_settings
        self.max_fps = max_fps
        self.paralell_models = paralell_models
        self.engines_960=[]
        self.engines_1280=[]
        self.goal_engines=[]
        self.futures=[]
        self.completed_futures = set()
        self.W_960 = 0
        self.H_960 = 0
        self.W_1280 = 0
        self.H_1280 = 0

        self.ball_color_total = defaultdict(int)
        self.ball_color_counter = defaultdict(int)

        self.goal_W = 0
        self.goal_H = 0
        self.goal_l_reference_taken=False
        self.goal_r_reference_taken=False
        self.false_detection = 0        
        self.best_camera = -1
        self.last_best_camera = -1
        self.last_best_camera_obj = None
        self.last_proc_time = 0
        self.second_best_camera = -1
        self.detector_cycle = 0
        self.current_stage = 1
        self.executor = ThreadPoolExecutor(max_workers = self.paralell_models)
        self.outstanding_detections = 0
        self.last_outstanding_detections = 0
        self.device = create_device(0)

        self.stage_start_time = 0
        self.last_left_goalkeeper=None
        self.broadcast_camera = -1

        for m_id in range(self.paralell_models ):
            print("CREATE MODEL: ", m_id)
            self.W_960, self.H_960 = create_engine(0, self.engines_960, "960")
            self.W_1280, self.H_1280 = create_engine(0, self.engines_1280, "1280")
            
            self.goal_W, self.goal_H = create_engine(0, self.goal_engines, "", create_goal_engine=True)

    def start_cycle(self, detector_cycle):
        self.detector_cycle = detector_cycle

    def get_number_of_outstanding_detections(self):
        self.outstanding_detections = 0
        for camera in self.cameras:
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None and obj.skip == False and obj.processed ==False:
                    self.outstanding_detections += 1
        return self.outstanding_detections
    
    def process(self, current_detector_cycle):
        if (current_detector_cycle<=self.detector_cycle and self.current_stage==1):
            #print(f'Returning ********* Current live{current_detector_cycle}  next: {self.detector_cycle}')
            return
        
        #print(self.current_stage,self.detector_cycle )
        #time.sleep(0.05)
        self.process_results_non_blocking()
        if self.futures:
            return

        self.get_number_of_outstanding_detections()
        if self.outstanding_detections>0:
            #print("Outstanding detections.... Returning.... ")
            return

        if self.current_stage==1:
            self.stage_start_time = time.monotonic_ns()
            #self.current_stage = 99
            print("---> STAGE 1")


            self.run_stage_1(number_of_frames=self.ai_settings.detection_first_stage_frames)
            self.current_stage = 2
            return
        
        if self.current_stage==2:
            print("---> STAGE 2")
            self.make_camera_selection()
            return
        
        if self.current_stage==3:
            print("---> STAGE 3")
            
            if (self.best_camera!=-1):
                processed_cntr = 0
                ball_on=0
                for cam in self.cameras:
                    if (cam.capture_only_device == True):
                        continue                    
                    if cam.camera_id == self.best_camera:
                        detector_objs = cam.get_detector_collection(self.detector_cycle)
                        for obj in detector_objs:
                            if obj is not None:
                                if obj.ball > 0:
                                    ball_on += 1
                                if obj.processed == True:
                                    processed_cntr+=1

                if ball_on<4 and self.second_best_camera!=-1:
                    print("DO attional detection on second best <<<")
                    self.do_detection_selected_camera(self.second_best_camera, 6)
                    self.run_detection()
                    self.current_stage = 6
                    return

                '''else:'''
                print(f'**** FINAL -> best camera: {self.best_camera}. Tota ball frame: {ball_on} / {processed_cntr}')
                self.current_stage = 99
            else:
                print("NO Best camera found......\n\n")
                self.current_stage = 99

            return

        if self.current_stage==4:
            print("---> STAGE 4")
            self.current_stage = 99
            #self.make_camera_selection(give_up_if_no_best_found=True)
            return


        if self.current_stage==6:
            print("---> STAGE 6")
            self.make_camera_selection(just_do_selection_nothing_else=True)
            self.current_stage = 99
            return        

        if (self.current_stage==99):
            self.current_stage = 110
            return

        if (self.current_stage == 110):
            #self.run_goal_detection()
            #self.run_goal_detection2()
    
            self.current_stage = 111
            return

        if (self.current_stage == 111):
            self.run_stage_final()


            '''            if self.goal_l_reference_taken==False:
                for camera in self.cameras:
                    if camera.camera_id==11:
                        frames=[]
                        people_cntr=0
                        detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                        for det_obj in detector_objs:
                            if (det_obj is not None and det_obj.skip == False):
                                if det_obj.processed ==True:
                                    print(">>>>Found processed LEFT Goal frame")
                                    if det_obj.people==0:
                                        frames.append(det_obj.frame)
                                    people_cntr+=det_obj.people
                        print(">>>> Total ppl cntr:", people_cntr)
                        if people_cntr==0 and len(frames)>=5:
                            cv2.imwrite("ref_l_goal.png", frames[3])
                            self.goal_l_reference_taken=True

            if self.goal_r_reference_taken==False:
                for camera in self.cameras:
                    if camera.camera_id==10:
                        frames=[]
                        people_cntr=0
                        detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                        for det_obj in detector_objs:
                            if (det_obj is not None and det_obj.skip == False):
                                if det_obj.processed ==True:
                                    print(">>>>Found processed RIGHT Goal frame")
                                    if det_obj.people==0:
                                        frames.append(det_obj.frame)
                                    people_cntr+=det_obj.people
                        print(">>>> Total ppl cntr:", people_cntr)
                        if people_cntr==0 and len(frames)>=5:
                            cv2.imwrite("ref_r_goal.png", frames[3])
                            self.goal_r_reference_taken=True'''



            #torch.cuda.empty_cache()
            self.detector_cycle+=1
            self.current_stage = 1
            #self.current_stage = 555


        
        
        
        #print("*")
        

        '''if self.last_outstanding_detections != self.outstanding_detections:
            self.last_outstanding_detections = self.outstanding_detections
            print("OUTSTANDING DETECTIONS:", self.outstanding_detections)
            if self.outstanding_detections==0:
                if self.current_stage == 2:
                    self.make_camera_selection()
                self.current_stage+=1
                print(f'Current Stage is {self.current_stage}')'''


    def get_best_camera(self):
        best_camera = -1
        second_best_camera = -1
        best_camera_avrg_conf = 0
        second_best_camera_avrg_conf = 0
        best_camera_ball_on = 0  # Track the number of valid detections for the best camera
        second_best_camera_ball_on = 0  # Track the number of valid detections for the second best camera
        total_ball_on = 0

        for camera in self.cameras:
            ball_on = 0
            confid = 0
            if (camera.capture_only_device == True):
                continue            
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for obj in detector_objs:
                if obj is not None and obj.ball > 0:
                    ball_on += 1
                    total_ball_on += 1
                    confid += obj.confidence
            if ball_on > 0:
                this_avrg_conf = confid / ball_on
                # Update based on ball_on first, then average confidence
                if ball_on > best_camera_ball_on or (ball_on == best_camera_ball_on and this_avrg_conf > best_camera_avrg_conf):
                    # Move current best to second best
                    second_best_camera = best_camera
                    second_best_camera_avrg_conf = best_camera_avrg_conf
                    second_best_camera_ball_on = best_camera_ball_on
                    # Update the best camera
                    best_camera = camera.camera_id
                    best_camera_avrg_conf = this_avrg_conf
                    best_camera_ball_on = ball_on
                elif ball_on > second_best_camera_ball_on or (ball_on == second_best_camera_ball_on and this_avrg_conf > second_best_camera_avrg_conf):
                    # Update the second best camera without changing the best
                    second_best_camera = camera.camera_id
                    second_best_camera_avrg_conf = this_avrg_conf
                    second_best_camera_ball_on = ball_on

        return best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on

    def make_camera_selection(self, just_do_selection_nothing_else=False):

        best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on = self.get_best_camera()

        print("Best camera ->", best_camera, "with", best_camera_ball_on, "valid detections - confidence:", round(best_camera_avrg_conf,2))
        print("Second best camera ->", second_best_camera, "with", second_best_camera_ball_on, "valid detections - confidence:", round(second_best_camera_avrg_conf,2))

        self.best_camera = best_camera
        self.second_best_camera = second_best_camera

        if (just_do_selection_nothing_else==True):
            self.current_stage=99
            return

        if (best_camera!=-1):
            print("RUNNING ADDITIONAL DETECTIONS on CAM",best_camera)
            
            self.do_detection_selected_camera(best_camera, self.ai_settings.detection_last_stage_on_best_frames)
            #if best_camera!=2:
            #    self.do_detection_selected_camera(2, 8)
            #if best_camera!=3:
            #    self.do_detection_selected_camera(3, 8)

            self.run_detection()
            
            self.current_stage = 3
            print("Going to next stage - 3")

        else:

            print("Let's do detection again on all cameras as the 1st resulted in -1")
            #detections=8
            self.do_detection_selected_camera(2, self.ai_settings.detection_last_stage_on_both_centers_frames)
            #imre
            #imre
            #self.do_detection_selected_camera(3, self.ai_settings.detection_last_stage_on_both_centers_frames)
            
            self.run_detection()
                
            outs =self.get_number_of_outstanding_detections()
            print("Outstanding", outs)
            self.current_stage=4

    def do_detection_selected_camera(self, camera_id, nbr_of_frames = 6):
            frames_marked = [-1,self.max_fps+1]  # we have an entire second as a starting reference
            available_frames_to_process=[]
            for camera in self.cameras:
                if camera.camera_id==camera_id:
                    detector_objs = camera.get_detector_collection(self.detector_cycle)
                    for det_obj in detector_objs:
                        if det_obj is not None:
                            if det_obj.skip==True and det_obj.in_progress == False:
                                available_frames_to_process.append(det_obj.position)
                            else:
                                frames_marked.append(det_obj.position)
            #print("DOne already:", frames_marked)
            #print("Available:", available_frames_to_process)
            if len(available_frames_to_process)>0:
                for i in range(nbr_of_frames):
                    frame_pos = insert_for_even_distribution(frames_marked, available_frames_to_process)
                    #print("pos", pos)
                    if (frame_pos is not None):
                        for det_obj in detector_objs:
                            if det_obj is not None and det_obj.position == frame_pos:
                                det_obj.skip=False                        
                                frames_marked.append(frame_pos)

    def run_stage_final(self):
        print("run_stage_final()")

        best_camera, best_camera_avrg_conf, best_camera_ball_on, second_best_camera, second_best_camera_avrg_conf, second_best_camera_ball_on = self.get_best_camera()
        self.best_camera = best_camera
        self.second_best_camera = second_best_camera

        processed_cntr = 0
        goal_processed_cntr = 0
        ball_on=0
        goal_ball_on=0
        frames = 0
        goal_frames = 0
        framepos = []
        for cam in self.cameras:
            if (cam.capture_only_device == True):
                continue            
            detector_objs = cam.get_detector_collection(self.detector_cycle)
            #detector_objs_goal = cam.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
            for obj in detector_objs:
                if obj is not None:
                    frames+=1
                    
                    if obj.ball > 0:
                        ball_on += 1
                        framepos.append(obj.position)
                    if obj.processed == True:
                        processed_cntr+=1
            '''if detector_objs_goal!=[]:
                for obj in detector_objs_goal:
                    if obj is not None:
                        goal_frames+=1
                        if obj.ball > 0:
                            goal_ball_on += 1
                        if obj.processed == True:
                            goal_processed_cntr+=1'''

        
        if self.best_camera == -1:
            if self.last_best_camera!=-1:
                self.best_camera = self.last_best_camera
            else:
                self.best_camera = self.cameras[0].camera_id
        for cam in self.cameras:
            if cam.camera_id==self.best_camera:
                detector_objs = cam.get_detector_collection(self.detector_cycle)
                print("best:", self.best_camera, "second:", self.second_best_camera)
                self.fill_gaps(cam, self.detector_cycle, self.second_best_camera)
                
                self.last_best_camera_obj = cam
                
                det_frames=[]
                for det in detector_objs:
                    if det is not None and det.processed==True:
                        det_frames.append(det.position)
                print("DETECTION FRAMES ALL:", det_frames)
                
                break
        
        for camera in self.cameras:
            camera.detector_cycle_completed(self.detector_cycle - 4)

        
        self.broadcast_camera = self.best_camera
        if (self.best_camera!=0 and self.best_camera!=1):
            if second_best_camera_ball_on>3:
                self.broadcast_camera = second_best_camera

        self.last_best_camera = self.best_camera
        elapsed = round((time.monotonic_ns() - self.stage_start_time) /1000 /1000)
        print(f'\n***********************')
        print(f'**** FINAL -> Tota ball frame: {ball_on} / {processed_cntr}, {frames},{elapsed}')
        print(f'***********************\n')
        framepos = sorted(set(framepos))
        self.last_proc_time = elapsed
        logging.info(f'{self.detector_cycle};{self.best_camera};{ball_on};{processed_cntr};{frames};{goal_ball_on};{goal_processed_cntr};{goal_frames};{elapsed};{framepos}')

        print(f'Cycle {self.detector_cycle} COMPLETED in {elapsed} ms')



    def linear_interpolate(self,start_value, end_value, steps):
        """Linearly interpolate between two values over a given number of steps."""
        return [(start_value + (end_value - start_value) * step / (steps - 1)) for step in range(steps)]


    def identify_wrong_numbers(self, x_coords):
        # Helper function to calculate the average change between non-missing, consecutive values
        def calculate_average_change(cleaned_coords):
            diffs = [abs(cleaned_coords[i+1] - cleaned_coords[i]) for i in range(len(cleaned_coords)-1)]
            return sum(diffs) / len(diffs) if diffs else 0
        
        # Clean the input list by removing missing values (-1)
        cleaned_coords = [x for x in x_coords if x != -1]
        
        # Calculate the average change to set a dynamic threshold
        average_change = calculate_average_change(cleaned_coords)
        threshold = average_change * 6  # Adjust the threshold as needed
        
        wrong_numbers = []  # List to store identified wrong numbers
        
        # Iterate through the list of cleaned coordinates to identify outliers
        for i, x in enumerate(cleaned_coords):
            # For each point, check if it significantly deviates from the trajectory of its neighbors
            if i > 0 and i < len(cleaned_coords) - 1:  # Ensure there are both previous and next items
                prev_val = cleaned_coords[i-1]
                next_val = cleaned_coords[i+1]
                if abs(prev_val - x) > threshold and abs(next_val - x) > threshold:
                    wrong_numbers.append(x)
                    
        return wrong_numbers

    def remove_wrong(self,cam, cycle):
        numbers_x =[]
        numbers_y =[]
        detector_objs = cam.get_detector_collection(cycle)
        for obj in detector_objs:
            if obj is not None:
                if obj.x1>-1:
                    numbers_x.append(obj.x1)
                if obj.y1>-1:
                    numbers_y.append(obj.y1)

        print("numbers_x:", numbers_x)
        print("numbers_y:", numbers_y)
        wrong_x = self.identify_wrong_numbers(numbers_x)
        wrong_y = self.identify_wrong_numbers(numbers_y)
        print("Wrong x numbers:", wrong_x)
        print("Wrong y numbers:", wrong_y)

        for obj in detector_objs:
            if obj is not None and (obj.x1 in wrong_x or obj.y1 in wrong_y):
                obj.x1=-1
                obj.x2=-1
                obj.y1=-4
                obj.y2=-1
                obj.confidence = 0

    def calculate_speed(self, detector_objs):
        processing_complete = False


        for obj in detector_objs:
            if obj is not None and obj.x1 != -1:  # Check if data is valid
                    if (self.ball_color_counter[obj.camera_id]>100):
                        curr_avrg = self.ball_color_total[obj.camera_id] / self.ball_color_counter[obj.camera_id]
                        if abs(curr_avrg-obj.mean_saturation)>50:
                            obj.x1 = -1
                            obj.y1 = -1
                            obj.x2 = -1
                            obj.y2 = -1
                            obj.confidence = -9
                            obj.x_2d = -1
                            obj.y_2d = -1                            


        while not processing_complete:
            last_valid_obj_x = None
            last_valid_obj_y = None
            last_valid_obj_x_2d = None
            last_valid_obj_y_2d = None
            last_valid_obj_position = None
            frame_gap_count = 0  # Counter for frames between valid data points
            restart_iteration = False

            for obj in detector_objs:
                if obj is not None and obj.x1 != -1 and obj.x_2d!=-1:  # Check if data is valid
                    if last_valid_obj_position is not None:
                        # Calculate the time elapsed, considering the frame gap
                        #frame_gap_count = obj.position - last_valid_obj_position
                        time_s = 0.04 * (frame_gap_count+1)  # Each frame is 40 ms

                        # Calculate distance in pixels
                        distance_px = math.sqrt((obj.x_2d - last_valid_obj_x_2d) ** 2 + (obj.y_2d - last_valid_obj_y_2d) ** 2)

                        # Convert distance from pixels to meters (using given field dimensions and pixel ranges)
                        distance_m = (distance_px / 500) * 25  # 500 pixels = 25 meters width

                        # Calculate speed in m/s
                        speed_ms = distance_m / time_s
                        # Convert speed to km/h (1 m/s = 3.6 km/h)
                        speed_kmh = speed_ms * 3.6
                        print(f'-> speed calc [{obj.position} - {last_valid_obj_position}] speed: {speed_kmh}, time_s: {time_s}, gap: {frame_gap_count}, dist(px): {distance_px}, distance_m: {distance_m}, time_s:{time_s},x: {obj.x_2d} - {last_valid_obj_x_2d},y: {obj.y_2d} - {last_valid_obj_y_2d}')

                        # Calculate direction in radians
                        dy = obj.y1- last_valid_obj_y
                        dx = obj.x1 - last_valid_obj_x
                        angle_radians = math.atan2(dy, dx)
                        # Convert radians to degrees
                        angle_degrees = math.degrees(angle_radians)
                        angle_degrees = (angle_degrees + 360) % 360

                        # Store speeds in the object
                        obj.ball_speed_ms = round(speed_ms, 1)
                        obj.ball_speed_kmh = round(speed_kmh, 1)
                        obj.ball_direction = int(angle_degrees)

                        # Check if the speed exceeds 140 km/h
                        if speed_kmh > 135:
                            obj.x1 = -1
                            obj.y1 = -1
                            obj.x2 = -1
                            obj.y2 = -1
                            obj.confidence = -8
                            obj.x_2d = -1
                            obj.y_2d = -1                            
                            restart_iteration = True
                            break

                        self.ball_color_counter[obj.camera_id]+=1
                        self.ball_color_total[obj.camera_id]+=obj.mean_saturation

                        # Reset frame gap counter after processing
                        frame_gap_count = 0

                    # Update last valid object
                    last_valid_obj_x = obj.x1
                    last_valid_obj_y = obj.y1
                    last_valid_obj_x_2d = obj.x_2d
                    last_valid_obj_y_2d = obj.y_2d
                    last_valid_obj_position = obj.position
                else:
                    # Increment the frame gap counter if current data point is invalid
                    frame_gap_count += 1

            if not restart_iteration:
                processing_complete = True
            else:
                # Reset necessary variables if we need to restart
                last_valid_obj_position = None
                frame_gap_count = 0

    def fill_gaps(self, cam, cycle, second_cam_id):
        print("Fill gap")
        cam.initial_gap_fill(cycle)
        second_cam =None
        for seccam in self.cameras:
            if seccam.camera_id==second_cam_id:
                second_cam = seccam

        
        #self.remove_wrong(cam, cycle)
        
        detector_objs = cam.get_detector_collection(cycle)
        second_detector_objs=None
        if (second_cam is not None):
            second_detector_objs = second_cam.get_detector_collection(cycle)
        
        if (second_detector_objs is not None):
            c=0
            for obj in detector_objs:
                if obj.x1 == -1 and c < len(second_detector_objs) and second_detector_objs[c]!=None and second_detector_objs[c].x1 != -1:
                    print("FFilled")
                    obj.x1 = second_detector_objs[c].x1
                    obj.y1 = second_detector_objs[c].y1
                    obj.x2 = second_detector_objs[c].x2
                    obj.y2 = second_detector_objs[c].y2
                    obj.x_2d = second_detector_objs[c].x_2d
                    obj.y_2d = second_detector_objs[c].y_2d
                    obj.confidence = 0.19

                c+=1
        '''if len(detector_objs)>0 and self.last_best_camera_obj is not None and detector_objs[0].x1==-1:
            old_detector_objs = self.last_best_camera_obj.get_detector_collection(cycle-1)
            last_known_det_obj=None
            for odo in old_detector_objs:
                if odo is not None and odo.x1!=-1:
                    last_known_det_obj = odo
            if last_known_det_obj is not None:
                xx, yy = cam.convert_to_3d(last_known_det_obj.x_2d, last_known_det_obj.y_2d)
                rad = last_known_det_obj.ball_radius
                detector_objs[0].x1=int(xx -rad/2)
                detector_objs[0].x2=int(xx +rad/2)
                detector_objs[0].y1=int(yy -rad/2)
                detector_objs[0].y2=int(yy +rad/2)
                
                detector_objs[0].x_2d=last_known_det_obj.x_2d
                detector_objs[0].y_2d=last_known_det_obj.y_2d
                
                detector_objs[0].ball_speed_ms=last_known_det_obj.ball_speed_ms
                detector_objs[0].ball_speed_kmh=last_known_det_obj.ball_speed_kmh
                detector_objs[0].ball_direction=last_known_det_obj.ball_direction

                if last_known_det_obj.confidence<=0:
                    detector_objs[0].confidence=0
                else:
                    detector_objs[0].confidence=0.12
                
                detector_objs[0].fake = True'''
        
        #self.calculate_speed(detector_objs)

        cntr=0
        total_confidence=0
        valid_obj_collection=[]
        for obj in detector_objs:
            if obj is not None and obj.confidence>0:
                valid_obj_collection.append(obj)
                cntr+=1
                total_confidence+=obj.confidence
        avrg_confidence = 0
        if cntr>0:
            avrg_confidence=total_confidence / cntr
        
        print("AVRG confidence:",avrg_confidence, "Entries:", cntr)

        length = len(valid_obj_collection)

        if length>1:

            if valid_obj_collection[0].x1==-1 and self.last_best_camera_obj is not None:
                last,_=self.last_best_camera_obj.get_last_value(cycle-1)
                if last is not None and last.position>20 and last.x1!=-1:
                    valid_obj_collection[0].x1 = last.x1
                    valid_obj_collection[0].y1 = last.y1
                    valid_obj_collection[0].x2 = last.x2
                    valid_obj_collection[0].y2 = last.y2
                    valid_obj_collection[0].x_2d = last.x_2d
                    valid_obj_collection[0].y_2d = last.y_2d                    
                    valid_obj_collection[0].ball_speed_ms = last.ball_speed_ms
                    valid_obj_collection[0].ball_speed_kmh = last.ball_speed_kmh
                    valid_obj_collection[0].ball_direction = last.ball_direction
                    valid_obj_collection[0].confidence = 0.09
                    #cam.convert_to_2d(valid_obj_collection[0])


            for c in range(length-1):
                obj_left = valid_obj_collection[c]
                obj_right = valid_obj_collection[c+1]
                #print("objc", c, c+1)
                dif = obj_right.position - obj_left.position
                #print("dif", dif)
                if (dif)>1:
                    x2_step=(obj_right.x2 - obj_left.x2)/dif
                    x2 = obj_left.x2
                    x1_step=(obj_right.x1 - obj_left.x1)/dif
                    x1 = obj_left.x1
                    y2_step=(obj_right.y2 - obj_left.y2)/dif
                    y2 = obj_left.y2
                    y1_step=(obj_right.y1 - obj_left.y1)/dif
                    y1 = obj_left.y1

                    x2d_step=(obj_right.x_2d - obj_left.x_2d)/dif
                    x_2d = obj_left.x_2d
                    y2d_step=(obj_right.y_2d - obj_left.y_2d)/dif
                    y_2d = obj_left.y_2d

                    #speed_ms_step=(obj_right.ball_speed_ms - obj_left.ball_speed_ms)/dif
                    speed_ms = obj_left.ball_speed_ms                    

                    #speed_kmh_step=(obj_right.ball_speed_kmh - obj_left.ball_speed_kmh)/dif
                    speed_kmh = obj_left.ball_speed_kmh                    

                    #direction_step=(obj_right.ball_direction - obj_left.ball_direction)/dif
                    direction = obj_left.ball_direction                    

                    for i in range(obj_left.position+1, obj_right.position):
                        x2+=x2_step
                        x1+=x1_step
                        y1+=y1_step
                        y2+=y2_step
                        x_2d+=x2d_step
                        y_2d+=y2d_step
                        

                        #speed_kmh+=speed_kmh_step
                        #speed_ms+=speed_ms_step
                        #direction+=direction_step
                        detector_objs[i].x2=int(x2)
                        detector_objs[i].x1=int(x1)
                        detector_objs[i].y2=int(y2)
                        detector_objs[i].y1=int(y1)
                        detector_objs[i].x_2d=int(x_2d)
                        detector_objs[i].y_2d=int(y_2d)
                        
                        try:
                            detector_objs[i].ball_speed_ms=round(speed_ms,1)
                            detector_objs[i].ball_speed_kmh=int(speed_kmh)
                            detector_objs[i].ball_direction=int(direction)
                        except:
                            detector_objs[i].ball_speed_ms=-1
                            detector_objs[i].ball_speed_kmh=-1
                            detector_objs[i].ball_direction=0

                        

                        detector_objs[i].confidence=0.11
                        
                        if (obj_right.confidence==0.19):
                           detector_objs[i].x1=-1
                           detector_objs[i].x2=-1
                           detector_objs[i].y1=-1 
                           detector_objs[i].y2=-1
                        else:
                            cam.convert_to_2d(detector_objs[i])
                        #cam.set_detector_obj(cycle,i, -8)
                #print("dif:", dif, "done:", c)

        self.calculate_speed(detector_objs)
    '''def fill_gaps_old(self, records):

        # Initialize variables
        last_valid_index = None
        last_valid_values = None

        # Iterate through records
        for i, record in enumerate(records):
            # Check if current record is valid
            if record.x1 != -1:
                if last_valid_index is not None and (i - last_valid_index) > 1:
                    # Calculate steps (including start, excluding end)
                    steps = i - last_valid_index
                    # Perform linear interpolation for each field
                    interpolated_values = {
                        'x1': self.linear_interpolate(last_valid_values.x1, record.x1, steps),
                        'y1': self.linear_interpolate(last_valid_values.y1, record.y1, steps),
                        'x2': self.linear_interpolate(last_valid_values.x2, record.x2, steps),
                        'y2': self.linear_interpolate(last_valid_values.y2, record.y2, steps),
                    }
                    # Update records with interpolated values
                    for j in range(1, steps):
                        records[last_valid_index + j].x1 = int(interpolated_values['x1'][j-1])
                        records[last_valid_index + j].y1 = int(interpolated_values['y1'][j-1])
                        records[last_valid_index + j].x2 = int(interpolated_values['x2'][j-1])
                        records[last_valid_index + j].y2 = int(interpolated_values['y2'][j-1])
                        records[last_valid_index + j].confidence = 0.11

                # Update last valid record and index
                last_valid_index = i
                last_valid_values = record
            # If the record is the last one and invalid, fill with the last valid value
            elif i == len(records) - 1 and last_valid_values is not None:
                for j in range(last_valid_index + 1, len(records)):
                    records[j].x1 = int(last_valid_values.x1)
                    records[j].y1 = int(last_valid_values.y1)
                    records[j].x2 = int(last_valid_values.x2)
                    records[j].y2 = int(last_valid_values.y2)
                    records[j].confidence = 0.11'''

    def run_stage_1(self, camera_id=-1, number_of_frames=6):
        print("Stage-1 - cam_id", camera_id, " frames to detect:", number_of_frames)
        if number_of_frames==0:
            return
        total_frm = 0
        for camera in self.cameras:
            #print("CHECKING CAM", camera.camera_id)
            detector_objs = camera.get_non_none_detector_collection(self.detector_cycle)
            #print("DET OBJECTS:", len(detector_objs))
            total_frm += len(detector_objs)
        if (total_frm<12):
            print("LEAVING CYCLE - EMPTY")
            self.current_stage =99
            #self.run_stage_final()
            return
        
        print(f'Total frames to use in this [{self.detector_cycle}] detection cycle: {total_frm} frames')

        for camera in self.cameras:
            frames_marked = []#[-1,self.max_fps]
            if camera_id==-1 or camera.camera_id==camera_id:
                if (camera.capture_only_device == True):
                    continue

                #if (camera.camera_id!=2 and camera.camera_id!=3):
                #    continue

                available_frames_to_process=[]
                detector_objs = camera.get_detector_collection(self.detector_cycle)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.skip==True and det_obj.in_progress == False:
                            available_frames_to_process.append(det_obj.position)
                print(f'[CAM{camera.camera_id}] Available frames for detection: {len(available_frames_to_process)}')
                print(f'marking a maximum of {number_of_frames} frames for analyses from ')#,available_frames_to_process )
                if len(available_frames_to_process)>0:
                    for i in range(number_of_frames):
                        frame_pos = insert_for_even_distribution(frames_marked, available_frames_to_process)

                        if (frame_pos is not None and frame_pos>=0):
                            try:
                                for det_obj in detector_objs:
                                    if det_obj is not None and det_obj.position == frame_pos:
                                        det_obj.skip=False
                                        frames_marked.append(frame_pos)
                                        break
                            except:
                                print(f'ERROR - FAILED TO get frame from position {frame_pos} on camera {camera.camera_id}')

                    print(f'[CAM{camera.camera_id}] has marked {len(frames_marked)} for analyses.')#, frames_marked)


        if self.goal_l_reference_taken==False:
            for camera in self.cameras:
                if camera.camera_id==11:
                    l_cntr=0
                    detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                    for det_obj in detector_objs:
                        if (det_obj is not None and det_obj.frame is not None):
                            det_obj.skip=False
                            l_cntr+=1
                            if l_cntr>=6:
                                break
                if camera.camera_id==10:
                    r_cntr=0
                    detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                    for det_obj in detector_objs:
                        if (det_obj is not None and det_obj.frame is not None):
                            det_obj.skip=False
                            r_cntr+=1
                            if r_cntr>=6:
                                break

        self.run_detection()

    def run_detection(self):
        engine_position=0
        start = time.monotonic_ns()
        for camera in self.cameras:
            if camera.capture_only_device:
                continue
                
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for det_obj in detector_objs:
                if det_obj is not None and det_obj.skip==False and det_obj.processed==False and det_obj.in_progress==False:
                    det_obj.in_progress=True
                    
                    # Select appropriate engine based on camera resolution
                    # Priority: Use your custom 2300x896 engine if available
                    if hasattr(camera, 'frame_w') and (camera.frame_w == 2300 or camera.frame_w >= 2000):
                        engine = self.engines_2300
                        engine_w = self.W_2300
                        engine_h = self.H_2300
                    elif hasattr(camera, 'frame_w') and camera.frame_w == 1280:
                        engine = self.engines_1280
                        engine_w = self.W_1280
                        engine_h = self.H_1280
                    else:
                        engine = self.engines_960
                        engine_w = self.W_960
                        engine_h = self.H_960

                    self.futures.append(self.executor.submit(
                        self.process_frame, camera, self.detector_cycle, det_obj, 
                        engine[engine_position], engine_w, engine_h
                    ))
                    
                    engine_position += 1
                    if engine_position >= self.paralell_models:
                        engine_position = 0

        elapsed = (time.monotonic_ns() - start) / 1000 / 1000
        if elapsed > 2:
            print("run_detection process time:", elapsed)

    '''def run_goal_detection(self):
        engine_position=0
        start = time.monotonic_ns()
        for camera in self.cameras:
            if (camera.camera_id == 10 or camera.camera_id == 11):
                detector_objs = camera.get_detector_collection(self.detector_cycle, DetectionType.GOAL)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.frame is not None:
                        h, w = det_obj.frame.shape[:2]
                        if h ==288 and w == 288:
                            self.futures.append(self.executor.submit(self.process_frame_goal, camera, self.detector_cycle, det_obj, self.goal_engines[engine_position], 
                                                                    self.goal_W, self.goal_H))
                            engine_position+=1
                            if (engine_position>=self.paralell_models):
                                engine_position=0
                        else:
                            print("****ERROR: Incorrect goal img size...", h, w, self.goal_W, self.goal_H)

        elapsed = (time.monotonic_ns() - start) /1000 /1000
        if (elapsed>2):
            print("run_detection process time:", elapsed)         '''   

    def run_goal_detection2(self):
        '''engine_position=0
        start = time.monotonic_ns()
        

        for camera in self.cameras:
            if (camera.camera_id == 11):
                detector_objs = camera.get_detector_collection(self.detector_cycle, DetectionType.GOAL)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.frame is not None:
                        if self.cameras[2].detector_objects[self.detector_cycle][det_obj.position] is not None:
                            players = self.cameras[2].detector_objects[self.detector_cycle][det_obj.position].players
                            
                            left_goalkeeper=None
                            if (players):
                                for player in players:
                                    if (left_goalkeeper is None or left_goalkeeper.x1>player.x1):
                                        left_goalkeeper = player

                            if (left_goalkeeper is None):
                                left_goalkeeper = self.last_left_goalkeeper

        last_frame = None    
        for camera in self.cameras:
            if (camera.camera_id == 11):
                detector_objs = camera.get_detector_collection(self.detector_cycle, DetectionType.GOAL)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.frame is not None:
                        if last_frame is not None:
                            det_obj.frame = det_obj.frame.copy()
                            # Convert the images to grayscale
                            gray1 = cv2.cvtColor(det_obj.frame, cv2.COLOR_BGR2GRAY)
                            gray2 = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

                            # Compute the absolute difference between the images
                            difference = cv2.absdiff(gray1, gray2)

                            # Increase the threshold to make the comparison less sensitive
                            _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

                            # Optionally, focus on high-brightness differences to highlight white or near-white objects
                            # Mask for areas of high brightness in both images
                            min_white=160
                            high_brightness1 = cv2.inRange(det_obj.frame, (min_white, min_white, min_white), (255, 255, 255))
                            high_brightness2 = cv2.inRange(last_frame, (min_white, min_white, min_white), (255, 255, 255))
                            brightness_mask = cv2.bitwise_and(high_brightness1, high_brightness2)

                            # Combine the brightness mask with the threshold mask
                            final_mask = cv2.bitwise_and(thresh, brightness_mask)

                            # Find contours from the thresholded difference image
                            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            

                            last_frame = det_obj.frame.copy()
                            #det_obj.frame = thresh
                            # Draw contours on the original image
                            ##for contour in contours:
                             #   x, y, w, h = cv2.boundingRect(contour)
                              #  cv2.rectangle(det_obj.frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

                            for contour in contours:
                                # Minimum enclosing circle
                                (x, y), radius = cv2.minEnclosingCircle(contour)
                                # Check if the radius meets the minimum size
                                if radius >= 1:
                                    # Calculate circle's area and the contour's area
                                    circle_area = np.pi * (radius ** 2)
                                    contour_area = cv2.contourArea(contour)
                                    # Check if the shape is circular by comparing the areas
                                    if contour_area >= 0.5 * circle_area:
                                        # Draw the circle on the image
                                        center = (int(x), int(y))
                                        cv2.circle(det_obj.frame, center, int(radius), (0, 255, 0), 4)                                
                        else:
                            last_frame = det_obj.frame.copy()

                        

        return'''

        '''for camera in self.cameras:
            if (camera.camera_id == 11):
                detector_objs = camera.get_detector_collection(self.detector_cycle, DetectionType.GOAL)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.frame is not None:
                        if self.cameras[2].detector_objects[self.detector_cycle][det_obj.position] is not None:
                            players = self.cameras[2].detector_objects[self.detector_cycle][det_obj.position].players
                            
                            left_goalkeeper=None
                            if (players):
                                for player in players:
                                    if (left_goalkeeper is None or left_goalkeeper.x1>player.x1):
                                        left_goalkeeper = player

                            if (left_goalkeeper is None):
                                left_goalkeeper = self.last_left_goalkeeper
                            if (left_goalkeeper is not None):
                                
                                x_offset=116
                                y_offset = 102
                                x1=left_goalkeeper.x1 - x_offset
                                x2=left_goalkeeper.x2 - x_offset
                                y1=left_goalkeeper.y1 - y_offset
                                y2=left_goalkeeper.y2 - y_offset
                                det_obj.frame=det_obj.frame.copy()
                                #cv2.rectangle(det_obj.frame, (20,20), (40, 40), (0,0,255), 1)
                                print("->>>>>>>>",x1, y1, x2, y2)
                                cv2.rectangle(det_obj.frame, (x1,y1), (x2, y2), (0,0,0), -1)
                                if self.last_left_goalkeeper is not None:
                                    gray1 = cv2.cvtColor(det_obj.frame, cv2.COLOR_BGR2GRAY)
                                    gray2 = cv2.cvtColor(self.last_left_goalkeeper, cv2.COLOR_BGR2GRAY)

                                    # Calculate the difference and apply threshold
                                    diff = cv2.subtract(gray1, gray2)
                                    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

                                    # Highlight in the original image
                                    result = self.last_left_goalkeeper
                                    det_obj.frame[mask == 255] = (0, 0, 255)  # Highlight in red color

                                    self.last_left_goalkeeper = det_obj.frame'''
                                    


                                


                        

        ##elapsed = (time.monotonic_ns() - start) /1000 /1000
        #if (elapsed>2):
        #    print("run_detection process time:", elapsed)            


    '''def process_results(self):
        while self.futures:
                for future in as_completed(self.futures):
                    try:
                        camera_id, cycle_id, det_obj = future.result()
                        det_obj.processed = True
                        self.completed_futures.add(future)  # Mark this future as completed
                        #print("COMPLETED")
                    except Exception as e:
                        print(f"Operation generated an exception: {e}")
                        self.completed_futures.add(future)  # Mark this future as completed, even if there was an exception

                self.futures = [f for f in self.futures if f not in self.completed_futures]
                self.completed_futures.clear()  # Clear the completed futures set for the next iteration'''


    def process_results_non_blocking(self):
        start_time = time.time()  # Capture start time

        # Temporarily hold futures that are not yet completed
        pending_futures = []

        for future in self.futures:
            if future.done():  # Check if the future is done
                try:
                    camera_id, cycle_id, det_obj = future.result(timeout=0)  # Non-blocking
                    det_obj.processed = True
                    # Mark this future as completed, process if necessary
                except Exception as e:
                    print(f"Operation generated an exception: {e}")
            else:
                pending_futures.append(future)  # Re-queue the future if it's not done

        # Update the list of futures with those still pending
        self.futures = pending_futures

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        return elapsed_time  # Optionally return the elapsed time for monitoring


    def get_image(self, tensor):
        return get_image_from_gpu(tensor)


    '''    def deep_check(self, cycle_id, frame, x1, y1, x2, y2):
        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]
        
        # Calculate total number of pixels in the region
        total_pixels = region.shape[0] * region.shape[1]
        
        # Define thresholds
        white_threshold_lower = np.array([190, 190, 190], dtype="uint8")  # Lower bound for white
        white_threshold_upper = np.array([255, 255, 255], dtype="uint8")  # Upper bound for white
        gray_threshold_lower = np.array([0, 0, 0], dtype="uint8")  # Adjust these values based on need
        gray_threshold_upper = np.array([190, 190, 190], dtype="uint8")  # Upper bound for gray (less than white lower bound)

        # Mask for white pixels
        white_mask = cv2.inRange(region, white_threshold_lower, white_threshold_upper)
        # Mask for gray pixels
        gray_mask = cv2.inRange(region, gray_threshold_lower, gray_threshold_upper)

        # Calculate white and gray pixels
        white_pixels = np.sum(white_mask == 255)
        gray_pixels = np.sum(gray_mask == 255)

        # Total white or gray pixels
        white_gray_total = white_pixels + gray_pixels

        # Calculate the percentage of the region that appears to be white or gray
        white_gray_percentage = (white_gray_total / total_pixels) * 100

        if white_gray_percentage < 50:  # Less than 70% of the region is white or gray
            self.false_detection += 1
            #cv2.imwrite(f'false{self.false_detection}.png', region)
            #return False

        return white_gray_percentage'''
    def deep_check(self, cycle_id, frame, x1, y1, x2, y2):
        # Calculate center and radius of the circle within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]

        # Create a circular mask
        mask = np.zeros(region.shape[:2], dtype="uint8")
        cv2.circle(mask, (radius, radius), radius, 255, -1)

        # Apply the mask to the region to isolate the circle
        masked_region = cv2.bitwise_and(region, region, mask=mask)

        # Define color thresholds for white and gray
        white_threshold_lower = np.array([190, 190, 190], dtype="uint8")
        white_threshold_upper = np.array([255, 255, 255], dtype="uint8")
        gray_threshold_lower = np.array([0, 0, 0], dtype="uint8")
        gray_threshold_upper = np.array([190, 190, 190], dtype="uint8")

        # Create masks for detecting white and gray pixels within the circular region
        white_mask = cv2.inRange(masked_region, white_threshold_lower, white_threshold_upper)
        gray_mask = cv2.inRange(masked_region, gray_threshold_lower, gray_threshold_upper)

        # Count the number of white and gray pixels
        white_pixels = np.sum(white_mask == 255)
        gray_pixels = np.sum(gray_mask == 255)

        # Calculate the total number of pixels within the circle
        total_pixels = np.sum(mask == 255)

        # Calculate the percentage of white and gray pixels within the circle
        if total_pixels > 0:
            white_gray_percentage = white_pixels/ gray_pixels * 100
        else:
            white_gray_percentage = 0  # Avoid division by zero

        return white_gray_percentage

    def deep_check2(self, cycle_id, frame, x1, y1, x2, y2):
        # Calculate center and radius of the circle within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]

        # Create a circular mask
        mask = np.zeros(region.shape[:2], dtype="uint8")
        cv2.circle(mask, (center_x - x1, center_y - y1), radius, 255, -1)  # Adjust circle position in mask

        # Apply the mask to the region to isolate the circle
        masked_region = cv2.bitwise_and(region, region, mask=mask)
        if masked_region is None:
            return 255, 255, 0

        # Convert the masked region to HSV color space
        hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)

        # Calculate the average saturation and value in the masked region
        masked_hsv = cv2.bitwise_and(hsv_region, hsv_region, mask=mask)
        s_channel = masked_hsv[:, :, 1]  # Saturation channel
        v_channel = masked_hsv[:, :, 2]  # Value (brightness) channel
        mean_saturation = np.mean(s_channel[s_channel > 0])  # Mean saturation of non-zero mask areas
        mean_value = np.mean(v_channel[v_channel > 0])  # Mean value of non-zero mask areas

        # Deviation from perfect white
        # Perfect white is Saturation = 0 and Value = 255
        deviation = np.sqrt((255 - mean_value) ** 2 + mean_saturation ** 2)

        # Check if the ball is considered 'white'
        #is_white = mean_saturation < 20 and mean_value > 235  # Thresholds can be adjusted

        if mean_saturation==np.nan:
            mean_saturation=255
        if mean_value==np.nan:
            mean_value=255

        return mean_saturation, mean_value, 0

    def deep_check3(self, cycle_id, frame, x1, y1, x2, y2):
        # Calculate center and radius of the circle within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]

        # Create a circular mask
        mask = np.zeros(region.shape[:2], dtype="uint8")
        cv2.circle(mask, (center_x - x1, center_y - y1), radius, 255, -1)  # Adjust circle position in mask

        # Apply the mask to the region to isolate the circle
        masked_region = cv2.bitwise_and(region, region, mask=mask)
        if masked_region is None:
            return 255, 255, 0

        # Convert the masked region to HSV color space
        hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)

        # Calculate the average saturation and value in the masked region
        masked_hsv = cv2.bitwise_and(hsv_region, hsv_region, mask=mask)
        s_channel = masked_hsv[:, :, 1]  # Saturation channel
        v_channel = masked_hsv[:, :, 2]  # Value (brightness) channel
        
        # Only consider the non-zero mask areas
        non_zero_mask = mask > 0
        if np.count_nonzero(non_zero_mask) == 0:
            return 255, 255, 0  # Return if the mask is completely zero
        
        mean_saturation = np.mean(s_channel[non_zero_mask])  # Mean saturation of non-zero mask areas
        mean_value = np.mean(v_channel[non_zero_mask])  # Mean value of non-zero mask areas

        # Deviation from perfect white
        # Perfect white is Saturation = 0 and Value = 255
        deviation = np.sqrt((255 - mean_value) ** 2 + mean_saturation ** 2)

        # Additional checks to improve accuracy
        # Check if the ball is considered 'white'
        is_white = mean_saturation < 20 and mean_value > 235  # Thresholds can be adjusted
        
        # Edge detection to help handle partial occlusions
        gray_masked = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_masked, 100, 200)
        edge_density = np.sum(edges > 0) / (np.pi * radius * radius)
        is_edge_dense = edge_density < 0.2  # Adjust based on ball size and edge density expectations

        # Final decision
        if is_white and is_edge_dense:
            return mean_saturation, mean_value, 1  # Return a flag indicating it is likely a ball
        else:
            return mean_saturation, mean_value, 0
        
    def process_frame(self, camera, cycle_id, det_obj, Engine, W, H):
        try:
            # Resize input frame if necessary to match engine requirements
            det_obj.tensor = create_tensor(det_obj.frame, self.device, W, H)
            
            if (det_obj.tensor is None):
                print(f"Failed to create tensor for camera {camera.camera_id}, cycle {cycle_id}")
                return camera.camera_id, cycle_id, det_obj

            data = Engine(det_obj.tensor)  # Process the batch through the engine

            bboxes, scores, labels = det_postprocess(data)
            
            if bboxes.numel() == 0:
                # No detections
                pass
            else:
                highest_score_ball = -1
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)

                    x1 = bbox[:2][0]
                    y1 = bbox[:2][1]
                    x2 = bbox[2:][0]
                    y2 = bbox[2:][1]       

                    # Handle player detection
                    if cls_id == CLASSID_PLAYER and score >= self.ai_settings.people_confidence:
                        if (camera.camera_id == 2 or camera.camera_id == 3) and y2 < 76:
                            # Skip the people who are on the other field
                            pass
                        elif y2 - y1 > 80:  # If the box is not high enough, ignore
                            det_obj.people += 1
                            player = Player()
                            player.x1 = int(x1)
                            player.x2 = int(x2)
                            player.y1 = int(y1)
                            player.y2 = int(y2)
                            player.confidence = round(float(score), 2)
                            det_obj.players.append(player)

                    # Handle ball detection
                    if cls_id == CLASSID_BALL and score >= self.ai_settings.ball_confidence:
                        if score < highest_score_ball:
                            det_obj.ball += 1
                        else:
                            MIN_BALL_SIZE = self.ai_settings.min_ball_size
                            rad = min(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)))

                            if rad >= MIN_BALL_SIZE:
                                if self.ai_settings.ball_do_deep_check:
                                    det_obj.mean_saturation, det_obj.mean_value, det_obj.white_gray = self.deep_check2(cycle_id, det_obj.frame, x1, y1, x2, y2)
                                    if (det_obj.mean_saturation > self.ai_settings.ball_mean_saturation or 
                                        det_obj.mean_value < self.ai_settings.ball_mean_value):
                                        continue
                                
                                highest_score_ball = score
                                det_obj.ball += 1
                                det_obj.x1 = int(x1)
                                det_obj.x2 = int(x2)
                                det_obj.y1 = int(y1)
                                det_obj.y2 = int(y2)
                                det_obj.ball_radius = int(rad)
                                det_obj.confidence = round(float(score), 2)
                                camera.convert_to_2d(det_obj)

                            else:
                                det_obj.x1 = -1
                                det_obj.x2 = -1
                                det_obj.y1 = -1
                                det_obj.y2 = -1
                                det_obj.x_2d = -1
                                det_obj.y_2d = -1
                                det_obj.ball_radius = -1
                                det_obj.confidence = -1

        except Exception as e:
            print(f"Error in process_frame: {e}")
            import traceback
            traceback.print_exc()
            
        return camera.camera_id, cycle_id, det_obj
