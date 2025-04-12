import time
import math
import logging
from src.gpu import *
from src.detector_utils import *
from src.camera import *
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

class AIDetector:
    def __init__(self, cameras, paralell_models, max_fps) -> None:

        logging.basicConfig(level=logging.DEBUG, filename='detector.log', filemode='w',
                    format='%(message)s')        
        self.cameras = cameras
        self.max_fps = max_fps
        self.paralell_models = paralell_models
        self.engines=[]
        self.goal_engines=[]
        self.futures=[]
        self.completed_futures = set()
        self.W = 0
        self.H = 0
        self.goal_W = 0
        self.goal_H = 0
        self.false_detection = 0        
        self.best_camera = -1
        self.last_proc_time = 0
        self.second_best_camera = -1
        self.detector_cycle = 0
        self.current_stage = 1
        self.executor = ThreadPoolExecutor(max_workers = self.paralell_models)
        self.outstanding_detections = 0
        self.last_outstanding_detections = 0
        self.device = create_device(0)

        self.stage_start_time = 0

        for m_id in range(self.paralell_models ):
            print("CREATE MODEL: ", m_id)
            self.W, self.H = create_engine(self.device, self.engines)
            #self.goal_W, self.goal_H = create_engine(self.device, self.goal_engines, create_goal_engine=True)

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
            self.run_stage_1(0,4)
            #self.current_stage = 2
            self.current_stage =110
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

                '''if ball_on<4 and self.second_best_camera!=-1:
                    print("DO attional detection on second best <<<")
                    self.do_detection_selected_camera(self.second_best_camera, 4)
                    self.run_detection()
                    self.current_stage = 6

                else:'''
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
            '''processed_cntr = 0
            ball_on=0
            for cam in self.cameras:
                if cam.camera_id == self.best_camera:
                    detector_objs = cam.get_detector_collection(self.detector_cycle)
                    for obj in detector_objs:
                        if obj is not None:
                            if obj.ball > 0:
                                ball_on += 1
                            if obj.processed == True:
                                processed_cntr+=1

            print(f'**** FINAL -> best camera: {self.best_camera}. Tota ball frame: {ball_on} / {processed_cntr}')'''


            self.current_stage = 99
            return        

        '''if (self.current_stage==99):
            print("---> STAGE 99")
            
            self.current_stage = 100

            return'''

        if (self.current_stage==99):
            #self.current_stage = 110
            #return
            '''print("Start goal line")
            engine_position = 0
            for camera in self.cameras:
                if (camera.camera_id<10):
                    continue
                detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                for det_obj in detector_objs:
                    if det_obj is not None and det_obj.processed==False and det_obj.in_progress==False:
                        det_obj.in_progress=True
                        det_obj.skip = False
                        self.futures.append(self.executor.submit(self.process_frame, camera.camera_id, self.detector_cycle, det_obj, self.goal_engines[engine_position], 
                                                                 self.goal_W, self.goal_H))
                        engine_position+=1
                        if (engine_position>=self.paralell_models):
                            engine_position=0'''

            self.current_stage = 110
            return


        if (self.current_stage == 110):
            self.run_stage_final()
    
            '''for camera in self.cameras:
                if (camera.camera_id<10):
                    continue
                detector_objs = camera.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
                for det_obj in detector_objs:
                    if det_obj is not None:
                        camera.print_detector_obj(det_obj)'''

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

        print("Best camera ->", best_camera, "with", best_camera_ball_on, "valid detections")
        print("Second best camera ->", second_best_camera, "with", second_best_camera_ball_on, "valid detections")

        self.best_camera = best_camera
        self.second_best_camera = second_best_camera

        if (just_do_selection_nothing_else==True):
            self.current_stage=99
            return

        if (best_camera!=-1):
            print("Ball on:", best_camera_ball_on, "avrg confidence:", best_camera_avrg_conf)
            print("RUNNING ADDITIONAL DETECTIONS")
            
            self.do_detection_selected_camera(best_camera, 12)
            #if (second_best_camera!=-1):
            #    self.do_detection_selected_camera(second_best_camera, 3)

            self.run_detection()
            #self.run_stage_1(best_camera, 6)
            #print("\n\n")
            
            self.current_stage = 3


            print("Going to next stage - 3")

        else:

            print("Let's do detection again on all cameras as the 1st resulted in -1")
            detections=8
            #for cam in self.cameras:
            #    self.do_detection_selected_camera(cam.camera_id, detections)
            self.do_detection_selected_camera(3, detections)
            self.do_detection_selected_camera(4, detections)
            
            self.run_detection()
            #self.current_stage = 4
                
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
                    pos = insert_for_even_distribution(frames_marked, available_frames_to_process)

                    if (pos is not None and detector_objs[pos] is not None):
                        detector_objs[pos].skip=False
                        frames_marked.append(pos)

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
            detector_objs_goal = cam.get_detector_collection(self.detector_cycle, detection_type=DetectionType.GOAL)
            for obj in detector_objs:
                if obj is not None:
                    frames+=1
                    
                    if obj.ball > 0:
                        ball_on += 1
                        framepos.append(obj.position)
                    if obj.processed == True:
                        processed_cntr+=1
            if detector_objs_goal!=[]:
                for obj in detector_objs_goal:
                    if obj is not None:
                        goal_frames+=1
                        if obj.ball > 0:
                            goal_ball_on += 1
                        if obj.processed == True:
                            goal_processed_cntr+=1

        
        if self.best_camera == -1:
            self.best_camera = 0 #IMRE
        
        
        
        
        for cam in self.cameras:
            if cam.camera_id==self.best_camera:
                detector_objs = cam.get_non_none_detector_collection(self.detector_cycle)
                self.fill_gaps(detector_objs)
                break
        
        for camera in self.cameras:
            camera.detector_cycle_completed(self.detector_cycle - 4)

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

    def fill_gaps(self, records):
        print("Fill_gaps(), len=", len(records))
        if len(records)==0:
            return
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
                    records[j].confidence = 0.11

    def run_stage_1(self, camera_id=-1, number_of_frames=4):
        print("Stage-1")
        total_frm = 0
        for camera in self.cameras:
            detector_objs = camera.get_non_none_detector_collection(self.detector_cycle)
            total_frm += len(detector_objs)
            if (total_frm<12):
                print("LEAVING CYCLE - EMPTY")
                self.current_stage =99
                #self.run_stage_final()
                return
        
        print(f'Total frames to use in this [{self.detector_cycle}] detection cycle: {total_frm}')

        frames_marked = [-1,self.max_fps+1]  # we have an entire second as a starting reference

        for camera in self.cameras:
            if camera_id==-1 or camera.camera_id==camera_id:
                if (camera.capture_only_device == True):
                    continue

                available_frames_to_process=[]
                detector_objs = camera.get_detector_collection(self.detector_cycle)
                for det_obj in detector_objs:
                    #if (camera.camera_id==1):
                    #camera.print_detector_obj(det_obj)
                    if det_obj is not None and det_obj.skip==True and det_obj.position not in frames_marked and det_obj.in_progress == False:
                            available_frames_to_process.append(det_obj.position)
                print(f'[CAM{camera.camera_id}] Available frames for detection: {len(available_frames_to_process)}')
                
                if len(available_frames_to_process)>0:
                    for i in range(number_of_frames):
                        pos = insert_for_even_distribution(frames_marked, available_frames_to_process)

                        if (pos is not None):
                            try:
                                detector_objs[pos].skip=False
                                frames_marked.append(pos)
                            except:
                                print(f'ERROR - FAILED TO get frame from position {pos} on camera {camera.camera_id}')
                            #print(frames_marked)



        self.run_detection()

    def run_detection(self):
        
        engine_position=0
        start = time.monotonic_ns()
        det_start = time.perf_counter()
        det_obj_collection=[]

        for camera in self.cameras:
            if (camera.capture_only_device == True):
                continue
            detector_objs = camera.get_detector_collection(self.detector_cycle)
            for det_obj in detector_objs:
                if det_obj is not None and det_obj.skip==False and det_obj.processed==False and det_obj.in_progress==False:
                    det_obj.in_progress=True
                    det_obj.check = 88
                    det_obj_collection.append(det_obj)
                    
                    if len(det_obj_collection)==4:
                        print(f'start detection on cam {det_obj.camera_id} position {det_obj.position} - cycle:{self.detector_cycle} - size:', len(det_obj_collection))
                        #self.futures.append(self.executor.submit(self.process_frame, det_obj_collection, self.detector_cycle, self.engines[engine_position], 
                        #                                        self.W, self.H))
                        self.process_frame(det_obj_collection, self.detector_cycle, self.engines[engine_position],self.W, self.H)

                        engine_position+=1
                        if (engine_position>=self.paralell_models):
                            engine_position=0
                        det_obj_collection=[]


        elapsed22 = (time.perf_counter() - det_start) * 1000
        print("elapsed22:", elapsed22)

        elapsed = (time.monotonic_ns() - start) /1000 /1000
        if len(det_obj_collection)!=0:
            print(".")
            print("..")
            print("...")
            print("ERROR!!!!!")
        if (elapsed>2):
            print("run_detection process time:", elapsed)

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
                    future.result(timeout=0)  # Non-blocking
                    #for det_obj in det_obj_collection:
                    #    print("Marking processed....", det_obj.x1)
                    #    det_obj.processed = True
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


    def deep_check(self, cycle_id, frame, x1, y1, x2, y2):
        # Extract the region from the frame
        region = frame[y1:y2, x1:x2]
        
        # Calculate total number of pixels in the region
        total_pixels = region.shape[0] * region.shape[1]
        
        # Define thresholds
        white_threshold_lower = np.array([180, 180, 180], dtype="uint8")  # Lower bound for white
        white_threshold_upper = np.array([255, 255, 255], dtype="uint8")  # Upper bound for white
        gray_threshold_lower = np.array([100, 100, 100], dtype="uint8")  # Adjust these values based on need
        gray_threshold_upper = np.array([180, 180, 180], dtype="uint8")  # Upper bound for gray (less than white lower bound)

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

        if white_gray_percentage < 70:  # Less than 70% of the region is white or gray
            self.false_detection += 1
            cv2.imwrite(f'false{self.false_detection}.png', region)
            return False

        return True      


    def process_frame(self, det_obj_collection, cycle_id, Engine, W, H):
        processed_tensors = []
        for det_obj in det_obj_collection:
            tensor = create_tensor_new(det_obj.frame, self.device,W,H)
            height, width, channels = det_obj.frame.shape

            #print(f'Width: {width}px')
            #print(f'Height: {height}px')           
            '''bgr = det_obj.frame
            # Preprocess each image
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.as_tensor(dwdh * 2, dtype=torch.float16, device=self.device)
            tensor = torch.as_tensor(tensor, device=self.device)'''
            # Append processed tensor
            print("Add tensor")
            processed_tensors.append(tensor.unsqueeze(0))  # Add batch dimension

        # Stack tensors to form a batch
        batch_tensor = torch.cat(processed_tensors)        

        print("Running batch on AI engine.... det_obj_collection.size=", len(det_obj_collection))
        data = Engine(batch_tensor)  # Process the batch through the engine
        print("Done running batch on AI engine - det_obj_collection.size=", len(det_obj_collection))
#        for det_obj in det_obj_collection:
#                det_obj.in_progress = False
#                det_obj.processed = True

        det_pos=0
        for det_obj in det_obj_collection:
            print("det_postprocess() - camid:", det_obj.camera_id, "pos:", det_obj.position, "check:",det_obj.check)
            bboxes, scores, labels = det_postprocess(data, pos=det_pos)
            det_pos+=1
            #draw = self.get_image_from_gpu(det_obj.tensor)
            #det_obj.y1=22
            if bboxes.numel() == 0:
                det_obj.in_progress = False
                det_obj.processed = True
                print(f'  no object!')

                pass
                #return instance, draw, 0, 0
            else:
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    cls = CLASSES[cls_id]

                    x1 = bbox[:2][0]
                    y1 = bbox[:2][1]
                    x2 = bbox[2:][0]
                    y2 = bbox[2:][1]       


                    if cls_id==0 and score>=0.3:
                        det_obj.people+=1
                        player = Player()
                        player.x1=int(x1)
                        player.x2=int(x2)
                        player.y1=int(y1)
                        player.y2=int(y2)
                        player.confidence=round(float(score),2)
                        det_obj.players.append(player)
                        #print(f'  Player!')

                    if cls_id==32 and score>=0.5:

                        #2 524 126 530 133
                        #356 73 362 77     
                        # 2 522 126 530 133                   
                        if (det_obj.camera_id==2 and (x1>=520 and x2<=532) and (y1>=124 and y2<=135)):
                            continue
                        if (det_obj.camera_id==1 and (x1>=355 and x2<=363) and (y1>=72 and y2<=78)):
                            continue

                        #is_real_ball = self.deep_check(cycle_id, det_obj.frame, x1, y1, x2, y2)
                        #if is_real_ball==False:
                        #    continue

                        #else:
                        det_obj.ball +=1
                        det_obj.x1=int(x1)
                        det_obj.x2=int(x2)
                        det_obj.y1=int(y1)
                        det_obj.y2=int(y2)
                        det_obj.confidence=round(float(score),2)
                        #print(f'  ball!')


                        #print(f'[{camera_id}, {det_obj.x1},{det_obj.x2},{det_obj.y1},{det_obj.y2}]')
                det_obj.processed = True
                det_obj.in_progress = False

        return
        return cycle_id, det_obj_collection
    #  bboxes -= dwdh
    #  bboxes /= ratio

        '''ball=0
        people=0
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            
            display=True
            if cls_id==0 and score>=0.4:
                people+=1
            elif cls_id==0 and score<0.4:
                display=False

            #print(cls_id)
            if (display):
                color = COLORS[cls]
                if cls_id==32:
                    x1 = bbox[:2][0]
                    y1 = bbox[:2][1]
                    x2 = bbox[2:][0]
                    y2 = bbox[2:][1]                
                    if ((x1>=337 and x2<=343) and (y1>=96 and y2<=104)) or score <0.3:
                        #false
                        continue
                    else:
                        ball+=1
                    #print(bbox[:2][0], bbox[2:])
                cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                cv2.putText(draw,
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, [225, 255, 255],
                            thickness=1)

        return instance, draw, ball, people
        #return instance, None, ball, people'''