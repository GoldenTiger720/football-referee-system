from ultralytics import YOLO
import concurrent.futures
from playerutils import *
import os
import cv2
import time
import sys

class PlayerAnalyzer:
    def __init__(self, seg_model, pos_model, save_folder="tracker_imgs_send2"):
        self.model_seg = YOLO(seg_model)
        self.model_pose = YOLO(pos_model)
        self.stitched_img = None
        self.debug = False
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)        

    def get_grid_coordinates(self, x, y, grid_cols, grid_rows, img_w, img_h):
        tile_w = img_w / grid_cols
        tile_h = img_h / grid_rows

        return int(x/tile_w),int(y/tile_h)

    def update_object(self, image_objects, tile_x, tile_y, segment=None, mask=None, keypoints=None, area=None, total_confidence=None, avrg_confidence = None):
        for img_obj in image_objects:
            if img_obj.grid_col==tile_x and img_obj.grid_row==tile_y:
                if segment is not None:
                    area_ratio = area / (img_obj.resized_width*img_obj.resized_height)*100

                    #keep only the largest area and ignore if too small
                    if (area_ratio>2 and area_ratio>img_obj.area_ratio):
                        img_obj.segment = segment
                        img_obj.area_ratio = area_ratio
                        img_obj.mask = mask
                if keypoints is not None:
                    if total_confidence is None or img_obj.total_confidence is None or img_obj.total_confidence<total_confidence:                        
                        img_obj.keypoints = keypoints
                        img_obj.total_confidence = total_confidence
                        img_obj.avrg_confidence = round(avrg_confidence,2)
                return img_obj
        return None

    def do_pose_estimation(self, debug, results_pose, image_objects, grid_cols, grid_rows, img_w, img_h):
        # print(type(results_pose[0]),'-------------------------')
        # print(results_pose[0],'++++++++++++++++++++')
        # print(results_pose[0].keypoints,'^^^^^^^^^^^^^^')
        process_time = ProcTimer(debug)
        kp_len = len(results_pose[0].keypoints)
        # print(kp_len,'++++++++8888888888')
        keypoints_collections = {}
        kpp2 = results_pose[0].cpu()
        # print(kpp2.keypoints,'000000000000000000000000000')
        # sys.exit()
        resized_img_height = image_objects[0].resized_height

        for i in range(kp_len):
            keypoint_array = kpp2.keypoints[i].xy
            confidence_array = kpp2.keypoints[i].conf

            avrg_x, avrg_y, total_confidence = 0, 0, 0
            confident_point_cntr, sample_size = 0, 0
            min_y, max_y = None, None
            scale_x = img_w / image_objects[0].resized_width
            scale_y = img_h / image_objects[0].resized_height
            keypoints_collections[i] = []
            for c, (x, y) in enumerate(keypoint_array[0]):
                if x > 0 and y > 0:
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y
                    avrg_x += x
                    avrg_y += y
                    sample_size += 1
                    min_y = y if min_y is None else min(min_y, y)
                    max_y = y if max_y is None else max(max_y, y)

                conf = float(confidence_array[0][c])
                if conf > 0.5:
                    confident_point_cntr += 1
                total_confidence += conf
                keypoints_collections[i].append([int(x), int(y), conf])
                # keypoints_collections[i].append([int(x_scaled), int(y_scaled), conf])

            size_ratio = (max_y - min_y) / resized_img_height if min_y is not None and max_y is not None else 0

            if confident_point_cntr > 8 and size_ratio > 0.65:
                avrg_x /= sample_size
                avrg_y /= sample_size
                avrg_confidence = total_confidence / len(keypoint_array[0])

                tile_x, tile_y = self.get_grid_coordinates(avrg_x, avrg_y, grid_cols, grid_rows, img_w, img_h)
                self.update_object(image_objects, tile_x, tile_y, keypoints=keypoints_collections[i],
                                total_confidence=total_confidence, avrg_confidence=avrg_confidence)

        process_time.stop("Pose Estimation")

    def do_segmentation(self,debug, results, image_objects, grid_cols, grid_rows, img_w, img_h):
        ##SEGMENTATION
        process_time = ProcTimer(debug)
        img = np.copy(results[0].orig_img)
        if results[0].masks is None:
            return
        for i in range(len(results[0].masks.xy)):
            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = results[0].masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
            area = cv2.contourArea(contour)
            if area>0:
                x, y = contour[0][0]
                
                tile_x, tile_y = self.get_grid_coordinates(x, y, grid_cols, grid_rows, img_w, img_h)
                
                cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)

                self.update_object(image_objects, tile_x, tile_y, segment=isolated, mask=b_mask, area=area)
        process_time.stop("Segmentation")

    def analyze(self, image_objects, debug=False, segment_enabled=True, verbosity=False, ai_confidence = 0.2, ai_iou=0.3, cycle=None, position=None, img_w=640, img_h=640):
        MAX_PER_IMAGE=6

        total = len(image_objects)
        subs = math.ceil(total/MAX_PER_IMAGE)
        sub_img_objects={}
        res_img_objects={}
        results={}
        for i in range(subs):
            sub_img_objects[i]=[]
            results[i]=None
            res_img_objects[i]=None
            for c in range(MAX_PER_IMAGE):
                pos = i*MAX_PER_IMAGE+c
                if pos<total:
                    sub_img_objects[i].append(image_objects[pos])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(subs):
                results[i] = executor.submit(self.do_analyze, sub_img_objects[i], debug, segment_enabled, verbosity, ai_confidence, ai_iou, cycle, position, img_w, img_h)

            for i in range(subs):
                res_img_objects[i] = results[i].result()
            
    def do_analyze(self, image_objects, debug=False, segment_enabled=True, verbosity=False, ai_confidence = 0.2, ai_iou=0.3, cycle=None, position=None, img_w=640, img_h=640):
        self.debug = debug
        for img in image_objects:
            img.reset()
        grid_cols, grid_rows, self.stitched_img = PlayerUtils.stitch_images(image_objects, debug=debug)
        cv2.imwrite(f"stich_images{time.time()}.jpg",self.stitched_img)
        if (grid_cols==0 or grid_cols is None):
            return False
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if segment_enabled:
                future_pose = executor.submit(self.model_pose, self.stitched_img, verbose = verbosity, conf = ai_confidence, iou=ai_iou)
                future_seg = executor.submit(self.model_seg, self.stitched_img, verbose = verbosity)

                results_seg = future_seg.result()
                #segmentation slightly faster then pose estimate so let's start doing it while waiting for the pose estimation results
                self.do_segmentation(debug, results_seg, image_objects, grid_cols, grid_rows, img_w, img_h)
                results_pose = future_pose.result()
                # self.do_pose_estimation(debug, results_pose, image_objects, grid_cols, grid_rows, img_w, img_h)
            else:
                results_pose = self.model_pose(self.stitched_img, verbose = verbosity, conf = ai_confidence, iou=ai_iou)
                # self.do_pose_estimation(debug, results_pose, image_objects, grid_cols, grid_rows, img_w, img_h)

        # results_pose = self.model_pose(self.stitched_img, verbose = verbosity, conf = ai_confidence, iou=ai_iou)
        self.do_pose_estimation(debug, results_pose, image_objects, grid_cols, grid_rows, img_w, img_h)
        
        process_time = ProcTimer(debug)
        ##RESULT POSTPROCESS
        segment_w = int(img_w / grid_cols)
        segment_h = int(img_h / grid_rows)
        temp_list = []
        for num,img_obj in enumerate(image_objects):
            # cv2.imwrite(f'segment_image_{num}.jpg',img_obj.segment)
            img_obj.tile_height = segment_h
            img_obj.tile_width = segment_w
            if segment_enabled==False:
                img_obj.segment = img_obj.resized

            if debug:
                print(f'Tile dimension: {segment_w}x{segment_h}')

            delta_x = int((segment_w - img_obj.resized_width)/2)
            delta_y = int((segment_h - img_obj.resized_height)/2)

            if img_obj.segment is None:
                img_obj.valid = False
            else:
                img_obj.valid = True
                if segment_enabled:
                    shape_w, shape_h = img_obj.segment.shape[:2]
                    if debug:
                        print(f'[{img_obj.filename}] segment size: {shape_w}x{shape_h}')
                    img_obj.segment = img_obj.segment[img_obj.grid_row*segment_h + delta_y:img_obj.grid_row*segment_h + delta_y + img_obj.resized_height,
                                                    img_obj.grid_col*segment_w + delta_x:img_obj.grid_col*segment_w + delta_x + img_obj.resized_width]
                    # cv2.imwrite(f'update_segment_image_{num}.jpg',img_obj.segment)
                if debug:
                    shape_w, shape_h = img_obj.segment.shape[:2]
                    print(f'[{img_obj.filename}] resized segment size: {shape_w}x{shape_h}')

                    img_obj.segment_w_points = img_obj.segment.copy()
                    img_obj.segment_w_poly = img_obj.segment.copy()


                if img_obj.mask is not None:
                    # cv2.imwrite(f'mask_image_{num}.jpg',img_obj.mask)
                    img_obj.mask = img_obj.mask[img_obj.grid_row*segment_h + delta_y:img_obj.grid_row*segment_h + delta_y + img_obj.resized_height,
                                                    img_obj.grid_col*segment_w+delta_x:img_obj.grid_col*segment_w + delta_x + img_obj.resized_width]
                    # cv2.imwrite(f'update_mask_image_{num}.jpg',img_obj.mask)

            if img_obj.keypoints is None:
                img_obj.valid = False
            else:
                for i in range(len(img_obj.keypoints)):
                    img_obj.keypoints[i][1] -= img_obj.grid_row*segment_h + delta_y
                    img_obj.keypoints[i][0] -= img_obj.grid_col*segment_w+delta_x

            if img_obj.valid:
                kps = [(int(x), int(y)) for (x, y, conf) in img_obj.keypoints]

                '''with concurrent.futures.ThreadPoolExecutor() as executor:
                    shirt_ext = executor.submit(self.get_shirt_color,kps, img_obj)
                    pants_ext = executor.submit(self.get_pants_color, img_obj.keypoints, kps, img_obj)
                    socks_ext = executor.submit(self.get_socks_color, img_obj.keypoints, kps, img_obj)

                    img_obj.features.shirt_color1, img_obj.features.shirt_color2 = shirt_ext.result()
                    img_obj.features.pants_color1, img_obj.features.pants_color2 = pants_ext.result()
                    img_obj.features.socks_color1, img_obj.features.socks_color2= socks_ext.result()'''

                img_obj.features.shirt_color1, img_obj.features.shirt_color2 = self.get_shirt_color(kps, img_obj)
                img_obj.features.pants_color1, img_obj.features.pants_color2 = self.get_pants_color(img_obj.keypoints, kps, img_obj)                
                #img_obj.features.socks_color1, img_obj.features.socks_color2 = self.get_socks_color(img_obj.keypoints, kps, img_obj)
                #_,_ = self.get_hair_color(img_obj.keypoints, kps, img_obj)
            # PlayerUtils.save_debug_image(self.save_folder, img_obj)
            temp_list.append(img_obj)
        if debug:
            if (cycle is None or position is None):
                cv2.imwrite(os.path.join(self.save_folder, "stiched.png"), self.stitched_img)
            else:
                cv2.imwrite(os.path.join(self.save_folder, f"C{cycle}P{position}_stiched.png"), self.stitched_img)
            # for img_obj in image_objects:
            for img_obj in temp_list:
                PlayerUtils.save_debug_image(self.save_folder, img_obj)
        process_time.stop("Color Extraction")

        return True

    def get_lower_two_third(self, points):
        points = [list(pt) for pt in points]

        for i in range(2, 4):
            y_difference = points[i][1] - points[i - 2][1]
            points[i][1] -= y_difference / 3  
            points[i][1] = int(points[i][1])
        
        points = [tuple(pt) for pt in points]
        return points

    def get_shirt_color(self, kps, img_obj):
        left_t_x,left_t_y = kps[BodyPoint.LEFT_SHOULDER]
        right_t_x, right_t_y = kps[BodyPoint.RIGHT_SHOULDER]
        left_h = kps[BodyPoint.LEFT_HIP][1] - left_t_y
        right_h = kps[BodyPoint.RIGHT_HIP][1]- right_t_y

        min_w = abs(right_t_x-left_t_x)
        
        #don't calculate if it is not wide enough
        #if min_w<15:
        #    return None, None
        widen=4
        left_t_y=left_t_y+left_h-int(left_h*1)
        right_t_y=right_t_y+right_h-int(right_h*1)

        if left_t_x>left_t_y:
            widen=-1*widen
        points =  np.array([(left_t_x-widen, left_t_y), (right_t_x+widen, right_t_y), kps[BodyPoint.RIGHT_HIP], kps[BodyPoint.LEFT_HIP]])
        return PlayerUtils.average_color_in_polygon(img_obj, points, self.debug)        

    def get_pants_color(self, keypoints, kps, img_obj):
        conf_l = keypoints[BodyPoint.LEFT_KNEE][2]
        conf_r = keypoints[BodyPoint.RIGHT_KNEE][2]
        
        t_x, t_y=kps[BodyPoint.LEFT_KNEE]
        b_x, b_y=kps[BodyPoint.LEFT_HIP]
        w=int((t_y-b_y)/6)
        if w<2:
            w=2
        points_l =  np.array(self.get_lower_two_third([(t_x-w, t_y),(t_x+w, t_y),(b_x+w, b_y),(b_x-w, b_y)]))

        t_x, t_y=kps[BodyPoint.RIGHT_KNEE]
        b_x, b_y=kps[BodyPoint.RIGHT_HIP]
        w=int((t_y-b_y)/6)
        if w<2:
            w=2
        points_r =  np.array(self.get_lower_two_third([(t_x-w, t_y),(t_x+w, t_y),(b_x+w, b_y),(b_x-w, b_y)]))

        #points =  np.array(self.get_lower_two_third([kps[BodyPoint.LEFT_KNEE], kps[BodyPoint.RIGHT_KNEE], kps[BodyPoint.RIGHT_HIP], kps[BodyPoint.LEFT_HIP]]))

        #cv2.fillPoly(img_obj.segment, pts=[points],color=(0,255,0))
        if abs(conf_l-conf_r)>0.3:
            if conf_l >conf_r:
                return PlayerUtils.average_color_in_polygon(img_obj, points_l, self.debug)
            else:
                return PlayerUtils.average_color_in_polygon(img_obj, points_r, self.debug)
        else:
            res_l_col1, res_l_col2=PlayerUtils.average_color_in_polygon(img_obj, points_l, self.debug)
            res_r_col1, res_r_col2=PlayerUtils.average_color_in_polygon(img_obj, points_r, self.debug)
            return self.get_final_colors(res_l_col1, res_l_col2, res_r_col1, res_r_col2)
  
    def get_socks_points(self, kps, knee, ankle):
        knee_x, knee_y=kps[knee]
        ankle_x, ankle_y = kps[ankle]

        #ankle_y+=5
        middle_x = int((knee_x+ankle_x)/2)
        middle_y = int((knee_y+ankle_y)/2)

        width = (middle_y-ankle_y)/2
        if width<3:
            width=3

        width=int(width)

        return np.array([(middle_x-width, middle_y), (middle_x+width, middle_y), (ankle_x+width, ankle_y), (ankle_x-width, ankle_y)])

    def get_final_colors(self, l1, l2, r1, r2):
        THRESHOLD=30
        if PlayerUtils.rgb_diff(l1, r1)<THRESHOLD:
            res=PlayerUtils.color_avrg(l1, r1)
            return res, res
        if PlayerUtils.rgb_diff(l1, r2)<THRESHOLD:
            res=PlayerUtils.color_avrg(l1, r2)
            return res, res
        if PlayerUtils.rgb_diff(l2, r1)<THRESHOLD:
            res=PlayerUtils.color_avrg(l2, r1)
            return res, res
        if PlayerUtils.rgb_diff(l2, r2)<THRESHOLD:
            res=PlayerUtils.color_avrg(l2, r2)
            return res, res
        
        return l1, r1

    def get_socks_color(self, keypoints, kps, img_obj):
        left_bottom = keypoints[BodyPoint.LEFT_ANKLE][1]
        right_bottom = keypoints[BodyPoint.RIGHT_ANKLE][1]
        
        if left_bottom==0 and right_bottom==0:
            return None, None

        points_l = self.get_socks_points(kps, BodyPoint.LEFT_KNEE, BodyPoint.LEFT_ANKLE)
        points_r = self.get_socks_points(kps, BodyPoint.RIGHT_KNEE, BodyPoint.RIGHT_ANKLE)
        #cv2.fillPoly(img_obj.segment, pts=[points],color=(0,255,0))
        res_l_col1, res_l_col2= PlayerUtils.average_color_in_polygon(img_obj, points_l, self.debug)
        res_r_col1, res_r_col2= PlayerUtils.average_color_in_polygon(img_obj, points_r, self.debug)
        return self.get_final_colors(res_l_col1, res_l_col2, res_r_col1, res_r_col2)

    '''def get_hair_color(self, keypoints, kps, img_obj):
        n_x, n_y, n_c=keypoints[BodyPoint.NOSE]

#        if n_c<=0:
        el_x, el_y, el_c = keypoints[BodyPoint.LEFT_EAR]
        er_x, er_y, er_c = keypoints[BodyPoint.RIGHT_EAR]
        if (el_x>0 and er_x>0):
            n_x = int((el_x+er_x)/2)
            n_y = int((el_y+er_y)/2)
            n_c = (el_c+er_c)/2

        sr_x, sr_y, sr_c=keypoints[BodyPoint.RIGHT_SHOULDER]
        sl_x, sl_y, sl_c=keypoints[BodyPoint.LEFT_SHOULDER]
        #n_x=int((sl_x+sr_x)/2)
        if n_c>0 and sl_c>0 and sr_c>0:
            radius =int(abs(n_y-(sr_y+sl_y)/2)*0.7)
            cv2.ellipse(img_obj.segment_w_poly, 
            (int(n_x), int(n_y)), 
            (int(0.8 * radius), int(radius)), 
            0, 0, 360, 
            (0, 255, 0), 
            1)

        return None, None'''