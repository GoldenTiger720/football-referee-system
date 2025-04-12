from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import time
import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
from ultralytics import YOLO
import threading

lock = threading.Lock()
device = torch.device(0)
Engine = TRTModule("yolov8m_4.engine", device)
Engine_sin = TRTModule("yolov8l.engine", device)

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed


def my_detect2(img, model):
    results = model([img,img], classes=[32], conf=0.1, device=0, verbose=False, half=True, show_boxes=False)

def my_detect(images, W, H, ppos=0):
        global Engine, device
        # Assuming 'images' is a list of 3 BGR images
        #print("LEN:", len(images))

        processed_tensors = []
        for bgr in images:
            tensor = create_tensor_new(bgr, device,W,H)
            '''bgr = det_obj.frame
            # Preprocess each image
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.as_tensor(dwdh * 2, dtype=torch.float16, device=self.device)
            tensor = torch.as_tensor(tensor, device=self.device)'''
            # Append processed tensor
#            print("Add tensor")
            processed_tensors.append(tensor.unsqueeze(0))  # Add batch dimension

        # Stack tensors to form a batch
        batch_tensor = torch.cat(processed_tensors)          
        '''processed_tensors = []
        for bgr in images:
            # Preprocess each image
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.as_tensor(dwdh * 2, dtype=torch.float16, device=device)
            tensor = torch.as_tensor(tensor, device=device)
            # Append processed tensor
            processed_tensors.append(tensor.unsqueeze(0))  # Add batch dimension

        # Stack tensors to form a batch
        batch_tensor = torch.cat(processed_tensors)'''

        # Inference
        data = Engine(batch_tensor)

        p=0
        for bgr in images:
            bboxes, scores, labels = det_postprocess(data)
            p+=1

        return

        draw = images[ppos]

        if bboxes.numel() == 0:
            # if no bounding box
            print(f'no object!')
            return
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        cv2.imshow('result', draw)
        cv2.waitKey(0)

def my_detect_single(bgr, Engine, device, W, H, display=False):
        # Assuming 'images' is a list of 3 BGR images
        # Preprocess each image
        draw = bgr
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.as_tensor(dwdh * 2, dtype=torch.float16, device=device)
        tensor = torch.as_tensor(tensor, device=device)
        # Append processed tensor

        # Inference
        data = Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)

        if display==False:
             return
        
        if bboxes.numel() == 0:
            # if no bounding box
            print(f'no object!')
            return
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        cv2.imshow('result', draw)
        cv2.waitKey(0)

def create_tensor_new(frame, device, w, h):
    global lock
    with lock:        
        bgr, ratio, dwdh = letterbox(frame, (w, h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.as_tensor(dwdh * 2, dtype=torch.float16, device=device)
        #dwdh = torch.as_tensor(dwdh * 2, device=device)
        tensor = torch.as_tensor(tensor, device=device)

        return tensor  

def parallel_detect(frames, W, H, identifier):
    # Assuming `my_detect` is your function to process a set of frames.
    # The `identifier` argument is just to demonstrate how you might differentiate
    # the calls if necessary (for logging or other purposes).
    my_detect(frames, W, H, identifier)

def main() -> None:

    
    #yolo_model=YOLO('C:/Forex/yolo/yolov8m_4.engine')
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])



    '''image = cv2.imread("gl.png")
    my_detect_single(image, Engine_sin, device, W, H, True)

    overall_start_time =time.perf_counter() 
    my_detect_single(image, Engine_sin, device, W, H)
    overall_elapsed = (time.perf_counter() - overall_start_time) * 1000
    print(overall_elapsed)    
    return'''

    cap = cv2.VideoCapture("c:/develop/cam3.mp4", cv2.CAP_FFMPEG)
    frames = []
    for i in range(480):
        #print("Add")
        success, frame = cap.read()
        frames.append(frame)
    my_detect([frames[0],frames[1],frames[3],frames[4]], W, H, 0)
    my_detect_single(frames[0], Engine_sin, device, W, H)

    overall_start_time =time.perf_counter() 
    takes = 0
    size = 4
    frms=[]
    executor  = ThreadPoolExecutor(max_workers=2)
    print("Start")
    future1=None
    future2=None
    future3=None
    future4=None    
    for img in frames:
         height, width, channels = img.shape
         my_detect_single(img,Engine_sin, device, W, H, False)
         continue
         #print(f'Width: {width}px')
         #print(f'Height: {height}px')          
         #img = cv2.resize(img, (640, 384), interpolation=cv2.INTER_LINEAR)
         frms.append(img)
         if len(frms)==size:

            #first_half = frms[:size]  # This will contain the first 4 images
            #second_half = frms[size:]  # This will contain the second set of 4 images
            #print("Chck")
            while (future1 is not None and future2 is not None  and future3 is not None and future4 is not None):
                if future1 is not None and future1.done():
                    result1 = future1.result(timeout=0)
                    future1 = None

                if future2 is not None and future2.done():
                    result2 = future2.result(timeout=0)
                    future2 = None

                if future3 is not None and future3.done():
                    result3 = future3.result(timeout=0)
                    future3 = None

                if future4 is not None and future4.done():
                    result4 = future4.result(timeout=0)
                    future4 = None


                time.sleep(0.001)

            # Schedule the execution of `my_detect` for each set of frames
            if future1 is None:
                #print("Exec 1")
                future1 = executor.submit(parallel_detect, frms, W, H,1)
            elif future2 is None:
                #print("Exec 2")
                future2 = executor.submit(parallel_detect, frms, W, H,1)
            elif future3 is None:
                #print("Exec 2")
                future3 = executor.submit(parallel_detect, frms, W, H,1)
            elif future4 is None:
                #print("Exec 2")
                future4 = executor.submit(parallel_detect, frms, W, H,1)

            #future2 = executor.submit(parallel_detect, second_half, Engine, device, W, H, 2)

            # Wait for both functions to complete (this is optional and depends on
            # whether you need to process results further)

            

            #my_detectSin(frames[0], Engine, device, W, H)
            #my_detect(first_half, W, H, 0)
            #my_detect(frms, Engine, device, W, H, 0)
            takes+=1
            #print("takes:",takes)
            #my_detect_single(img, Engine_sin, device, W, H)
            #for image in frames:         
            #    my_detect([image,image,image,image], Engine, device, W, H)
                #my_detect2(image, yolo_model)
            
            frms=[]
            #break
    print("End")
    overall_elapsed = (time.perf_counter() - overall_start_time) * 1000
    print(overall_elapsed)
    per_frame = overall_elapsed / len(frames)
    print(per_frame)
    fps = 1000 / per_frame
    print("FPS:", fps)
    

if __name__ == '__main__':
    main()