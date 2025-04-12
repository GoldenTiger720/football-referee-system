from models import TRTModule  # isort:skip
import numpy as np
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list
import cv2
import torch
import threading

lock = threading.Lock()

def get_image_from_gpu(tensor):
    
    '''tensor = tensor.to('cpu').detach() 
    image_np = tensor.squeeze().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)'''

    #print(tensor.shape)

    #tensor = tensor.to('cpu').detach() 
    tensor = tensor.squeeze(0)  # This changes the shape from [1, 3, 640, 384] to [3, 640, 384]

    # Step 2: Convert the tensor to a numpy array and transpose it
    # Now, we transpose it from PyTorch's [C, H, W] format to numpy's [H, W, C] format
    image_np = tensor.cpu().numpy().transpose(1, 2, 0)  # This changes the shape to [640, 384, 3]

    # If the tensor was normalized to a range [0, 1], convert it back to [0, 255] and to uint8
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    # Step 3: Convert from RGB to BGR
    frame= cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return frame

def create_tensor(frame, device, w, h):
    global lock
    with lock:
        try:
                
            #print("add to gpu ->", w, h)
            #bgr, ratio, dwdh = letterbox(frame, (640, 384))
            bgr, ratio, dwdh = letterbox(frame, (w, h))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            #dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float8_e4m3fn, device=device)
            tensor = torch.asarray(tensor, device=device)

            return tensor
        except Exception as err:
            print("CUDA ERROR ", err)
    return None

def create_device(device_id):
    return torch.device(f"cuda:{device_id}")
     
def initialize_engine(device, engine_id, create_goal_engine=False):
    engine = None
    if (create_goal_engine==True):
        engine = TRTModule("yolov8m_320.engine", device)
    else:
        if engine_id=="1280":
            print("LOAD ENGINE 1280")
            engine = TRTModule("best_downloaded_0429_2300.engine", device)

        else:
            print("LOAD ENGINE 960")
            engine = TRTModule("best_downloaded_0429_960.engine", device)

    H, W = engine.inp_info[0].shape[-2:]
    print(f"Engine loaded: Actual input shape {H}x{W}")
    engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    return engine, W, H

def create_engine(device, engines, engine_id, create_goal_engine=False):
    engine, W, H = initialize_engine(device, engine_id, create_goal_engine)  # For GPU 0
    print(engine, W, H)
    engines.append(engine)

    return W, H

'''def create_engine_new(device,create_goal_engine=False):
    engine, W, H = initialize_engine(device, create_goal_engine)  # For GPU 0

    return engine, W, H'''
