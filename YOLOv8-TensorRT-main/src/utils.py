import proto_messages.frame_pb2 as frame_pb2
import proto_messages.rpc_pb2 as rpc_pb2
import proto_messages.players_pb2 as players_pb2
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
from ecal.core.subscriber import ProtoSubscriber
from enum import Enum
import time

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

#global_executor = ThreadPoolExecutor(max_workers=2)

rpc_req_id =0



class DetectionType(Enum):
    NORMAL = 1
    GOAL = 2

def create_protobuf(frame_id, camera_id, frame, timestamp, w, h):
    protobuf_message = frame_pb2.FrameData()
    protobuf_message.frame_id=frame_id
    protobuf_message.camera_id = camera_id
    protobuf_message.width = w
    protobuf_message.height = h
    protobuf_message.unix_timestamp = timestamp
    protobuf_message.frame = frame.tobytes()
    return protobuf_message

'''message FootPlayer {
  int32 cycle = 1;
  int32 position = 2;
  int32 x1 = 3;
  int32 x2 = 4;
  int32 y1 = 5;
  int32 y2 = 6;
  bytes img = 7; // Image data serialized as bytes
}
'''
def create_player_protobuf_msg(cycle, position, x1, x2, y1, y2, x_2d, y_2d, img):
    height, width = img.shape[:2]
    protobuf_message = players_pb2.FootPlayer()
    protobuf_message.cycle=cycle
    protobuf_message.position=position
    protobuf_message.x1=x1
    protobuf_message.x2=x2
    protobuf_message.y1=y1
    protobuf_message.y2=y2
    protobuf_message.x_2d=x_2d
    protobuf_message.y_2d=y_2d
    protobuf_message.img_w = width
    protobuf_message.img_h=height
    protobuf_message.img=img.tobytes()
    return protobuf_message

def restart_session_rpc():
    global rpc_req_id
    protobuf_message = rpc_pb2.RPCData()
    rpc_req_id+=1
    protobuf_message.request_id=rpc_req_id
    protobuf_message.procedure = "restart_stream()"
    protobuf_message.params = ""
    return protobuf_message

def camera_selection(cycle, camera_id,score_cntr_l, score_cntr_r, match_time, json_ball_coords):
    global rpc_req_id
    protobuf_message = rpc_pb2.RPCData()
    rpc_req_id+=1
    protobuf_message.request_id=rpc_req_id
    protobuf_message.procedure = "camera_selection()"
    protobuf_message.params = f'{cycle}|{camera_id}|{score_cntr_l}|{score_cntr_r}|{match_time}|{json_ball_coords}'
    return protobuf_message