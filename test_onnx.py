import onnxruntime as ort
import os
import numpy as np


providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 1024 * 1024 * 1024,  # 1GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': '/home/maojingwei/project/sribd_attendance/trt_cache',
    }),
#    ('CUDAExecutionProvider', {
#        'device_id': 0,
#        'arena_extend_strategy': 'kNextPowerOfTwo',
#        'gpu_mem_limit': 1024 * 1024 * 1024,  # 1GB
#        'cudnn_conv_algo_search': 'EXHAUSTIVE',
#        'do_copy_in_default_stream': True,
#    }),
#    'CPUExecutionProvider',
]

config = {
    'face_detection': {
        'onnx_path': './sribd_face/models/face_detection/yolov5face-s-640x640.onnx',
        'providers': providers,
        'input_h_w': (640, 640),  # 模型输入大小
        'score_threshold': 0.7,  # 模型输出去掉score过小的bbox
        'nms_iou_threshold': 0.213,  # nms时iou阈值
    },
    'face_feature': {
        'onnx_path': './sribd_face/models/face_feature/p2_r50_cosface_simp.onnx',
        'providers': providers,
        'input_h_w': (112, 96),
    },
    "obj_detection":{
        'onnx_path': "/home/maojingwei/project/sribd_attendance/sribd_face/models/human_detection/yolov4.onnx",
        'providers': providers,
        'input_h_w': (416, 416),
        'score_threshold': 0.7,  # 模型输出去掉score过小的bbox
        'nms_iou_threshold': 0.213,  # nms时iou阈值
    },
    "body_feature": {
        "onnx_backbone_path": "/home/maojingwei/project/centroids-reid/tmp_backbone.onnx",
        "onnx_bn_path": "/home/maojingwei/project/centroids-reid/tmp_bn.onnx",
        "providers": providers
    },
    'face_database': {
        'dir_path': './face_database',
        'cos_score_threshold': 0.25,
    },
    "body_database":{
        "names":[],
        "threshold":-100
    },
    'door_roi_xyxy': [250, 0, 920, 500],
    'face_h_w': [112, 96],
}

body_database_dir = "/home/maojingwei/project/resources/sribd_attendance_data/body_labeled/features"
tmp_ls = list()
for ele in os.listdir(body_database_dir):
    config["body_database"]["names"].append(ele.replace(".npy","")) 
    tmp_feat = np.load(os.path.join(body_database_dir, ele))
    tmp_ls.append(tmp_feat)
config["body_database"]["features"] = np.concatenate(tmp_ls, axis=0)


body_feat_config = config["body_feature"]

if 'backbone_sess' not in body_feat_config:
    body_feat_config['backbone_sess'] = ort.InferenceSession(body_feat_config['onnx_backbone_path'], providers=body_feat_config['providers'])
    body_feat_config['input_h_w'] = (256, 128)
if 'bn_sess' not in body_feat_config:
    body_feat_config['bn_sess'] = ort.InferenceSession(body_feat_config['onnx_bn_path'], providers=body_feat_config['providers'])

x = np.random.rand(1,3,256,128)
_, outputs_backbone = body_feat_config['backbone_sess'].run(None, {'nchw_rgb': x.astype(np.float32)})
outputs_bn = body_feat_config['bn_sess'].run(None, {'nd': outputs_backbone})
print(outputs_bn[0].shape)


