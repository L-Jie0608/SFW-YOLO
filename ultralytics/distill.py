import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'runs/prune/yolov8n-lamp-exp1-prune/weights/prune.pt',
        'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
        'imgsz': 640,
        'epochs': 250,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project':'runs/distill',
        'name':'yolov8n-lamp-BCKD-exp2',
        
        # distill
        'prune_model': True,
        'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
        'teacher_cfg': 'yolov8s-GhostHGNetV2-SlimNeck-ASF.yaml',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 2.0,
        
        'teacher_kd_layers': '14,18,21,24',
        'student_kd_layers': '14,18,21,24',
        'feature_loss_type': 'mgd',
        'feature_loss_ratio': 0.2
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()