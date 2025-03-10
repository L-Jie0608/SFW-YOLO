import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'E:\distall\runs\prune\lamp-exp3-prune\weights\prune.pt',
        'data': r'E:\distall\dataset\data.yaml',
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
        'teacher_weights': r'E:\distall\runs\train\inner2\weights\best.pt',
        'teacher_cfg': r'E:\distall\yolov8-dyhead1.yaml',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 2.0,
        
        'teacher_kd_layers': '18,21,24,27',
        'student_kd_layers': '18,21,24,27',
        'feature_loss_type': 'mgd',
        'feature_loss_ratio': 0.2
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()