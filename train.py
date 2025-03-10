import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\jianzhi\yolov8-dyhead1.yaml')
    model.load(r'E:\distall\runs\train\inner2\weights\best.pt') # loading pretrain weights
    model.train(data=r'E:\jianzhi\dataset\data.yaml',
                cache=False,
                imgsz=640,
                epochs=2,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )