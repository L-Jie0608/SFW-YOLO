import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\jianzhi\runs\prune\lamp-exp3-finetune\weights\best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='Pd',
              )