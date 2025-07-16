import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/exp21/weights/best.pt')
    model.val(data='dataset/yachi.yaml',
              split='test',
              imgsz=640,
              batch=1,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )