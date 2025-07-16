
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    model.load('yolov8s.pt') # loading pretrain weights # 从 YAML加载 然后再加载权重
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                close_mosaic=10,
                workers=0,
                device='',
                optimizer='SGD', # using SGD
                #resume='True', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                #project='runs/train',
                name='exp',
                )