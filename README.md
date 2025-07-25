Introduction

This is the implementation code for the paper "LGPN: A Lightweight Algorithm for Enhanced Dental Lesion Detection in Panoramic X-rays."
The model is built based on YOLOv8s, using the pretrained configuration file yolov8s.pt.


Software Platform

CUDA 11.3

PyTorch 1.12.0

Python 3.8


Training and Testing

You can use PyCharm as the development environment and run the code through Python command-line execution.

Training

To perform training, create a Python file named train.py and input the following code:

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s.yaml')
    
    model.load('yolov8s.pt')  # loading pre-trained weights
    
    model.train(
    
        data='your_data.yaml',
        
        cache=False,
        
        imgsz=640,
        
        epochs=,          # specify number of epochs
        
        batch=,           # specify batch size
        
        close_mosaic=,    # optionally close mosaic augmentation
        
        workers=0,
        
        device='',        # specify CUDA device (e.g. '0')
        
        optimizer='SGD',  # using SGD optimizer
        
        # resume='True',   # path to last.pt (if resuming training)
        
        # amp=False,       # disable AMP (automatic mixed precision)
        
        # fraction=0.2,
        
        # project='runs/train',
        
        name='exp',
        
    )

Some of the parameters (like epochs, batch, device, etc.) should be specified according to your specific needs.


Testing

The testing procedure follows the same logic: create a test script similarly and modify the mode or use .predict() or .val() methods as needed.
