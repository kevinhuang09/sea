import os
from ultralytics import YOLO

def train():
    # 載入預訓練模型 (可選用 yolo11x.pt, yolov8x.pt 等)
    model = YOLO('yolo11x.pt')

    # 開始訓練模型
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='namr_debris_exp',
        project='runs/train',
        device=0
    )

if __name__ == '__main__':
    train()
