import os
from ultralytics import YOLO

def predict():
    # 載入最佳模型權重
    model_path = 'runs/train/namr_debris_exp/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到權重檔 {model_path}，將改用預訓練模型進行示範推論。")
        model = YOLO('yolo11x.pt')
    else:
        model = YOLO(model_path)

    # 對測試影像進行推論預測
    results = model.predict(
        source='data/images/val',  # 亦可指向測試集目錄
        save=True,
        conf=0.25,
        project='runs/predict',
        name='namr_debris_preds'
    )
    print("✅ 推論完成，結果已儲存至 runs/predict/namr_debris_preds")

if __name__ == '__main__':
    predict()
