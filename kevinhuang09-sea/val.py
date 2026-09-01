import os
from ultralytics import YOLO

def validate():
    # 載入訓練好的權重檔
    model_path = 'runs/train/namr_debris_exp/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到權重檔 {model_path}，請先執行 train.py 或指定正確權重路徑。")
        return

    model = YOLO(model_path)

    # 進行驗證評估 (計算 mAP@0.5)
    metrics = model.val(
        data='dataset.yaml',
        split='val',
        project='runs/val',
        name='namr_debris_eval'
    )

    print(f"✅ Validation mAP@0.5: {metrics.box.map50}")

if __name__ == '__main__':
    validate()
