import os
from pathlib import Path

# 專案根目錄名稱
PROJECT_NAME = "kevinhuang09-sea"

# 1. 定義要建立的資料夾清單
DIRECTORIES = [
    f"{PROJECT_NAME}/data/images/train",
    f"{PROJECT_NAME}/data/images/val",
    f"{PROJECT_NAME}/data/labels/train",
    f"{PROJECT_NAME}/data/labels/val",
    f"{PROJECT_NAME}/models",
    f"{PROJECT_NAME}/runs",
]

# 2. 定義各檔案的初始內容

# dataset.yaml
DATASET_YAML = """# YOLO 資料集設定檔 (NAMR 海廢辨識)
path: ./data
train: images/train
val: images/val

# 類別數量 (33 + 1 類)
nc: 34

# 類別名稱對照表
names:
  0: plastic_bottle
  1: plastic_bottle_cap
  2: non_PET_food_beverage_container
  3: non_food_plastic_bottle
  4: takeaway_beverage_cup
  5: straw
  6: disposable_tableware
  7: plastic_bag
  8: food_wrapper
  9: metal_can
  10: cigarette_butt
  11: lighter
  12: drink_carton
  13: glass_bottle
  14: fishing_net_rope
  15: fishing_buoy_float
  16: fishing_gear
  17: syringe_needle
  18: toothbrush
  19: other
  20: plastic_lid
  21: cup
  22: foam_buoy_float
  23: cigarette_pack
  24: textile
  25: net_like_item
  26: anthropogenic_fragment
  27: disposable_food_container
  28: soft_float
  29: foam_container
  30: non_PET_food_container
  31: non_food_plastic_container
  32: aluminum_packaging
  33: fish_trap_and_bait
"""

# train.py
TRAIN_PY = """import os
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
"""

# val.py
VAL_PY = """import os
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
"""

# predict.py
PREDICT_PY = """import os
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
"""

# README.md 占位提示
README_MD = """# 🌊 kevinhuang09-sea 海廢影像辨識專案

本專案為參加國家海洋研究院（NAMR）海廢影像辨識競賽之主要儲存庫。
請參閱完整說明或直接更新本檔。
"""

# 3. 檔案對照表
FILES = {
    f"{PROJECT_NAME}/dataset.yaml": DATASET_YAML,
    f"{PROJECT_NAME}/README.md": README_MD,
    f"{PROJECT_NAME}/train.py": TRAIN_PY,
    f"{PROJECT_NAME}/val.py": VAL_PY,
    f"{PROJECT_NAME}/predict.py": PREDICT_PY,
    f"{PROJECT_NAME}/models/__init__.py": "# 客製化模型模組目錄\n",
}

def create_structure():
    print(f"🚀 開始建立專案架構：{PROJECT_NAME} ...\n")
    
    # 建立所有資料夾
    for folder in DIRECTORIES:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 已建立資料夾: {folder}")

    # 建立所有檔案並寫入預設內容
    for filepath, content in FILES.items():
        path = Path(filepath)
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content.strip() + "\n")
            print(f"📄 已建立檔案: {filepath}")
        else:
            print(f"⚠️ 檔案已存在，跳過: {filepath}")

    print(f"\n🎉 專案架構建立完成！請使用 `cd {PROJECT_NAME}` 進入專案目錄。")

if __name__ == "__main__":
    create_structure()