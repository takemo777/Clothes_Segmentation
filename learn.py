from ultralytics import YOLO

# YOLOv8モデルのロード
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')

# トレーニングの実行
results = model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)
