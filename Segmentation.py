from ultralytics import YOLO
import cv2

model = YOLO("runs/train/weights/best.pt")

img = cv2.imread("images/path_to_image.jpg")

results = model(img, task = "segment", save = True, save_txt = True)

# オブジェクトの種類を調べる
for e in results[0].boxes.cls.cpu():
  print(e, model.names[int(e)])
