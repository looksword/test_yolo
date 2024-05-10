import cv2
import torch
from PIL import Image
import numpy as np

# 加载 YOLOv5 模型
# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True # http 403报错
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# print(model.names)

# 设置模型参数
model.conf = 0.5  # 置信度阈值
model.iou = 0.45  # NMS IOU 阈值

# 读取图片
img = Image.open('data/469.jpg')

# 进行车辆检测
results = model(img)

# 绘制检测结果
img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
for *box, conf, cls in results.xyxy[0]:
    if model.names[int(cls)] != "car":
        continue
    # print(box,conf,model.names[int(cls)])
    x1, y1, x2, y2 = [int(i) for i in box]
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{model.names[int(cls)]} {conf:.2f}"
    cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

# 调整窗口大小以显示整张图片
height, width, _ = img_cv2.shape
cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vehicle Detection', width, height)

# 显示结果
cv2.imshow('Vehicle Detection', img_cv2)
cv2.waitKey(0)
cv2.imwrite('data/detected_image.jpg', img_cv2)
print('Detected image saved as detected_image.jpg')
cv2.destroyAllWindows()
