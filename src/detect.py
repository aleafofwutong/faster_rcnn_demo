import cv2
import torch
import torchvision
from PIL import Image
import numpy as np

# 配置参数
MODEL_PATH = '/data/checkpoints/fasterrcnn_epoch10.pth'      # 你的模型权重文件路径
NUM_CLASSES = 2                    # 你的类别数, 包含背景类
THRESHOLD = 0.5                    # 检测置信度阈值

# 加载模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR转RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转成Tensor
    img = Image.fromarray(img_rgb)
    img_tensor = torchvision.transforms.ToTensor()(img)

    # 检测
    with torch.no_grad():
        outputs = model([img_tensor])

    # 解析结果
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    # 画框
    for box, score, label in zip(boxes, scores, labels):
        if score > THRESHOLD:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}:{score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow('FasterRCNN Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()