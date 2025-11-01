"""
直接支持 YOLO txt 的 Dataset（作为替代 dataset.py）
每张图片有一个同名 txt（例如 img1.jpg 对应 img1.txt）
txt 每行： class x_center y_center width height （归一化 0..1）

返回 target 与之前示例相同（boxes, labels, area, iscrowd, image_id）
注意：labels 从 0-based 转为 1-based（0 作为背景）
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir=None, img_exts=("jpg","jpeg","png"), transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir if labels_dir is not None else images_dir
        self.transforms = transforms
        self.img_paths = []
        for ext in img_exts:
            self.img_paths += glob.glob(os.path.join(images_dir, f"*.{ext}"))
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def _read_yolo_txt(self, txt_path, img_w, img_h):
        boxes = []
        labels = []
        if not os.path.exists(txt_path):
            return boxes, labels
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w/2.0) * img_w
                y1 = (yc - h/2.0) * img_h
                x2 = (xc + w/2.0) * img_w
                y2 = (yc + h/2.0) * img_h
                # clamp
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w, x2); y2 = min(img_h, y2)
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # to 1-based
        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        name_wo_ext = os.path.splitext(filename)[0]
        txt_path = os.path.join(self.labels_dir, name_wo_ext + ".txt")

        boxes, labels = self._read_yolo_txt(txt_path, img_w, img_h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.nelement() else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target