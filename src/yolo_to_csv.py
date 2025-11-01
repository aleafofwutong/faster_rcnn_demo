"""
YOLO -> CSV 转换脚本
YOLO 标签格式（每个图片对应同名 txt）：
class x_center y_center width height   （数值为归一化在 0..1）

输出 CSV 每行：
image_filename, xmin, ymin, xmax, ymax, label

说明：
- 将 label 从 YOLO 的 0-based 转成 1-based（因为 Faster R-CNN 示例中 0 留作背景）
- 支持可选 class_map.json（将原始类 id 映射为新 id），若不提供则 +1 处理
"""
import os
import argparse
from PIL import Image
import csv
import json

def yolo_box_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w/2.0) * img_w
    y1 = (yc - h/2.0) * img_h
    x2 = (xc + w/2.0) * img_w
    y2 = (yc + h/2.0) * img_h
    # clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2

def convert(images_dir, labels_dir, out_csv, img_exts=("jpg","jpeg","png"), class_map_path=None):
    # load optional class map
    class_map = None
    if class_map_path:
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_map = json.load(f)  # expects dict of str(id)->int(new_id)
    rows = []
    for filename in os.listdir(images_dir):
        if filename.split(".")[-1].lower() not in img_exts:
            continue
        name_wo_ext = os.path.splitext(filename)[0]
        img_path = os.path.join(images_dir, filename)
        txt_path = os.path.join(labels_dir, name_wo_ext + ".txt")
        if not os.path.exists(txt_path):
            # 没有标注则跳过（或可选择产生空行）
            continue
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                xc, yc, w, h = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_box_to_xyxy(xc, yc, w, h, img_w, img_h)
                if class_map:
                    # class_map 的 key 可能是字符串或数字
                    new_label = class_map.get(str(cls), class_map.get(int(cls), None))
                    if new_label is None:
                        raise ValueError(f"Class {cls} not found in class_map")
                    label = int(new_label)
                else:
                    # YOLO default 0-based -> 转成 1-based
                    label = int(cls) + 1
                rows.append([filename, int(x1), int(y1), int(x2), int(y2), label])
    # write CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)
    print(f"Converted {len(rows)} boxes into {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="images directory")
    p.add_argument("--labels", required=True, help="YOLO labels directory (txt files)")
    p.add_argument("--out", default="annotations.csv", help="output CSV path")
    p.add_argument("--class-map", default=None, help="optional JSON path mapping original class -> new label (e.g. {\"0\":1, \"1\":2})")
    args = p.parse_args()
    convert(args.images, args.labels, args.out, class_map_path=args.class_map)