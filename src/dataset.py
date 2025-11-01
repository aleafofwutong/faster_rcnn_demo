import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import csv
from collections import defaultdict

class CSVDataset(Dataset):
    """
    简单 CSV 数据集
    CSV 每行: filename, xmin, ymin, xmax, ymax, label
    label: int (从1开始)
    """

    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        # 构建索引： filename -> list of boxes/labels
        self._index = defaultdict(list)
        self._filenames = []
        with open(annotations_file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                filename = row[0].strip()
                xmin, ymin, xmax, ymax = map(float, row[1:5])
                label = int(row[5])
                self._index[filename].append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "label": label
                })
        self._filenames = sorted(list(self._index.keys()))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        filename = self._filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        img = Image.open(img_path).convert("RGB")
        ann = self._index[filename]

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for a in ann:
            x1, y1, x2, y2 = a["bbox"]
            boxes.append([x1, y1, x2, y2])
            labels.append(a["label"])
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
    
if __name__=="__main__":
    dataset=CSVDataset("/data/a_DL_beginner/datasets/images/train","/data/code3/annotations.csv")
    def collate_fn(batch):
        return tuple(zip(*batch))
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    data_loader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4,collate_fn=collate_fn)
    pbar=tqdm(data_loader,desc=f"hahaha")
    img_set=[]
    tgs_set=[]
    for imgs,tgs in pbar:
        # print(type(imgs[0]))
        img_set.append(imgs)
        tgs_set.append(tgs)
        # print(type(tgs[0]))
        # successful done
    img_set[0][0].show()
    print(tgs_set[0][0]['boxes'])

