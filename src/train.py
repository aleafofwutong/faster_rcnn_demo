import torch
import torchvision
from torch.utils.data import DataLoader
import argparse
from dataset import CSVDataset
from dataset_yolo import YOLODataset
from transforms import Compose, ToTensor, RandomHorizontalFlip
import os
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    # 加载预训练的 Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    # dataset = YOLODataset(args.images, args.data, transforms=transform)
    dataset=CSVDataset(f"{args.data}",f"{args.images}",transforms=transform)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = get_model(args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_val = losses.item()
            epoch_loss += loss_val
            pbar.set_postfix(loss=loss_val)

        lr_scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

        # 保存权重
        ckpt_path = os.path.join(args.output_dir, f"fasterrcnn_epoch{epoch+1}.pth")
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint to", ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./datasets/images/train", help="annotations.csv")
    parser.add_argument("--images", default="./annotations.csv", help="images directory")
    parser.add_argument("--output-dir", default="checkpoints", help="output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-classes", type=int, default=2, help="包含背景的类别数，例如 3 表示 2 个前景类 + 背景")
    args = parser.parse_args()
    train(args)