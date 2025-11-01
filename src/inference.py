import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import argparse
from transforms import ToTensor
import os

def get_model(num_classes, ckpt_path=None, device=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(sd)
    if device:
        model.to(device)
    return model

def predict_and_visualize(model, image_path, device, score_thresh=0.5, out_path=None):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    to_tensor = ToTensor()
    img_t, _ = to_tensor(img, {})
    img_t = img_t.to(device)
    with torch.no_grad():
        outputs = model([img_t])
    out = outputs[0]
    boxes = out["boxes"].cpu()
    scores = out["scores"].cpu()
    labels = out["labels"].cpu()

    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1+3, y1+3), f"{int(label.item())}:{score:.2f}", fill="yellow")
    if out_path:
        img.save(out_path)
        print("Saved visualization to", out_path)
    else:
        img.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(args.num_classes, args.ckpt, device)
    predict_and_visualize(model, args.image, device, score_thresh=args.score_thresh, out_path=args.out)