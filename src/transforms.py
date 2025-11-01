import random
import torchvision.transforms.functional as F
from PIL import Image
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # image: PIL -> Tensor CxHxW, torchvision expects Image list
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            w, h = image.shape[2], image.shape[1]  # after to_tensor: C,H,W -> careful
            # Note: if this is called before ToTensor, you must get size from PIL
            # Here we assume ToTensor may be after; to be safe, handle both:
            if isinstance(image, torch.Tensor):
                _, H, W = image.shape
            else:
                W, H = image.size
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target