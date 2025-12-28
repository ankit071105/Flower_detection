import os
import torch
import torchvision
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from PIL import Image
import random

DATA_DIR = "../dataset/flower_photos/flower_photos"

def load_images():
    x = []
    for cls in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_path): 
            continue
        for img in os.listdir(class_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                x.append(os.path.join(class_path, img))
    return x

class FlowerDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # full image as a flower
        boxes = torch.tensor([[0, 0, w, h]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        img = F.to_tensor(img)
        return img, {"boxes": boxes, "labels": labels}

def main():
    print("Loading dataset...")
    images = load_images()
    random.shuffle(images)

    dataset = FlowerDataset(images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    print("Loading base model...")
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

    # modify head → 2 classes (background + flower)
    model.head.classification_head.num_classes = 2

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    print("Training started...")
    for epoch in range(10):
        for imgs, targets in loader:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "flower_ssd.pth")
    print("Training finished → flower_ssd.pth saved.")

if __name__ == "__main__":
    main()
