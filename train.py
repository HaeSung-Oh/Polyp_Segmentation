import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from monai.losses import TverskyLoss
from model import model
from dataset import ImagesDataset

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_coef(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    inter = (pred & gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-6)

def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    losses = []

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = torch.sigmoid(model(img))
        loss = criterion(out, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

@torch.no_grad()
def validate_epoch(loader, model, criterion, device):
    model.eval()
    losses, dices = [], []

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = torch.sigmoid(model(img))
        losses.append(criterion(out, mask).item())
        dices.append(dice_coef(out.cpu().numpy(), mask.cpu().numpy()))

    return np.mean(losses), np.mean(dices)

if __name__ == "__main__":
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_imgs = ["path/to/train_img.png"]
    train_masks = ["path/to/train_mask.png"]
    val_imgs = ["path/to/val_img.png"]
    val_masks = ["path/to/val_mask.png"]

    train_loader = DataLoader(
        ImagesDataset(train_imgs, train_masks),
        batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        ImagesDataset(val_imgs, val_masks),
        batch_size=8
    )

    net = model(in_channels=3, num_classes=1).to(device)
    criterion = TverskyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    best_dice = 0

    for epoch in range(100):
        train_loss = train_epoch(train_loader, net, criterion, optimizer, device)
        val_loss, val_dice = validate_epoch(val_loader, net, criterion, device)
        scheduler.step()

        print(f"[{epoch+1:03d}] "
              f"Train {train_loss:.4f} | "
              f"Val {val_loss:.4f} | "
              f"Dice {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(net.state_dict(), "best_model.pth")
