import torch
import numpy as np
from torch.utils.data import DataLoader
from model import model
from dataset import ImagesDataset
from eval import pixel_accuracy, iou, dice, precision_recall_f1

@torch.no_grad()
def test(model, loader, device):
    model.eval()

    pa_list, iou_list, dice_list = [], [], []
    p_list, r_list, f1_list = [], [], []

    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        out = torch.sigmoid(model(img))

        pa_list.append(pixel_accuracy(out, mask).item())
        iou_list.append(iou(out, mask).item())
        dice_list.append(dice(out, mask).item())

        p, r, f1 = precision_recall_f1(out, mask)
        p_list.append(p.item())
        r_list.append(r.item())
        f1_list.append(f1.item())

    print("Test Results")
    print(f"PA   : {np.mean(pa_list):.4f}")
    print(f"IoU  : {np.mean(iou_list):.4f}")
    print(f"Dice : {np.mean(dice_list):.4f}")
    print(f"Prec : {np.mean(p_list):.4f}")
    print(f"Rec  : {np.mean(r_list):.4f}")
    print(f"F1   : {np.mean(f1_list):.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: test dataset paths
    test_imgs = ["path/to/test_img.png"]
    test_masks = ["path/to/test_mask.png"]

    test_loader = DataLoader(
        ImagesDataset(test_imgs, test_masks),
        batch_size=8
    )

    net = model(in_channels=3, num_classes=1).to(device)
    net.load_state_dict(torch.load("best_model.pth", map_location=device))

    test(net, test_loader, device)
