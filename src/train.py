import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from model import UNet


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── config ────────────────────────────────────────────────────────────────────
EXPERIMENT    = "augmentation_restored"
OUTPUT_DIR    = f"/u/aayyah3/oral-biomarker-segmentation/outputs/{EXPERIMENT}"
DATA_ROOT     = "/u/aayyah3/oral-biomarker-segmentation/data/Oral_Final_Biomarkers"
NUM_CLASSES   = 3
IN_CHANNELS   = 1
BATCH_SIZE    = 16
NUM_EPOCHS    = 50
LR            = 1e-4
CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 6.0], dtype=torch.float32)
IGNORE_IDX    = 255

os.makedirs(OUTPUT_DIR, exist_ok=True)


def dice_score(preds, targets, num_classes=3, ignore_index=255):
    scores = []
    for c in range(num_classes):
        pred_c   = (preds == c)
        target_c = (targets == c) & (targets != ignore_index)
        intersection = (pred_c & target_c).sum().float()
        union        = pred_c.sum().float() + target_c.sum().float()
        if union == 0:
            scores.append(float('nan'))
        else:
            scores.append((2.0 * intersection / union).item())
    return scores


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_dice   = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss   = criterion(logits, masks)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            batch_dice = dice_score(preds.cpu(), masks.cpu())
            all_dice.append(batch_dice)
    avg_loss  = total_loss / len(loader)
    all_dice  = np.array(all_dice, dtype=float)
    mean_dice = np.nanmean(all_dice, axis=0)
    return avg_loss, mean_dice


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE)

    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device),
        ignore_index=IGNORE_IDX
    )

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

    best_val_loss  = float('inf')
    best_mean_dice = 0.0
    log_lines      = ["epoch,train_loss,val_loss,dice_cls0,dice_cls1,dice_cls2,mean_dice"]

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Dice C0':>8} | {'Dice C1':>8} | {'Dice C2':>8} | {'Mean Dice':>10}")
    print("-" * 75)

    early_stopper = EarlyStopping(patience=20)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss          = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, mean_dice = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        elapsed       = time.time() - t0
        mean_dice_avg = np.nanmean(mean_dice)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{mean_dice[0]:>8.4f} | {mean_dice[1]:>8.4f} | {mean_dice[2]:>8.4f} | "
              f"{mean_dice_avg:>10.4f}   ({elapsed:.1f}s)")

        log_lines.append(
            f"{epoch},{train_loss:.4f},{val_loss:.4f},"
            f"{mean_dice[0]:.4f},{mean_dice[1]:.4f},{mean_dice[2]:.4f},{mean_dice_avg:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"        ↳ saved best model (val loss {best_val_loss:.4f})")

        if mean_dice_avg > best_mean_dice:
            best_mean_dice = mean_dice_avg
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_dice_model.pth"))

        if early_stopper(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))

    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nTraining complete. Log saved to {log_path}")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best mean Dice: {best_mean_dice:.4f}")


if __name__ == "__main__":
    main()