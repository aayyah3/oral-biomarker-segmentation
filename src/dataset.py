import os
import glob
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ALL_PATIENTS = [
    "RS_03948932", "RS_03948938", "RS_03948956", "RS_03948968",
    "RS_03948980", "RS_03948992", "RS_03949004", "RS_03949010",
    "RS_03949016", "RS_03949022", "RS_03949028", "RS_03949046",
    "RS_03949058", "RS_03949070", "RS_03949094",
    "RS_extra_HN-C-00044-7", "RS_extra_HN-C-00083-4",
    "RS_03959863", "RS_03959869", "RS_03959881"
]

TRAIN_PATIENTS = [
    "RS_03948932",
    "RS_03948938", "RS_03948968", "RS_03948992", "RS_03949004",
    "RS_03949010", "RS_03949022", "RS_03949028", "RS_03949046",
    "RS_03949070", "RS_03949094", "RS_extra_HN-C-00044-7",
    "RS_extra_HN-C-00083-4"
]

VAL_PATIENTS = [
    "RS_03949058",
    "RS_03948956", "RS_03948980", "RS_03949016"
]


class CD44Dataset(Dataset):
    """
    PyTorch Dataset for CD44 biomarker segmentation.
    - Images : channel 14 only → shape (1, 256, 256)
    - Masks  : .png with pixel values 0-3
                 0 = background  → ignore (255)
                 1 = CD44 negative → class 0
                 2 = CD44 low      → class 1
                 3 = CD44 high     → class 2
    """

    def __init__(self, data_root, patients, marker="CD44", augment=False):
        self.data_root = data_root
        self.marker    = marker
        self.augment   = augment
        self.samples   = []

        for patient in patients:
            img_dir  = os.path.join(data_root, patient, marker, "images")
            mask_dir = os.path.join(data_root, patient, marker, "masks")

            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                print(f"[WARN] Skipping {patient} — missing CD44 folder")
                continue

            mat_files = sorted(glob.glob(os.path.join(img_dir, "*.mat")))

            for mat_path in mat_files:
                patch_name = os.path.basename(mat_path).replace(".mat", "")
                mask_path  = os.path.join(mask_dir, patch_name + ".png")

                if not os.path.exists(mask_path):
                    continue

                mask = np.array(Image.open(mask_path))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]

                tissue_fraction = (mask != 0).sum() / mask.size
                if tissue_fraction < 0.3:
                    continue

                self.samples.append((mat_path, mask_path))

        print(f"  Loaded {len(self.samples)} patches from {len(patients)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, mask_path = self.samples[idx]

        # load channel 14 only
        with h5py.File(mat_path, 'r') as f:
            ftir = f['patch_15_chw'][:]              # (256, 256, 15)

        img = torch.tensor(ftir.copy(), dtype=torch.float32).permute(2, 0, 1)  # (15, 256, 256)
        img = img[13:14, :, :]                       # channel 14 only → (1, 256, 256)

        # load mask
        mask = np.array(Image.open(mask_path)).astype(np.int64)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        remap = {0: 255, 1: 0, 2: 1, 3: 2}
        remapped = np.full_like(mask, 255)
        for orig, new in remap.items():
            remapped[mask == orig] = new

        mask = torch.tensor(remapped, dtype=torch.long)

        # augmentation (training only)
        if self.augment:
            if torch.rand(1) > 0.5:
                img  = torch.flip(img,  dims=[2])
                mask = torch.flip(mask, dims=[1])
            if torch.rand(1) > 0.5:
                img  = torch.flip(img,  dims=[1])
                mask = torch.flip(mask, dims=[0])
            k = torch.randint(0, 4, (1,)).item()
            img  = torch.rot90(img,  k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])
            if torch.rand(1) > 0.5:
                factor = 0.8 + torch.rand(1).item() * 0.4
                img    = torch.clamp(img * factor, 0, 1)
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(img) * 0.02
                img   = torch.clamp(img + noise, 0, 1)

        return img, mask


def get_dataloaders(data_root, batch_size=16, num_workers=1):
    train_ds = CD44Dataset(data_root, TRAIN_PATIENTS, augment=True)
    val_ds   = CD44Dataset(data_root, VAL_PATIENTS,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    DATA_ROOT = "/u/aayyah3/oral-biomarker-segmentation/data/Oral_Final_Biomarkers"
    train_loader, val_loader = get_dataloaders(DATA_ROOT, batch_size=4)
    imgs, masks = next(iter(train_loader))
    print("Image batch shape :", imgs.shape)    # (4, 1, 256, 256)
    print("Mask  batch shape  :", masks.shape)  # (4, 256, 256)
    print("Image min/max      :", imgs.min().item(), imgs.max().item())
    print("Mask unique values :", masks.unique())