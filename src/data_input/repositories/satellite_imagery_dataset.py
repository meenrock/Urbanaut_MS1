from PIL import Image
from pathlib import Path
import torch

class SatelliteImageryDataset(torch.utils.data.Dataset):
    def __init__(
            self, root, train=True, transform=lambda x: x, mask_transform=lambda x: x
    ):
        super().__init__()
        self.transform = transform
        self.mask_transform = mask_transform
        self.root = Path(root)
        self.train_images = self.root / "train_images"
        self.train_images_files = sorted(list(self.train_images.iterdir()))

        self.train_masks = self.root / "train_masks"
        self.train_masks_files = sorted(list(self.train_masks.iterdir()))

        self.train = train

        if train:
            self.train_images_files = [
                f
                for f in self.train_images_files
                if int(f.with_suffix("").name[-1]) < 8
            ]
            self.train_masks_files = [
                f for f in self.train_masks_files if int(f.with_suffix("").name[-1]) < 8
            ]
        else:
            self.train_images_files = [
                f
                for f in self.train_images_files
                if int(f.with_suffix("").name[-1]) >= 8
            ]
            self.train_masks_files = [
                f
                for f in self.train_masks_files
                if int(f.with_suffix("").name[-1]) >= 8
            ]

        self.classes = [
            "Background",
            "Property Roof",
            "Secondary Structure",
            "Swimming Pool",
            "Vehicle",
            "Grass",
            "Trees / Shrubs",
            "Solar Panels",
            "Chimney",
            "Street Light",
            "Window",
            "Satellite Antenna",
            "Garbage Bins",
            "Trampoline",
            "Road/Highway",
            "Under Construction / In Progress Status",
            "Power Lines & Cables",
            "Water Tank / Oil Tank",
            "Parking Area - Commercial",
            "Sports Complex / Arena",
            "Industrial Site",
            "Dense Vegetation / Forest",
            "Water Body",
            "Flooded",
            "Boat",
        ]

        assert len(self.train_images_files) == len(self.train_masks_files)

    def __len__(self):
        return len(self.train_images_files)

    def __getitem__(self, idx):
        mask = Image.open(self.train_masks_files[idx])
        image = Image.open(self.train_images_files[idx]).convert("RGB")
        return self.transform(image), self.mask_transform(mask)