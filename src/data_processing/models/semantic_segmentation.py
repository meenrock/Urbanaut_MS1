import bentoml
import numpy as np
import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm
from torch import optim
from torch.hub import download_url_to_file

import zipfile
from pathlib import Path

from src.data_input.repositories.satellite_imagery_dataset import SatelliteImageryDataset
from src.data_processing.loss_function.semantic_cross_entropy_loss import SemanticCrossEntropyLoss
from src.data_processing.models.UNET.UNet import UNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE

if not Path("train_masks.zip").exists():
    download_url_to_file("https://figshare.com/ndownloader/articles/19961426/versions/1", dst="train_masks.zip")
if not Path("train_images.zip").exists():
    download_url_to_file("https://figshare.com/ndownloader/articles/19961336/versions/1", dst="train_images.zip")

train_images_dir = Path("train_images")
if not train_images_dir.exists():
    train_images_zip = zipfile.ZipFile("train_images.zip", "r")
    train_images_dir.mkdir(exist_ok=True)
    train_images_zip.extractall(path=train_images_dir)

train_masks_dir = Path("train_masks")
if not train_masks_dir.exists():
    train_masks_zip = zipfile.ZipFile("train_masks.zip", "r")
    train_masks_dir.mkdir(exist_ok=True)
    train_masks_zip.extractall(path=train_masks_dir)

train_dataset = SatelliteImageryDataset("./", train=True)
val_dataset = SatelliteImageryDataset("./", train=False)
len(train_dataset), len(val_dataset)


def view_sample(ds, idx):
    _, axs = plt.subplots(ncols=2)

    img, mask = ds[idx]

    axs[0].imshow(img)
    axs[0].axis("off")

    axs[1].imshow(mask)
    axs[1].axis("off")

    plt.show()

view_sample(train_dataset, 1)
view_sample(val_dataset, 1)

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4

N_CHANNELS = 3
N_CLASSES = 25

N_EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

NORMALIZATION_MEANS = [0.4643, 0.3185, 0.3141]
NORMALIZATION_STDS = [0.2171, 0.1561, 0.1496]

model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(DEVICE)

transform = T.Compose(
    [
        T.CenterCrop(3000),
        T.Resize(512),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZATION_MEANS, std=NORMALIZATION_STDS),
    ]
)

mask_transform = T.Compose(
    [
        T.CenterCrop(3000),
        T.Resize(512),
        np.asarray,
        torch.tensor,
    ]
)

train_dataset = SatelliteImageryDataset(
    "./", train=True, transform=transform, mask_transform=mask_transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
)
validation_dataset = SatelliteImageryDataset(
    "./", train=False, transform=transform, mask_transform=mask_transform
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True
)

# Try other optimizers if you want !
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Here choose the loss function.
loss_fn = SemanticCrossEntropyLoss()
# loss_fn = SemanticFocalLoss(gamma=1)

losses = []
accs = []

for epoch in range(N_EPOCHS):

    epoch_losses = []

    model.train()  # Setting the model in train mode i.e. activates all training layers.

    # Just to have a nice loading bar.
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} ")

    for images, labels in pbar:
        # Remember to send the data to the GPU (if you are using it !).
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        masks_pred = model(images)
        loss = loss_fn(masks_pred, labels)  # There was dice loss.

        # You know the drill.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = loss.cpu().item()

        epoch_losses.append(loss_value)
        pbar.set_description(f"Epoch {epoch}, Loss : {loss_value:.2f} ")

    losses.append(epoch_losses)

    model.eval()

    # Validations step.

    well_predicted = 0
    total_predicted = 0

    for images, labels in validation_loader:
        with torch.no_grad():
            preds = model(images.to(DEVICE))

        pred_labels = preds.argmax(dim=1)
        well_predicted += (labels.to(DEVICE) == pred_labels).sum()
        total_predicted += labels.numel()

    val_acc = well_predicted / total_predicted
    print(f"Validation Accuracy : {val_acc:2.3f}.")
    accs.append(val_acc)

    torch.save(model.state_dict(), f"semantic_segmentation_si_unet_epoch_{epoch:03d}.model")

    save_path = "semantic_segmentation_si_unet.model"
    torch.save(model.state_dict(), save_path)
    bentoml.pytorch.save_model("parkinglot_unet",model)