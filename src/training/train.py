
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.loveda_dataset import LoveDADataset
from src.models.segformer import create_model


# Configuration
ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7
IGNORE_INDEX = 255
CROP_SIZE = 512

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 6e-5
NUM_WORKERS = 2

CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet normalization (ADE20K pretrained model)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def random_crop(images, masks):
    """Apply the same random crop coordinates to each image-mask pair."""
    cropped_images = []
    cropped_masks = []

    height, width = masks.shape[-2:]

    for image, mask in zip(images, masks):
        top = torch.randint(0, height - CROP_SIZE + 1, (1,)).item()
        left = torch.randint(0, width - CROP_SIZE + 1, (1,)).item()

        cropped_images.append(
            image[:, top:top + CROP_SIZE, left:left + CROP_SIZE]
        )
        cropped_masks.append(
            mask[top:top + CROP_SIZE, left:left + CROP_SIZE]
        )

    return torch.stack(cropped_images), torch.stack(cropped_masks)


def prepare_validation(images, masks):
    """Resize validation samples to 512x512."""
    images = F.interpolate(
        images,
        size=(CROP_SIZE, CROP_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    masks = F.interpolate(
        masks.unsqueeze(1).float(),
        size=(CROP_SIZE, CROP_SIZE),
        mode="nearest",
    ).squeeze(1).long()

    return images, masks


def normalize(images):
    mean = MEAN.to(images.device)
    std = STD.to(images.device)
    return (images - mean) / std


def calculate_miou(confusion):
    intersection = torch.diag(confusion)
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection

    valid_classes = union > 0
    iou = intersection[valid_classes] / union[valid_classes]

    return iou.mean().item(), iou.cpu().tolist()


@torch.no_grad()
def validate(model, loader):
    model.eval()

    confusion = torch.zeros(
        NUM_CLASSES, NUM_CLASSES,
        dtype=torch.int64,
        device=DEVICE,
    )

    total_loss = 0.0

    for images, masks in loader:
        images, masks = prepare_validation(images, masks)

        images = normalize(images.to(DEVICE, non_blocking=True))
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda",
        ):
            logits = model(pixel_values=images).logits
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = F.cross_entropy(
                logits,
                masks,
                ignore_index=IGNORE_INDEX,
            )

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)
        valid = masks != IGNORE_INDEX

        encoded = (
            masks[valid] * NUM_CLASSES + predictions[valid]
        )

        confusion += torch.bincount(
            encoded,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    miou, class_ious = calculate_miou(confusion)

    return total_loss / len(loader), miou, class_ious


def main():
    print("Device:", DEVICE)

    train_dataset = LoveDADataset(root=ROOT / "data", split="Train")
    val_dataset = LoveDADataset(root=ROOT / "data", split="Val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda",
    )

    best_miou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, (images, masks) in enumerate(train_loader):
            # Crop on CPU to reduce GPU memory usage.
            images, masks = random_crop(images, masks)

            images = normalize(images.to(DEVICE, non_blocking=True))
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(pixel_values=images).logits

                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                loss = F.cross_entropy(
                    logits,
                    masks,
                    ignore_index=IGNORE_INDEX,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Step {step + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        val_loss, val_miou, class_ious = validate(model, val_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val mIoU:   {val_miou:.4f}")
        print("Class IoU:", [round(x, 4) for x in class_ious])

        if val_miou > best_miou:
            best_miou = val_miou

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou,
                },
                CHECKPOINT_DIR / "segformer_b0_best.pth",
            )

            print("Best model saved.")

        print("-" * 50)

    print("Training completed.")
    print("Best validation mIoU:", best_miou)


if __name__ == "__main__":
    main()
