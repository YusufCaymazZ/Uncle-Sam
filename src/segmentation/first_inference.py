
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "sam2"))

IMAGE_PATH = ROOT / "data/Test/Rural/images_png/4191.png"

CHECKPOINT = ROOT / "sam2/checkpoints/sam2.1_hiera_small.pt"

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "sam2_first_inference.png"

points = []
labels = []

image = None
predictor = None
window_name = "TerraVision - SAM 2.1 Small"


def predict_and_display():
    if not points:
        return

    coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=point_labels,
                multimask_output=True,
            )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)

    # Original image
    display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Overlay mask
    overlay = display.copy()
    overlay[best_mask] = (0, 200, 0)

    display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

    # Draw mask boundaries
    mask_uint8 = best_mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

    # Draw prompts
    for (x, y), label in zip(points, labels):
        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        cv2.circle(display, (int(x), int(y)), 6, color, -1)
        cv2.circle(display, (int(x), int(y)), 9, (255, 255, 255), 2)

    score = float(scores[best_idx])

    cv2.putText(
        display,
        f"Mask score: {score:.3f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "Left: Positive | Right: Negative | R: Reset | Q: Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow(window_name, display)

    return display


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)

        print(f"Positive point: ({x}, {y})")
        predict_and_display()

    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)

        print(f"Negative point: ({x}, {y})")
        predict_and_display()


def main():
    global image, predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    image = cv2.cvtColor(
        cv2.imread(str(IMAGE_PATH)),
        cv2.COLOR_BGR2RGB,
    )

    print(f"Image shape: {image.shape}")

    print("Loading SAM 2.1 Small...")

    model = build_sam2(
        MODEL_CONFIG,
        str(CHECKPOINT),
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    print("Computing image embeddings...")

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device == "cuda",
        ):
            predictor.set_image(image)

    print("Model ready.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)
    cv2.setMouseCallback(window_name, mouse_callback)

    cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            points.clear()
            labels.clear()

            print("Prompts cleared.")

            cv2.imshow(
                window_name,
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )

    if points:
        result = predict_and_display()

        if result is not None:
            cv2.imwrite(str(OUTPUT_PATH), result)
            print(f"Saved: {OUTPUT_PATH}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

