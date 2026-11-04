import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAM2_ROOT = ROOT / "sam2"
IMAGE_PATH = ROOT / "test-data" / "4191.png"
CHECKPOINT = SAM2_ROOT / "checkpoints/sam2.1_hiera_small.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OUTPUT_PATH = ROOT / "outputs/sam2_first_inference.png"
WINDOW_NAME = "TerraVision - SAM 2.1 Small"


class PromptState:
    def __init__(self):
        self.points = []
        self.labels = []

    def add_point(self, x, y, label):
        if label not in (0, 1):
            raise ValueError("Prompt labels must be 1 (positive) or 0 (negative).")
        self.points.append([int(x), int(y)])
        self.labels.append(int(label))

    def clear(self):
        self.points.clear()
        self.labels.clear()


def validate_image(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Images must have shape (height, width, 3) in RGB order.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Images must have non-zero height and width.")
    if image.dtype != np.uint8:
        raise ValueError("Images must use uint8 pixel values in the range [0, 255].")
    return image


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return validate_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def validate_prompts(points, labels, image_shape):
    coords = np.asarray(points, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)

    if len(coords) == 0:
        raise ValueError("At least one point prompt is required.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Point prompts must have shape (N, 2).")
    if point_labels.ndim != 1 or len(point_labels) != len(coords):
        raise ValueError("Each point prompt must have exactly one label.")
    if not np.isfinite(coords).all():
        raise ValueError("Point coordinates must be finite.")
    if not np.isin(point_labels, (0, 1)).all():
        raise ValueError("Prompt labels must be 1 (positive) or 0 (negative).")

    height, width = image_shape[:2]
    if (
        (coords[:, 0] < 0).any()
        or (coords[:, 0] >= width).any()
        or (coords[:, 1] < 0).any()
        or (coords[:, 1] >= height).any()
    ):
        raise ValueError("Point prompts must fall within the image bounds.")

    return coords, point_labels


def select_best_mask(masks, scores, image_shape):
    masks = np.asarray(masks)
    scores = np.asarray(scores)

    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("Predicted masks must have shape (N, height, width).")
    if scores.ndim != 1 or len(scores) != len(masks):
        raise ValueError("Each predicted mask must have exactly one score.")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must be finite.")
    if masks.shape[1:] != tuple(image_shape[:2]):
        raise ValueError("Predicted mask dimensions must match the input image.")

    best_index = int(np.argmax(scores))
    return masks[best_index].astype(bool), float(scores[best_index])


def predict_mask(predictor, image, points, labels):
    image = validate_image(image)
    coords, point_labels = validate_prompts(points, labels, image.shape)
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return select_best_mask(masks, scores, image.shape)


def render_prediction(image, mask, points, labels, score):
    image = validate_image(image)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask dimensions must match the input image.")
    coords, point_labels = validate_prompts(points, labels, image.shape)

    display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = display.copy()
    overlay[mask] = (0, 200, 0)
    display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

    for (x, y), label in zip(coords, point_labels, strict=True):
        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        cv2.circle(display, (int(x), int(y)), 6, color, -1)
        cv2.circle(display, (int(x), int(y)), 9, (255, 255, 255), 2)

    cv2.putText(
        display,
        f"Mask score: {float(score):.3f}",
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
    return display


def save_result(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Unable to save inference output: {path}")
    return path


def load_sam2_components():
    if not (SAM2_ROOT / "sam2" / "build_sam.py").is_file():
        raise FileNotFoundError(
            f"SAM 2 submodule is missing at {SAM2_ROOT}. "
            "Run `git submodule update --init --recursive`."
        )

    sam2_path = str(SAM2_ROOT)
    if sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return torch, build_sam2, SAM2ImagePredictor


def build_predictor(checkpoint_path=CHECKPOINT, model_config=MODEL_CONFIG):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint_path}")

    config_path = SAM2_ROOT / "sam2" / model_config
    if not config_path.is_file():
        raise FileNotFoundError(f"SAM 2 model config not found: {config_path}")

    torch, build_sam2, predictor_type = load_sam2_components()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_sam2(
        model_config,
        str(checkpoint_path),
        device=device,
    )
    return predictor_type(model), torch, device


def prepare_image_embeddings(predictor, image, torch, device):
    image = validate_image(image)
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device == "cuda",
        ):
            predictor.set_image(image)


def handle_mouse_event(event, x, y, prompts):
    if event == cv2.EVENT_LBUTTONDOWN:
        prompts.add_point(x, y, 1)
        return "positive"
    if event == cv2.EVENT_RBUTTONDOWN:
        prompts.add_point(x, y, 0)
        return "negative"
    return None


def main():
    global active_image, active_predictor, active_prompts, active_torch, active_device

    active_image = load_image(IMAGE_PATH)
    active_predictor, active_torch, active_device = build_predictor()
    active_prompts = PromptState()
    prepare_image_embeddings(
        active_predictor,
        active_image,
        active_torch,
        active_device,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1024, 768)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    cv2.imshow(WINDOW_NAME, cv2.cvtColor(active_image, cv2.COLOR_RGB2BGR))

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                active_prompts.clear()
                print("Prompts cleared.")
                cv2.imshow(
                    WINDOW_NAME,
                    cv2.cvtColor(active_image, cv2.COLOR_RGB2BGR),
                )

        if active_prompts.points:
            result = predict_and_display()
            if result is not None:
                print(f"Saved: {save_result(OUTPUT_PATH, result)}")
    finally:
        cv2.destroyAllWindows()


def predict_and_display():
    if not active_prompts.points:
        return None
    mask, score = predict_mask(
        active_predictor,
        active_image,
        active_prompts.points,
        active_prompts.labels,
    )
    display = render_prediction(
        active_image,
        mask,
        active_prompts.points,
        active_prompts.labels,
        score,
    )
    cv2.imshow(WINDOW_NAME, display)
    return display


def mouse_callback(event, x, y, flags, param):
    prompt_type = handle_mouse_event(event, x, y, active_prompts)
    if prompt_type is not None:
        print(f"{prompt_type.capitalize()} point: ({x}, {y})")
        predict_and_display()


if __name__ == "__main__":
    main()
