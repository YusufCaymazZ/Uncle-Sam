import os
from pathlib import Path

import numpy as np

from segmentation import first_inference as inference


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "test-data" / "4191.png"


def test_real_sam2_small_image_inference():
    checkpoint = Path(os.environ.get("SAM2_CHECKPOINT_PATH", inference.CHECKPOINT))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 2.1 Small checkpoint is missing: {checkpoint}. "
            "Download the official checkpoint before running integration tests."
        )

    torch, _, _ = inference.load_sam2_components()
    cpu_threads = int(os.environ.get("SAM2_CPU_THREADS", "2"))
    torch.set_num_threads(cpu_threads)
    assert torch.get_num_threads() == cpu_threads

    image = inference.load_image(IMAGE_PATH)
    predictor, torch, device = inference.build_predictor(checkpoint)
    print(f"Running actual SAM 2.1 inference on {device}.")
    inference.prepare_image_embeddings(predictor, image, torch, device)

    prompts = inference.PromptState()
    height, width = image.shape[:2]
    prompts.add_point(width // 2, height // 2, 1)
    prompts.add_point(0, 0, 0)
    mask, score = inference.predict_mask(
        predictor,
        image,
        prompts.points,
        prompts.labels,
    )

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.bool_
    assert np.isfinite(score)
