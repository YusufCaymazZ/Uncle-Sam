import cv2
import numpy as np
import pytest

from segmentation import first_inference as inference


class MockPredictor:
    def __init__(self, masks=None, scores=None):
        self.masks = masks
        self.scores = scores
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.masks, self.scores, None


def test_load_image_converts_bgr_to_rgb_and_preserves_shape(tmp_path):
    bgr_image = np.zeros((5, 7, 3), dtype=np.uint8)
    bgr_image[:] = (10, 20, 30)
    image_path = tmp_path / "input.png"
    assert cv2.imwrite(str(image_path), bgr_image)

    image = inference.load_image(image_path)

    assert image.shape == (5, 7, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], (30, 20, 10))


def test_load_image_reports_missing_and_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        inference.load_image(tmp_path / "missing.png")

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to decode"):
        inference.load_image(invalid_path)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((4, 4), dtype=np.uint8), "shape"),
        (np.zeros((4, 4, 4), dtype=np.uint8), "shape"),
        (np.zeros((4, 4, 3), dtype=np.float32), "uint8"),
        (np.zeros((0, 4, 3), dtype=np.uint8), "non-zero"),
    ],
)
def test_validate_image_rejects_invalid_dimensions_or_dtype(image, message):
    with pytest.raises(ValueError, match=message):
        inference.validate_image(image)


def test_prompt_state_tracks_positive_and_negative_points():
    prompts = inference.PromptState()
    prompts.add_point(3, 5, 1)
    prompts.add_point(7, 9, 0)

    assert prompts.points == [[3, 5], [7, 9]]
    assert prompts.labels == [1, 0]

    prompts.clear()
    assert prompts.points == []
    assert prompts.labels == []


def test_mouse_events_add_positive_and_negative_prompts():
    prompts = inference.PromptState()

    assert (
        inference.handle_mouse_event(cv2.EVENT_LBUTTONDOWN, 2, 4, prompts)
        == "positive"
    )
    assert (
        inference.handle_mouse_event(cv2.EVENT_RBUTTONDOWN, 6, 8, prompts)
        == "negative"
    )
    assert inference.handle_mouse_event(cv2.EVENT_MOUSEMOVE, 0, 0, prompts) is None
    assert prompts.points == [[2, 4], [6, 8]]
    assert prompts.labels == [1, 0]


def test_predict_mask_passes_typed_prompts_and_selects_highest_score():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    masks = np.zeros((3, 6, 8), dtype=np.uint8)
    masks[1, 2:4, 3:5] = 1
    masks[2, 0:2, 0:2] = 1
    predictor = MockPredictor(masks, np.array([0.2, 0.9, 0.6]))

    best_mask, score = inference.predict_mask(
        predictor,
        image,
        [[3, 2], [4, 3]],
        [1, 0],
    )

    assert best_mask.dtype == np.bool_
    assert best_mask[2, 3]
    assert not best_mask[0, 0]
    assert score == pytest.approx(0.9)
    assert predictor.calls[0]["point_coords"].dtype == np.float32
    np.testing.assert_array_equal(predictor.calls[0]["point_labels"], [1, 0])
    assert predictor.calls[0]["point_labels"].dtype == np.int32
    assert predictor.calls[0]["multimask_output"] is True


@pytest.mark.parametrize(
    ("points", "labels", "message"),
    [
        ([[1, 1]], [], "exactly one label"),
        ([[8, 1]], [1], "image bounds"),
        ([[1, 1]], [2], "labels must be"),
        ([[float("nan"), 1]], [1], "finite"),
        ([], [], "At least one"),
    ],
)
def test_predict_mask_rejects_invalid_prompts(points, labels, message):
    predictor = MockPredictor(
        np.zeros((1, 4, 8), dtype=np.uint8),
        np.array([0.5]),
    )
    with pytest.raises(ValueError, match=message):
        inference.predict_mask(
            predictor,
            np.zeros((4, 8, 3), dtype=np.uint8),
            points,
            labels,
        )
    assert predictor.calls == []


@pytest.mark.parametrize(
    ("masks", "scores", "message"),
    [
        (np.zeros((0, 4, 8)), np.array([]), "Predicted masks"),
        (np.zeros((2, 4, 8)), np.array([0.5]), "exactly one score"),
        (np.zeros((1, 4, 8)), np.array([np.nan]), "finite"),
        (np.zeros((1, 3, 8)), np.array([0.5]), "dimensions"),
    ],
)
def test_select_best_mask_rejects_invalid_predictions(masks, scores, message):
    with pytest.raises(ValueError, match=message):
        inference.select_best_mask(masks, scores, (4, 8, 3))


def test_render_and_save_prediction(tmp_path):
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    mask = np.zeros((16, 20), dtype=bool)
    mask[8:14, 8:18] = True

    rendered = inference.render_prediction(image, mask, [[0, 0]], [1], 0.875)

    assert rendered.shape == (16, 20, 3)
    assert rendered.dtype == np.uint8
    assert tuple(rendered[10, 10]) == (18, 92, 6)

    output = inference.save_result(tmp_path / "nested" / "mask.png", rendered)
    assert output.is_file()
    saved = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == rendered.shape


def test_save_result_reports_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.cv2, "imwrite", lambda *_args: False)
    with pytest.raises(OSError, match="Unable to save"):
        inference.save_result(tmp_path / "output.png", np.zeros((2, 2, 3)))


def test_build_predictor_reports_missing_checkpoint_before_loading_sam2(
    tmp_path,
    monkeypatch,
):
    def fail_if_called():
        pytest.fail("SAM 2 should not be imported when the checkpoint is missing")

    monkeypatch.setattr(inference, "load_sam2_components", fail_if_called)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        inference.build_predictor(tmp_path / "missing.pt")


def test_build_predictor_reports_missing_config(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(inference, "SAM2_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="model config not found"):
        inference.build_predictor(checkpoint)


def test_sam2_imports_when_submodule_and_dependencies_are_available():
    if not (inference.SAM2_ROOT / "sam2" / "build_sam.py").is_file():
        pytest.skip("SAM 2 submodule is not checked out in standard CI.")

    pytest.importorskip("torch", reason="SAM 2 requires PyTorch.")
    pytest.importorskip("hydra", reason="SAM 2 requires hydra-core.")
    _, build_sam2, predictor_type = inference.load_sam2_components()
    assert callable(build_sam2)
    assert callable(predictor_type)
