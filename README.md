# Uncle Sam

**Interactive, point-prompted image segmentation with Meta's Segment Anything Model 2 (SAM 2.1).**

Uncle Sam is an experimental Python project for exploring image segmentation with SAM 2.1 Small. Its first inference demo opens an image in a window, lets you mark foreground and background points, and displays the best predicted mask as you work.

> **Project status:** The segmentation demo is an early, local prototype. Its image and checkpoint paths are currently configured in the script. The installed `sam` command is only a starter CLI that prints a greeting; it does not launch the demo.

## What you can do

- Load an image and compute its SAM 2.1 image embeddings.
- Add **positive** prompts with the left mouse button and **negative** prompts with the right mouse button.
- Preview the selected mask, its score, and the prompt points.
- Clear prompts or save the final overlay to `outputs/sam2_first_inference.png`.

## Requirements

- Python **3.14 or newer**, as specified in [`pyproject.toml`](pyproject.toml).
- [uv](https://docs.astral.sh/uv/) for managing the root Python project.
- A working installation of [SAM 2](https://github.com/facebookresearch/sam2) and its compatible dependencies (including PyTorch and OpenCV).
- A SAM 2.1 Small checkpoint and an input image at the paths described below.
- A graphical desktop session for the OpenCV interactive window. CUDA is used when available; otherwise the demo falls back to CPU.

## Getting started

Clone the repository with its SAM 2 submodule:

```bash
git clone --recurse-submodules https://github.com/YusufCaymazZ/Uncle-Sam.git
cd Uncle-Sam
```

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

Set up the root project:

```bash
uv sync
```

The root project currently declares no runtime dependencies for the inference demo. Install SAM 2 and its dependencies into the same environment created by `uv sync`, following the [SAM 2 installation instructions](https://github.com/facebookresearch/sam2#installation) for the PyTorch build that matches your hardware. For example, after initializing the submodule, its editable install can be run from the repository root with:

```bash
uv pip install --python .venv/bin/python -e ./sam2
```

Check the upstream compatibility notes if your Python or PyTorch version is not supported by the current SAM 2 release.

Place the SAM 2.1 Small checkpoint at:

```text
sam2/checkpoints/sam2.1_hiera_small.pt
```

Provide an input image at:

```text
data/Test/Rural/images_png/4191.png
```

Then run the interactive demo from the repository root:

```bash
uv run python src/segmentation/first_inference.py
```

## Using the demo

| Action | Result |
| --- | --- |
| Left-click | Add a positive (foreground) point |
| Right-click | Add a negative (background) point |
| `R` | Clear all points |
| `Q` | Quit and save the current mask overlay if points were added |

The demo chooses the highest-scoring mask from SAM's multiple predictions. The saved image includes the mask overlay, contour, prompts, and score.

## Repository layout

```text
.
├── sam2/                           # SAM 2 Git submodule
├── src/
│   ├── sam/__init__.py             # Starter `sam` CLI entry point
│   └── segmentation/
│       └── first_inference.py      # Interactive SAM 2.1 image demo
├── data/Test/Rural/images_png/     # Expected location of the demo image
└── outputs/                        # Created for saved inference overlays
```

## Configuration

The current demo uses fixed paths and settings in `src/segmentation/first_inference.py`:

- **Image:** `data/Test/Rural/images_png/4191.png`
- **Checkpoint:** `sam2/checkpoints/sam2.1_hiera_small.pt`
- **Model config:** `configs/sam2.1/sam2.1_hiera_s.yaml` from the SAM 2 submodule
- **Output:** `outputs/sam2_first_inference.png`

Update those constants in the script to use a different image, checkpoint, or output location.

## Acknowledgements

This project builds on [Segment Anything 2](https://github.com/facebookresearch/sam2) by Meta AI. Please refer to the upstream repository for model, checkpoint, and license details.