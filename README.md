<div align="center">

# Uncle Sam

### Point. Prompt. Segment.

An interactive image-segmentation playground powered by Meta's **Segment Anything Model 2.1**.

[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SAM 2.1](https://img.shields.io/badge/Model-SAM%202.1-7B61FF)](https://github.com/facebookresearch/sam2)
[![PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![uv](https://img.shields.io/badge/Package%20manager-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![CI](https://github.com/YusufCaymazZ/Uncle-Sam/actions/workflows/ci.yml/badge.svg)](https://github.com/YusufCaymazZ/Uncle-Sam/actions/workflows/ci.yml)

**Interactive Demo** · **Computer Vision** · **Image Segmentation**

</div>

---

Uncle Sam is an experimental Python project for exploring promptable image segmentation. Click an image to mark foreground and background points, then watch SAM 2.1 Small generate and update the mask.

> [!NOTE]
> This is an early local prototype. The demo currently uses fixed image and checkpoint paths. The installed `sam` command is a starter CLI that prints a greeting; it does not launch the segmentation demo.

## ✨ Demo features

- Compute image embeddings with SAM 2.1 Small.
- Add **foreground** prompts with left-clicks and **background** prompts with right-clicks.
- Preview the selected mask, confidence score, contours, and prompt points.
- Reset prompts at any time and save the final overlay to `outputs/sam2_first_inference.png`.
- Use CUDA when available, with CPU fallback.

## 🚀 Getting started

### Requirements

- Python **3.14 or newer**, as specified in [`pyproject.toml`](pyproject.toml).
- [uv](https://docs.astral.sh/uv/) for managing the root project.
- [SAM 2](https://github.com/facebookresearch/sam2), PyTorch, and OpenCV installed in the same environment.
- A SAM 2.1 Small checkpoint, an input image, and a graphical desktop session.

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/YusufCaymazZ/Uncle-Sam.git
cd Uncle-Sam
```

Already cloned without submodules? Initialize SAM 2 with:

```bash
git submodule update --init --recursive
```

### 2. Set up the environment

```bash
uv sync --locked
```

The root project does not yet declare the inference demo's runtime dependencies. Follow [SAM 2's installation instructions](https://github.com/facebookresearch/sam2#installation) to install a PyTorch build that matches your hardware, then install the submodule into the project's environment:

```bash
uv pip install -e ./sam2
```

Check SAM 2's compatibility notes if your Python or PyTorch version is not supported by the current upstream release.

### 3. Add model and image files

Put the SAM 2.1 Small checkpoint here:

```text
sam2/checkpoints/sam2.1_hiera_small.pt
```

Put the input image here:

```text
data/Test/Rural/images_png/4191.png
```

### 4. Launch the demo

```bash
uv run python src/segmentation/first_inference.py
```

## 🖱️ Controls

| Input | Action |
|:--|:--|
| Left-click | Add a positive (foreground) point |
| Right-click | Add a negative (background) point |
| `R` | Clear all points |
| `Q` | Quit and save the current overlay if points were added |

The demo selects the highest-scoring mask from SAM's predictions. The saved overlay includes the mask, contour, prompts, and score.

## 🗂️ Project structure

```text
.
├── sam2/                           # SAM 2 Git submodule
├── src/
│   ├── sam/__init__.py             # Starter `sam` CLI entry point
│   └── segmentation/
│       └── first_inference.py      # Interactive SAM 2.1 image demo
├── data/Test/Rural/images_png/     # Expected input image location
└── outputs/                        # Created for saved inference overlays
```

## ⚙️ Demo configuration

The demo's paths and model settings are currently defined in `src/segmentation/first_inference.py`:

| Setting | Default |
|:--|:--|
| Input image | `data/Test/Rural/images_png/4191.png` |
| Checkpoint | `sam2/checkpoints/sam2.1_hiera_small.pt` |
| Model config | `configs/sam2.1/sam2.1_hiera_s.yaml` from the SAM 2 submodule |
| Saved overlay | `outputs/sam2_first_inference.png` |

Edit those constants in the script to use a different image, checkpoint, or output location.

## ✅ Continuous integration

GitHub Actions runs a smoke-test workflow on pushes and pull requests across Ubuntu, macOS, and Windows with Python 3.14. It installs the project, checks Python syntax, runs the installed `sam` command, and builds the package.

The workflow does not download the SAM 2 checkpoint or test model inference: those files are large, and inference depends on the local PyTorch/device setup and a graphical session. Follow the setup steps above to install SAM 2 and run the interactive demo on your machine.

## 🙏 Acknowledgements

Built on [Segment Anything 2](https://github.com/facebookresearch/sam2) by Meta AI. See the upstream repository for model, checkpoint, and license details.