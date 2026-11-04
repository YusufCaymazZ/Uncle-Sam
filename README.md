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
uv sync --locked --group demo
```

The `demo` dependency group installs NumPy and the OpenCV GUI build used by the interactive script. Follow [SAM 2's installation instructions](https://github.com/facebookresearch/sam2#installation) to install a PyTorch build that matches your hardware, then install the submodule into the project's environment:

```bash
uv pip install -e ./sam2
```

Check SAM 2's compatibility notes if your Python or PyTorch version is not supported by the current upstream release.

For development and standard CI checks, use the separate lightweight test dependency group (it uses headless OpenCV so automated tests cannot open GUI windows):

```bash
uv sync --locked --group test
```

### 3. Add model and image files

Put the SAM 2.1 Small checkpoint here:

```text
sam2/checkpoints/sam2.1_hiera_small.pt
```

The input image is stored in the repository here:

```text
test-data/4191.png
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
├── test-data/4191.png              # Tracked integration/demo input image
└── outputs/                        # Created for saved inference overlays
```

## ⚙️ Demo configuration

The demo's paths and model settings are currently defined in `src/segmentation/first_inference.py`:

| Setting | Default |
|:--|:--|
| Input image | `test-data/4191.png` |
| Checkpoint | `sam2/checkpoints/sam2.1_hiera_small.pt` |
| Model config | `configs/sam2.1/sam2.1_hiera_s.yaml` from the SAM 2 submodule |
| Saved overlay | `outputs/sam2_first_inference.png` |

Edit those constants in the script to use a different image, checkpoint, or output location.

## ✅ Continuous integration

GitHub Actions runs CPU-only checks on every push and pull request. The fast CI job installs the project and test dependencies with `uv`, compiles `src`, runs unit tests with mocked predictors, and builds the package. SAM 2 imports are checked when the submodule and its dependencies are available; normal CI does not download the submodule or model weights.

Actual SAM 2.1 inference is a separate, manually triggered workflow. It checks out the SAM 2 submodule, installs CPU-only PyTorch, verifies the Python and model dependencies, downloads and caches the official SAM 2.1 Small checkpoint, then runs inference against `test-data/4191.png`. A missing image or failed checkpoint download fails the workflow rather than reporting a successful skipped integration test. CPU execution is used when CUDA is unavailable, so no GPU is required.

The integration image is tracked at `test-data/4191.png`; the larger `data/` dataset remains excluded by `.gitignore`. Standard CI verifies that the image is present after checkout but does not download model weights or run real inference.

The model checkpoint is never committed. The manual workflow downloads it from Meta's [official SAM 2.1 Small checkpoint URL](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) and caches it between runs. SAM 2 source is maintained under Meta's [Apache 2.0 license](https://github.com/facebookresearch/sam2/blob/main/LICENSE); review the upstream terms for the checkpoint and give attribution to Meta AI and the [SAM 2 project](https://github.com/facebookresearch/sam2).

Run the same lightweight checks locally:

```bash
uv sync --locked --group test
uv run python -m compileall -q src
uv run --locked pytest -q tests/test_first_inference.py
uv build
```

To run actual CPU inference locally, initialize the submodule and install the CPU-only PyTorch build and SAM 2:

```bash
git submodule update --init --recursive
uv sync --locked --group test
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python -e ./sam2
mkdir -p .cache/sam2
curl --fail --location --retry 2 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
  --output .cache/sam2/sam2.1_hiera_small.pt
SAM2_CHECKPOINT_PATH="$PWD/.cache/sam2/sam2.1_hiera_small.pt" \
SAM2_CPU_THREADS=2 \
  uv run --locked --no-sync pytest -q tests/integration/test_sam2_inference.py
```

The integration test fails if its checkpoint is missing. It uses headless OpenCV and does not open GUI windows; the interactive desktop demo remains available via the launch command above.

## 🙏 Acknowledgements

Built on [Segment Anything 2](https://github.com/facebookresearch/sam2) by Meta AI. See the upstream repository for model, checkpoint, and license details.