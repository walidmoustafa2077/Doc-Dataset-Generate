# SynDoc_Wild_v1: Synthetic Document Shadow Removal Dataset Generator

Generate realistic synthetic document images with shadows, camera degradation, and variations for training deep learning shadow removal models.

## Overview

This project creates training datasets for document shadow removal by:
- Extracting document pages from PDFs (1000+ documents)
- Applying realistic paper textures and perspective warping (simulating handheld camera capture)
- Compositing realistic shadows with ambient color tinting
- Applying camera degradation (blur, noise, JPEG compression)
- Generating triplets: `input` (shadowed), `target` (clean), `mask` (shadow shape)

**Dataset Output**: `SynDoc_Wild_v1/train/` and `test/` splits with synchronized input/target/mask sets.

## Quick Start

### Prerequisites

- Python 3.8+
- Dependencies: `numpy`, `opencv-python`, `PyMuPDF`, `tqdm`

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Doc-Dataset-Generate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Extract Images from PDFs

Place PDF files in `PDFs/` folder, then:

```bash
python extract_pdf_images.py
```

Output: `Extracted_Images/1/`, `Extracted_Images/2/`, etc. with page images

#### 2. Generate Dataset

```bash
# Full dataset generation
python dataset_generator.py

# Process first 10 documents only
python dataset_generator.py --limit 10

# Debug mode: save intermediate images for first 5 documents
python dataset_generator.py --debug

# Combine options
python dataset_generator.py --limit 10 --debug
```

Output: `SynDoc_Wild_v1/train/` with `input/`, `target/`, `mask/` subdirectories

## Project Structure

```
Doc-Dataset-Generate/
├── config.py                    # Global configuration & parameters
├── asset_manager.py             # Image asset pooling & randomization
├── image_processor.py           # Core image transformation algorithms
├── dataset_generator.py         # Main orchestration pipeline
├── extract_pdf_images.py        # PDF extraction utility
├── requirements.txt             # Python dependencies
├── CONTRIBUTING.md              # Development guidelines
├── LICENSE                      # MIT License
│
├── PDFs/                        # Input: PDF documents
├── Extracted_Images/            # Intermediate: Extracted PDF pages
├── Paper Texture/               # Asset: Paper grain textures
├── Shadow Overlays/             # Asset: Shadow overlays
├── Background/                  # Asset: Background images (wood, carpet, etc.)
└── SynDoc_Wild_v1/              # Output dataset
    ├── train/
    │   ├── input/               # Shadowed + degraded images
    │   ├── target/              # Clean ground truth
    │   └── mask/                # Binary shadow masks
    └── test/
        ├── input/, target/, mask/
```

## Key Features

### Realistic Shadow Rendering
- Shadows are tinted with **ambient background color** (wood/carpet/white) for photorealism
- Gaussian blur (5-15px) for soft shadow edges
- Random opacity (0.6-0.9) for variation

### Paper Texture Blending
- **Multiply blending** preserves text contrast while adding grain
- Blend strength varies (0.6-0.8) for dataset diversity

### Perspective Warp
- Random corner offsets (100-150px) simulate handheld camera angles
- Creates trapezoid distortion (narrowing at top) for realism

### Camera Degradation Pipeline
1. Defocus blur (2-4px kernel)
2. Gaussian noise (σ=6-8)
3. JPEG compression (quality 80-90)

### Reproducibility
- Configurable `RANDOM_SEED` for deterministic generation
- Asset selection without repetition (round-robin with shuffle)

## Configuration

Edit `config.py` to adjust:

- **Image dimensions**: `TARGET_WIDTH=1240`, `TARGET_HEIGHT=1754` (A4 at 150 DPI)
- **Warp intensity**: `WARP_MIN_OFFSET=100`, `WARP_MAX_OFFSET=150`
- **Shadow opacity**: `SHADOW_OPACITY_MIN=0.6`, `SHADOW_OPACITY_MAX=0.9`
- **Camera blur**: `DEFOCUS_BLUR_MIN=2`, `DEFOCUS_BLUR_MAX=4`
- **Texture visibility**: `TEXTURE_BLEND_MIN=0.6`, `TEXTURE_BLEND_MAX=0.8`
- **Processing**: `NUM_WORKERS=8` (parallel threads), `RANDOM_SEED=None` (reproducibility)

## Dataset Format

Output triplets for machine learning:

```
SynDoc_Wild_v1/train/
├── input/0001.png              # Shadowed + degraded (input to model)
├── target/0001.png             # Clean document (ground truth)
└── mask/0001.png               # Binary shadow mask (supervision signal)
```

- **Sequential naming**: 0001.png to NNNN.png (4-digit zero-padded)
- **PNG lossless**: Compression=8 for smaller files
- **Equal triplets**: Same number of input/target/mask files (n=1000+ in train/)

## Debugging

### Debug Mode
```bash
python dataset_generator.py --debug
```

Saves 7 intermediate images per document to `SynDoc_Wild_v1/debug/`:
1. `01_resized.png` - A4 resized document
2. `02_warped.png` - Perspective warp applied
3. `03_textured_target.png` - Paper texture applied (TARGET)
4. `04_shadow_applied.png` - Shadow before compositing
5. `05_shadowed.png` - Shadow composited onto document
6. `06_degraded_input.png` - Camera degradation applied (INPUT)
7. `07_mask.png` - Binary shadow mask

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "No textures found" | Add PNG/JPG files to `Paper Texture/` directory |
| "No shadows found" | Add PNG/JPG files to `Shadow Overlays/` directory |
| "No backgrounds found" | Add PNG/JPG files to `Background/` directory |
| Slow processing | Increase `NUM_WORKERS` in config.py (if CPU allows) |
| Memory errors | Reduce `NUM_WORKERS` or use `--limit` to process in batches |

## Performance

**Baseline**: ~2-3 seconds per document on 8-core CPU with `NUM_WORKERS=8`

For 1000 documents:
- Single-threaded: ~30-50 minutes
- 8 workers: ~4-6 minutes

## Code Style & Standards

- **Type hints**: Required for all function signatures
- **Docstrings**: Triple-quoted with Args/Returns sections
- **Config usage**: Import from `config.py`, no hardcoded paths/values
- **Path handling**: Use `pathlib.Path`, not string concatenation
- **PEP 8**: 4-space indent, max 100 character line length

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Follow code standards in [CONTRIBUTING.md](CONTRIBUTING.md)
4. Submit a pull request with a clear description

## License

MIT License - See [LICENSE](LICENSE) for details

## References

- **A4 Standard**: 1240×1754 pixels @ 150 DPI
- **Paper Texture Blending**: Multiply mode preserves text contrast while adding grain
- **Shadow Removal ML**: Dataset designed for U-Net or similar architectures trained with triplet supervision
- **Camera Simulation**: Combines realistic blur, noise, and compression artifacts

## Troubleshooting & Support

For issues or questions:
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Review debug mode output to inspect processing pipeline
- Verify asset directories have image files with supported extensions (.png, .jpg, .jpeg, .bmp, .tiff)
- See `.github/copilot-instructions.md` for AI agent development guidance
