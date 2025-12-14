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

## Terminal Commands Reference

### 1. PDF Image Extraction

**Default batch extraction** (extract all PDFs from `PDFs/` to `Extracted_Images/`):
```bash
python extract_pdf_images.py
```

**Extract single PDF to default location**:
```bash
python extract_pdf_images.py path/to/file.pdf
```

**Extract single PDF to custom folder**:
```bash
python extract_pdf_images.py path/to/file.pdf path/to/output/folder
```

**Batch extraction from custom PDF folder**:
```bash
python extract_pdf_images.py --batch path/to/pdf/folder
```

**Batch extraction with custom output folder**:
```bash
python extract_pdf_images.py --batch path/to/pdf/folder --output-folder path/to/output
```

**Output**: Creates subfolder `Extracted_Images/1/`, `Extracted_Images/2/`, etc. with page images (e.g., `pdf1_0.png`, `pdf1_1.png`)

---

### 2. Dataset Generation

#### Basic Commands

**Generate full dataset** (all documents, 5 iterations per doc, train-only):
```bash
python dataset_generator.py
```

**Process limited number of documents**:
```bash
python dataset_generator.py --limit 10
```
- Only processes first 10 documents
- Useful for quick testing or small dataset generation

**Enable debug mode** (save intermediate images):
```bash
python dataset_generator.py --debug
```
- Saves 7 intermediate images per document for first 5 documents
- Output: `SynDoc_Wild_v1/debug/` with processing pipeline visualization
- Use to inspect: resizing, warping, texturing, shadow compositing, degradation

#### Advanced Commands

**Specify random seed for reproducibility**:
```bash
python dataset_generator.py --seed 42
```
- Generates identical dataset across runs with same seed
- Default: `None` (random each run)

**Set training iterations per document**:
```bash
python dataset_generator.py --iterations 3
```
- Default: 5 variations per document
- Each iteration uses different random textures, shadows, backgrounds
- Higher values = larger training dataset (3 × 1022 docs = 3066 samples)

**Create test/train split**:
```bash
python dataset_generator.py --test-split 0.2
```
- 20% of documents → `test/` split
- 80% of documents → `train/` split
- Creates separate input/target/mask triplets for each split

**Set test iterations (different from training)**:
```bash
python dataset_generator.py --test-split 0.2 --test-iterations 2
```
- Train: 5 iterations per document (default)
- Test: 2 iterations per document
- Useful for creating smaller test sets while maintaining large training data

#### Combined Examples

**Small dataset with reproducibility**:
```bash
python dataset_generator.py --limit 50 --seed 42
```
- Process 50 documents deterministically
- Generate 250 training samples (50 docs × 5 iterations)

**Dataset with train/test split**:
```bash
python dataset_generator.py --test-split 0.2 --iterations 5 --test-iterations 3
```
- 80% documents (816 of 1022) → `train/`: 4080 samples (816 × 5)
- 20% documents (206 of 1022) → `test/`: 618 samples (206 × 3)
- Total: 4698 samples

**Full featured with all options**:
```bash
python dataset_generator.py --limit 100 --seed 12345 --iterations 3 --test-split 0.25 --test-iterations 2 --debug
```
- Limit: 100 documents (75 train, 25 test)
- Reproducible: seed 12345
- Train: 225 samples (75 × 3), Test: 50 samples (25 × 2)
- Debug: Save intermediate images for first 5 documents

---

### Output Structure

**Training dataset** (`SynDoc_Wild_v1/train/`):
```
train/
├── input/0001.png to input/NNNN.png        # Shadowed + degraded images
├── target/0001.png to target/NNNN.png      # Clean ground truth
└── mask/0001.png to mask/NNNN.png          # Binary shadow masks
```

**Test dataset** (created when using `--test-split`, `SynDoc_Wild_v1/test/`):
```
test/
├── input/0001.png to input/MMMM.png        # Test shadowed + degraded
├── target/0001.png to target/MMMM.png      # Test ground truth
└── mask/0001.png to mask/MMMM.png          # Test shadow masks
```

**Debug output** (created when using `--debug`, `SynDoc_Wild_v1/debug/`):
```
debug/
├── doc_0_000_resized.png                    # After A4 resize
├── doc_0_001_warped.png                     # After perspective warp
├── doc_0_002_textured_target.png            # Paper texture applied (TARGET)
├── doc_0_003_shadow_applied.png             # Shadow overlay before compositing
├── doc_0_004_shadowed.png                   # Shadow composited onto document
├── doc_0_005_degraded_input.png             # Camera degradation applied (INPUT)
├── doc_0_006_mask.png                       # Binary shadow mask
└── doc_0_info.txt                           # Processing metadata
```

---

### Command-Line Arguments Reference

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--limit` | `-l` | int | None | Maximum documents to process (None = all) |
| `--iterations` | `-i` | int | 5 | Training iterations per document |
| `--test-split` | `-t` | float | 0.0 | Test set fraction (0.0-1.0) |
| `--test-iterations` | — | int | Same as `--iterations` | Test iterations per document |
| `--seed` | `-s` | int | None | Random seed for reproducibility |
| `--debug` | `-d` | flag | False | Save debug images for first 5 documents |

---

### Common Workflows

**Quick test** (5 documents, debug mode enabled):
```bash
python dataset_generator.py --limit 5 --debug
```

**Reproducible small dataset** (100 docs, fixed seed):
```bash
python dataset_generator.py --limit 100 --seed 42
```

**Production dataset** (all documents, train/test split):
```bash
python dataset_generator.py --test-split 0.1 --iterations 5 --test-iterations 3
```

**Performance benchmarking** (measure speed per document):
```bash
python dataset_generator.py --limit 10 --seed 0
```

**Iterative development** (small dataset, reproducible, debug):
```bash
python dataset_generator.py --limit 20 --seed 42 --debug --iterations 2
```

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
