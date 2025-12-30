# SynDoc_Wild: Synthetic Document Shadow Removal Dataset Generator

Generate realistic synthetic document images with shadows, camera degradation, and variations for training deep learning shadow removal models.

## Overview

This project provides **two approaches** for creating training datasets for document shadow removal:

### **Approach 1: Python-Based Synthetic Dataset** (`SynDoc_Wild`)
- Extracting document pages from PDFs (1000+ documents)
- Applying realistic paper textures and perspective warping (simulating handheld camera capture)
- Compositing realistic shadows with ambient color tinting
- Applying camera degradation (blur, noise, JPEG compression)
- Generating triplets: `input` (shadowed), `target` (clean), `mask` (shadow shape)

### **Approach 2: Blender 4.2 LTS Photorealistic Rendering** (`SynDoc_Wild_3D`)
- High-fidelity 3D document rendering with EEVEE-Next renderer
- Procedurally generated shadow casters with Gaussian blur for realism
- Real-time paper texture simulation with normal mapping
- Advanced material system: HSV, brightness/contrast, gamma correction
- Automatic shadow mask generation via emission shader technique
- Support for OBJ format 3D objects as scene clutter
- Multiple rendering iterations per document
- Flexible train/test split generation

**Dataset Output**: 
- Python approach: `SynDoc_Wild/train/` and `test/` with fast iteration
- Blender approach: `SynDoc_Wild_3D/train/` and `test/` with photorealistic quality

## Which Approach Should I Use?

### Use **Python Approach** (SynDoc_Wild) if:
- ✅ You need **fast iteration** (2-3 doc/sec)
- ✅ You want **reproducible results** (deterministic via seed)
- ✅ You need **large-scale dataset generation** (millions of samples)
- ✅ You prefer **simple setup** (pure Python dependencies)
- ✅ You're on a **CPU-only machine** (no GPU required)
- ✅ You need **debug visualization** of processing steps

### Use **Blender Approach** (SynDoc_Wild_3D) if:
- ✅ You need **photorealistic quality** (3D rendering)
- ✅ You want **complex shadows** (physically-based)
- ✅ You need **3D scene composition** (OBJ models for clutter)
- ✅ You have **GPU acceleration** available (10-30x speedup)
- ✅ You prefer **visual verification** (Blender UI)
- ✅ You need **advanced material control** (normal mapping, roughness)

### Use **Both** (Hybrid) if:
- 🎯 You want a **large training set** (Python fast batch) + **small validation set** (Blender quality)
- 🎯 You're doing **model robustness testing** (compare 2D vs 3D rendered shadows)
- 🎯 You need **flexibility** (mix different rendering qualities)

---

## Quick Start

### Prerequisites

- Python 3.8+
- Dependencies: `numpy`, `opencv-python`, `PyMuPDF`, `tqdm`
- **For Blender rendering**: Blender 4.2 LTS (with Python API)
  - Download: [blender.org](https://www.blender.org/download/lts/)
  - Must be configured with Python executable path

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

### 2. Python Dataset Generation (SynDoc_Wild)

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
- Output: `SynDoc_Wild/debug/` with processing pipeline visualization
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
- Default: 1 variations per document
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

### 3. Blender Dataset Generation (SynDoc_Wild_3D)

Requires Blender 4.2 LTS. Execute via Blender's Python API:

#### Basic Commands

**Generate full dataset** (all documents, 1 iteration per doc, train-only):
```bash
blender --background --python blender_generator.py
```

**Process limited number of documents**:
```bash
blender --background --python blender_generator.py -- --limit 10
```
- Renders first 10 documents with 1 iteration each per default
- Saves 30 PNG triplets (10 docs × 3 files: input, target, mask)

**Enable debug rendering with custom iterations**:
```bash
blender --background --python blender_generator.py -- --limit 5 --iterations 2
```
- Renders 5 documents with 2 iterations each
- Output: 30 PNG files (5 docs × 2 iterations × 3 files)

#### Advanced Commands

**Specify rendering iterations per document**:
```bash
blender --background --python blender_generator.py -- --iterations 3
```
- Default: 1 iteration per document
- Each iteration uses different random backgrounds, textures, shadows, light angles
- Higher values = larger, more diverse training dataset

**Create test/train split**:
```bash
blender --background --python blender_generator.py -- --test-split 0.2
```
- 20% of documents → `SynDoc_Wild_3D/test/` split
- 80% of documents → `SynDoc_Wild_3D/train/` split
- Creates separate input/target/mask triplets for each split

**Set test iterations (different from training)**:
```bash
blender --background --python blender_generator.py -- --test-split 0.2 --iterations 5 --test-iterations 2
```
- Train: 5 iterations per document
- Test: 2 iterations per document
- Useful for larger training set with smaller validation set

**Use custom document directory for test-only rendering**:
```bash
blender --background --python blender_generator.py -- --use-test --PATH /path/to/documents --test-iterations 3
```
- Renders ONLY test set from custom document directory
- Useful for rendering specific documents separately
- Output goes to `SynDoc_Wild_3D/test/`

**Set starting filename number**:
```bash
blender --background --python blender_generator.py -- --start-num 1001 --limit 10
```
- Output filenames: 01001.png to 01010.png instead of 00001.png
- Useful for continuing previous renders or custom numbering

#### Combined Examples

**Small photorealistic dataset**:
```bash
blender --background --python blender_generator.py -- --limit 20 --iterations 2
```
- Process 20 documents with 2 iterations each
- Generate 120 PNG files (20 docs × 2 × 3 files)
- Render time: ~5-10 minutes (depending on GPU/CPU)

**Full dataset with train/test split**:
```bash
blender --background --python blender_generator.py -- --test-split 0.15 --iterations 4 --test-iterations 2
```
- 85% documents (870 of 1022) → `train/`: 3480 samples (870 × 4)
- 15% documents (152 of 1022) → `test/`: 456 samples (152 × 2)
- Total: 3936 samples with photorealistic quality

### Output Structure

**Python Training dataset** (`SynDoc_Wild/train/`):
```
train/
├── input/0001.png to input/NNNN.png        # Shadowed + degraded images
├── target/0001.png to target/NNNN.png      # Clean ground truth
└── mask/0001.png to mask/NNNN.png          # Binary shadow masks
```

**Blender Training dataset** (`SynDoc_Wild_3D/train/`):
```
train/
├── input/00001.png to input/NNNNN.png      # Rendered with shadows (5-digit padding)
├── target/00001.png to target/NNNNN.png    # Rendered without shadows (photorealistic clean)
└── mask/00001.png to mask/NNNNN.png        # Binary shadow masks (grayscale)
```

**Test dataset** (created when using `--test-split` or `--use-test`):
```
test/
├── input/00001.png to input/MMMM.png       # Test shadowed renders
├── target/00001.png to target/MMMM.png     # Test clean renders
└── mask/00001.png to mask/MMMM.png         # Test shadow masks
```

**Debug output** (Python only, created with `--debug`, `SynDoc_Wild/debug/`):
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

#### Python Dataset Generator (`dataset_generator.py`)

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--limit` | `-l` | int | None | Maximum documents to process (None = all) |
| `--iterations` | `-i` | int | 5 | Training iterations per document |
| `--test-split` | `-t` | float | 0.0 | Test set fraction (0.0-1.0) |
| `--test-iterations` | — | int | Same as `--iterations` | Test iterations per document |
| `--seed` | `-s` | int | None | Random seed for reproducibility |
| `--debug` | `-d` | flag | False | Save debug images for first 5 documents |

#### Blender Renderer (`blender_generator.py`)

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--limit` | `-l` | int | None | Maximum documents to process (None = all) |
| `--iterations` | `-i` | int | 1 | Rendering iterations per document |
| `--test-split` | — | float | None | Test set fraction (0.0-1.0) |
| `--test-iterations` | — | int | Same as `--iterations` | Test iterations per document |
| `--use-test` | — | flag | False | Enable test-only mode (requires `--PATH`) |
| `--PATH` | — | str | None | Custom document directory path |
| `--start-num` | — | int | 1 | Starting filename number (e.g., 1 = 00001.png) |

---

### Common Workflows

**Quick test - Python** (5 documents, debug mode enabled):
```bash
python dataset_generator.py --limit 5 --debug
```

**Quick test - Blender** (5 documents, 1 iteration):
```bash
blender --background --python blender_generator.py -- --limit 5
```

**Reproducible small dataset - Python** (100 docs, fixed seed):
```bash
python dataset_generator.py --limit 100 --seed 42
```

**Photorealistic small dataset - Blender** (100 docs, 2 iterations):
```bash
blender --background --python blender_generator.py -- --limit 100 --iterations 2
```

**Production dataset - Python** (all documents, train/test split):
```bash
python dataset_generator.py --test-split 0.1 --iterations 5 --test-iterations 3
```

**Production dataset - Blender** (all documents, train/test split):
```bash
blender --background --python blender_generator.py -- --test-split 0.1 --iterations 5 --test-iterations 3
```

**Hybrid approach** (Python for speed, Blender for quality validation):
```bash
# Generate quick Python dataset
python dataset_generator.py --limit 50 --seed 42 --iterations 5

# Generate small Blender dataset for comparison
blender --background --python blender_generator.py -- --limit 5 --iterations 3
```

**Iterative development - Python** (small dataset, reproducible, debug):
```bash
python dataset_generator.py --limit 20 --seed 42 --debug --iterations 2
```

**Continuous rendering - Blender** (render in batches without interruption):
```bash
# Render documents 1-100
blender --background --python blender_generator.py -- --limit 100 --iterations 2 --start-num 1

# Continue with documents 101-200 (using --start-num to continue numbering)
blender --background --python blender_generator.py -- --limit 100 --iterations 2 --start-num 201
```

## Project Structure

```
Doc-Dataset-Generate/
├── config.py                    # Global configuration & parameters
├── asset_manager.py             # Image asset pooling & randomization
├── image_processor.py           # Core image transformation algorithms
├── dataset_generator.py         # Main orchestration pipeline (Python approach)
├── blender_generator.py         # Blender 4.2 LTS photorealistic renderer
├── extract_pdf_images.py        # PDF extraction utility
├── requirements.txt             # Python dependencies
├── CONTRIBUTING.md              # Development guidelines
├── LICENSE                      # MIT License
│
├── PDFs/                        # Input: PDF documents
├── Extracted_Images/            # Intermediate: Extracted PDF pages
├── Paper Texture/               # Asset: Paper grain textures
├── Shadow Overlays/             # Asset: Shadow overlays (Python only)
├── Background/                  # Asset: Background images (wood, carpet, etc.)
├── obj/                         # Asset: OBJ 3D models for Blender clutter
├── SynDoc_Wild/              # Output: Python-generated dataset
│   ├── train/
│   │   ├── input/               # Shadowed + degraded images
│   │   ├── target/              # Clean ground truth
│   │   └── mask/                # Binary shadow masks
│   ├── test/
│   │   └── input/, target/, mask/
│   └── debug/                   # Debug images from --debug mode
└── SynDoc_Wild_3D/           # Output: Blender-rendered dataset
    ├── train/
    │   ├── input/               # Rendered with shadows
    │   ├── target/              # Rendered without shadows
    │   └── mask/                # Shadow masks from emission shader
    └── test/
        └── input/, target/, mask/
```

## Key Features

### Python Approach (SynDoc_Wild)

#### Realistic Shadow Rendering
- Shadows are tinted with **ambient background color** (wood/carpet/white) for photorealism
- Gaussian blur (5-15px) for soft shadow edges
- Random opacity (0.6-0.9) for variation

#### Paper Texture Blending
- **Multiply blending** preserves text contrast while adding grain
- Blend strength varies (0.6-0.8) for dataset diversity

#### Perspective Warp
- Random corner offsets (100-150px) simulate handheld camera angles
- Creates trapezoid distortion (narrowing at top) for realism

#### Camera Degradation Pipeline
1. Defocus blur (2-4px kernel)
2. Gaussian noise (σ=6-8)
3. JPEG compression (quality 80-90)

#### Reproducibility
- Configurable `RANDOM_SEED` for deterministic generation
- Asset selection without repetition (round-robin with shuffle)

---

### Blender Approach (SynDoc_Wild_3D)

#### Photorealistic 3D Rendering (EEVEE-Next)
- Full 3D scene with realistic lighting, materials, and shadows
- High-fidelity document appearance with BRDF shading
- Real-time GPU-accelerated rendering (NVIDIA, AMD, Intel support)

#### Advanced Shadow Simulation
- Procedurally generated shadow casters (2-3 per document)
- Gaussian blur on shadow softening (realistic diffuse shadows)
- Realistic directional sun light with randomized angles
- Shadow jitter and proper depth sampling for physical accuracy

#### Material System
- **Document Enhancement**: HSV (saturation), Brightness/Contrast, Gamma correction
- **Paper Texture**: Multiply blending with normal mapping (bump mapping)
- **Roughness Mapping**: Varies surface roughness based on texture detail
- Support for multiple OBJ format objects in scene for natural clutter

#### Flexible Rendering
- **Target** render: Clean document without shadows or atmospheric effects
- **Input** render: Same scene with shadows and lighting variations
- **Mask** render: High-contrast binary shadow mask via emission shader technique

#### World Lighting
- Ambient color tinting from background images
- Configurable world strength (affects overall scene brightness)
- AgX color space for realistic tone mapping (with fallback to Standard)

---

## Comparison Table

| Feature | Python (Wild_v1) | Blender (Blender_v1) |
|---------|------------------|----------------------|
| **Speed** | ~2-3 sec/doc | ~5-10 sec/doc (GPU dependent) |
| **Realism** | Photographic simulation | Full 3D photorealistic |
| **Texture Detail** | 2D multiply blending | 3D normal mapped surfaces |
| **Shadow Quality** | Procedural overlay | Physically-based ray traced |
| **Reproducibility** | Deterministic via seed | Inherent to scene setup |
| **3D Objects** | None (2D only) | OBJ format supported |
| **Paper Variation** | Asset-based | Material shader-based |
| **Setup Complexity** | Python + dependencies | Python + Blender 4.2 LTS |
| **Parallelization** | ThreadPoolExecutor (8 workers) | Sequential (per-render) |
| **Memory Usage** | ~400MB (8 workers) | ~1-2GB per Blender instance |
| **GPU Required** | No | Yes (recommended for speed) |

## Configuration

### Python Approach (`config.py`)

Edit `config.py` to adjust:

- **Image dimensions**: `TARGET_WIDTH=1240`, `TARGET_HEIGHT=1754` (A4 at 150 DPI)
- **Warp intensity**: `WARP_MIN_OFFSET=100`, `WARP_MAX_OFFSET=150`
- **Shadow opacity**: `SHADOW_OPACITY_MIN=0.6`, `SHADOW_OPACITY_MAX=0.9`
- **Camera blur**: `DEFOCUS_BLUR_MIN=2`, `DEFOCUS_BLUR_MAX=4`
- **Texture visibility**: `TEXTURE_BLEND_MIN=0.6`, `TEXTURE_BLEND_MAX=0.8`
- **Processing**: `NUM_WORKERS=8` (parallel threads), `RANDOM_SEED=None` (reproducibility)

### Blender Approach (Hard-coded in `blender_generator.py`)

Key rendering parameters (adjust in `setup_render_settings()` and related functions):

- **Resolution**: `TARGET_WIDTH=1240`, `TARGET_HEIGHT=1754` (matches Python approach)
- **Render Engine**: `BLENDER_EEVEE_NEXT` (requires Blender 4.2+)
- **TAA Samples**: `eevee.taa_render_samples = 124` (temporal anti-aliasing)
- **Shadow Pool**: `eevee.shadow_pool_size = '256'` (shadow map resolution)
- **Sun Energy**: `random.uniform(1.75, 2.75)` (varies per render)
- **Sun Angle**: `random.uniform(0.02, 0.1)` (shadow softness in radians)
- **Document Scale**: `DOC_HEIGHT=1.414`, `DOC_WIDTH=1.0` (A4 aspect ratio)
- **World Strength**: `0.3` (ambient background contribution)

To modify Blender parameters, edit the relevant functions:
- `setup_render_settings()` - Render engine, resolution, antialiasing
- `setup_camera_and_plane()` - Camera position, background setup, world lighting
- `add_shadow_casters()` - Shadow position, rotation, scale
- `create_document_material()` - HSV saturation, brightness/contrast, gamma

## Dataset Format

Output triplets for machine learning:

```
SynDoc_Wild/train/
├── input/0001.png              # Shadowed + degraded (input to model)
├── target/0001.png             # Clean document (ground truth)
└── mask/0001.png               # Binary shadow mask (supervision signal)
```

- **Sequential naming**: 0001.png to NNNN.png (4-digit zero-padded)
- **PNG lossless**: Compression=8 for smaller files
- **Equal triplets**: Same number of input/target/mask files (n=1000+ in train/)

## Debugging

### Python Debug Mode
```bash
python dataset_generator.py --debug
```

Saves 7 intermediate images per document to `SynDoc_Wild/debug/`:
1. `01_resized.png` - A4 resized document
2. `02_warped.png` - Perspective warp applied
3. `03_textured_target.png` - Paper texture applied (TARGET)
4. `04_shadow_applied.png` - Shadow before compositing
5. `05_shadowed.png` - Shadow composited onto document
6. `06_degraded_input.png` - Camera degradation applied (INPUT)
7. `07_mask.png` - Binary shadow mask

### Blender Rendering Tips

**Verify Blender integration**:
```bash
blender --version
```

**Render with verbose output** (troubleshooting):
```bash
blender --python blender_generator.py -- --limit 1 (no --background flag)
```
This shows Blender UI and console output for debugging material issues.

**Check asset paths** (in `blender_generator.py`):
- Confirm `EXTRACTED_IMAGES_DIR` has documents
- Confirm `BACKGROUND_DIR` has background images
- Confirm `PAPER_TEXTURE_DIR` has texture files
- Optional: `OBJ_DIR` for 3D clutter objects

**GPU Acceleration** (if slow):
- Blender defaults to CPU rendering; enable GPU in Preferences → Render
- CUDA (NVIDIA), HIP (AMD), or OptiX (NVIDIA RTX) all supported
- TAA samples reduced from 124 to 64 for faster iteration during development

| Issue | Solution |
|-------|----------|
| "No textures found" | Add PNG/JPG files to `Paper Texture/` directory |
| "No shadows found" | Add PNG/JPG files to `Shadow Overlays/` directory (Python only) |
| "No backgrounds found" | Add PNG/JPG files to `Background/` directory |
| Slow processing (Python) | Increase `NUM_WORKERS` in config.py (if CPU allows) |
| Memory errors (Python) | Reduce `NUM_WORKERS` or use `--limit` to process in batches |
| Blender: "ModuleNotFoundError" | Ensure Blender's Python environment has cv2, numpy installed |
| Blender: Dark renders | Check world strength in `setup_camera_and_plane()`, increase sun energy |
| Blender: Blurry shadows | Reduce `sun_data.angle` value in `add_shadow_casters()` |
| Blender: Slow rendering | Enable GPU acceleration or reduce TAA samples (124 → 64) |
| Blender: Shadow mask is inverted | Check color ramp inversion logic in `prepare_shadow_only_scene()` |

## Performance

### Python Approach
**Baseline**: ~2-3 document per seconds  on 8-core CPU with `NUM_WORKERS=8`

For 1000 documents:
- Single-threaded: ~30-50 minutes
- 8 workers: ~4-6 minutes

### Blender Approach
**Baseline**: ~5-10 seconds per document (varies by GPU, resolution, TAA samples)

For 1000 documents with 1 iteration:
- Single-threaded (CPU render): ~2-3 hours
- GPU accelerated (NVIDIA/AMD): ~30-50 minutes
- With 2 iterations: Double the times above

**Optimization Tips**:
- Use GPU acceleration (CUDA/HIP) for 10-30x speedup vs CPU
- Reduce TAA samples from 124 to 64 for 2x faster iteration during development
- Use `--limit 10` for test renders before running full batch

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
