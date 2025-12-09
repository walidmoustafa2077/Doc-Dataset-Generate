"""
Configuration constants for Shadow Removal Dataset Generator.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent

# Input paths
EXTRACTED_IMAGES_DIR = BASE_DIR / "Extracted_Images"
PAPER_TEXTURE_DIR = BASE_DIR / "Paper Texture"
SHADOW_OVERLAYS_DIR = BASE_DIR / "Shadow Overlays"
BACKGROUND_DIR = BASE_DIR / "Background"

# Output paths
OUTPUT_DIR = BASE_DIR / "SynDoc_Wild_v1"
TRAIN_DIR = OUTPUT_DIR / "train"
INPUT_DIR = TRAIN_DIR / "input"
TARGET_DIR = TRAIN_DIR / "target"
MASK_DIR = TRAIN_DIR / "mask"
TEST_DIR = OUTPUT_DIR / "test"
TEST_INPUT_DIR = TEST_DIR / "input"
TEST_TARGET_DIR = TEST_DIR / "target"
TEST_MASK_DIR = TEST_DIR / "mask"

# =============================================================================
# IMAGE SETTINGS
# =============================================================================
# Target A4 size at ~150 DPI (width x height)
TARGET_WIDTH = 1240
TARGET_HEIGHT = 1754
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

# =============================================================================
# WARPING PARAMETERS
# =============================================================================
# Perspective warp: random corner offset range (pixels)
# Increased for more noticeable camera angle effect (trapezoid distortion)
WARP_MIN_OFFSET = 100
WARP_MAX_OFFSET = 150

# =============================================================================
# SHADOW PARAMETERS
# =============================================================================
# Gaussian blur kernel range for shadow softening (must be odd)
SHADOW_BLUR_MIN = 5
SHADOW_BLUR_MAX = 15

# Shadow opacity range (0.0 = invisible, 1.0 = fully opaque)
SHADOW_OPACITY_MIN = 0.6
SHADOW_OPACITY_MAX = 0.9

# =============================================================================
# CAMERA SIMULATION PARAMETERS
# =============================================================================
# Defocus blur kernel range (simulates lens aberration)
DEFOCUS_BLUR_MIN = 2
DEFOCUS_BLUR_MAX = 4

# Gaussian noise sigma range
NOISE_SIGMA_MIN = 6
NOISE_SIGMA_MAX = 8

# JPEG compression quality range (60-85, lower = more artifacts)
JPEG_QUALITY_MIN = 80
JPEG_QUALITY_MAX = 90

# =============================================================================
# TEXTURE BLENDING
# =============================================================================
# Paper texture blend strength (0.0 = no texture, 1.0 = full multiply)
# Increased further to make texture clearly visible on white areas
TEXTURE_BLEND_MIN = 0.6
TEXTURE_BLEND_MAX = 0.8

# =============================================================================
# PROCESSING OPTIONS
# =============================================================================
# Random seed for reproducibility (set to None for random each run)
RANDOM_SEED = None

# Number of worker threads for parallel processing (increase for faster generation)
# Recommended: 4-8 for most systems, adjust based on CPU cores available
NUM_WORKERS = 8

# Save debug images for first N documents (0 = disabled)
DEBUG_SAVE_COUNT = 0

# =============================================================================
# OUTPUT FORMAT
# =============================================================================
# PNG compression level (0-9, higher = smaller file, slower)
PNG_COMPRESSION = 8
