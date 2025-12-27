"""
Image Processor: Core image manipulation functions for the dataset generator.

This module contains all the image processing functions:
- Resize and pad to A4
- Paper texture blending
- Perspective warping
- Ambient color extraction
- Shadow tinting and compositing
- Camera simulation (blur, noise, compression)
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from config import (
    TARGET_SIZE, TARGET_WIDTH, TARGET_HEIGHT,
    WARP_MIN_OFFSET, WARP_MAX_OFFSET,
    SHADOW_BLUR_MIN, SHADOW_BLUR_MAX,
    SHADOW_OPACITY_MIN, SHADOW_OPACITY_MAX,
    DEFOCUS_BLUR_MIN, DEFOCUS_BLUR_MAX,
    NOISE_SIGMA_MIN, NOISE_SIGMA_MAX,
    JPEG_QUALITY_MIN, JPEG_QUALITY_MAX,
    TEXTURE_BLEND_MIN, TEXTURE_BLEND_MAX
)


def resize_to_a4(image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE,
                 pad_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Resize image to A4 dimensions while preserving aspect ratio.
    
    Process:
    1. Calculate scale to fit image within target (1240x1754)
    2. Resize using LANCZOS4 (high-quality downsampling)
    3. Create white canvas and center resized image
    4. Return exact target dimensions
    
    Args:
        image: Input image (BGR, any size)
        target_size: Target dimensions (width=1240, height=1754)
        pad_color: Color for padding (default white for documents)
        
    Returns:
        Resized and centered image (1240x1754, BGR uint8)
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # Calculate scale to fit within target while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize using high-quality interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create canvas with padding color
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # Center the resized image on canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def apply_paper_texture(document: np.ndarray, texture: np.ndarray,
                        blend_strength: Optional[float] = None) -> np.ndarray:
    """
    Apply paper texture to document using selective multiply blending.
    Blends paper texture onto document to simulate real paper grain and crumples.
    Texture affects lighter areas (text background) more than text itself.
    
    Algorithm:
    1. Resize texture to match document dimensions using high-quality interpolation
    2. Convert both to float (0-1 range) for precise blending mathematics
    3. Apply selective multiply blending:
       - Multiply: output = base * texture
       - Darker texture areas darken document, lighter areas preserve original
       - This naturally preserves text contrast while showing texture crumples
    4. Blend textured result with original based on blend_strength
       - Low blend: subtle texture, text remains crisp
       - High blend: texture clearly visible, crumples prominent
    5. Clip to valid range and convert back to uint8
    
    Args:
        document: Input document image (BGR, uint8, e.g., scanned page)
        texture: Paper texture image (BGR, uint8, showing grain/crumples)
        blend_strength: Blend factor 0.0-1.0 (None = random between TEXTURE_BLEND_MIN/MAX)
                       0.0 = no texture, 1.0 = full multiply effect
        
    Returns:
        Textured document (BGR, uint8) with visible paper grain on light areas,
        text preserved due to natural properties of multiply blending
        
    Physical Effect:
        Clean document 255,255,255 + texture 150,150,150 = 150,150,150 (grain visible)
        Black text 0,0,0 + texture 150,150,150 = 0,0,0 (text perfectly preserved)
        This is why multiply blending is ideal for preserving document content
    """
    # Resize texture to match document dimensions using high-quality interpolation
    texture_resized = cv2.resize(texture, (document.shape[1], document.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)
    
    # Random blend strength if not specified (ensures variation in dataset)
    if blend_strength is None:
        blend_strength = np.random.uniform(TEXTURE_BLEND_MIN, TEXTURE_BLEND_MAX)
    
    # Convert to float (0-1 range) for precise blending calculations
    doc_float = document.astype(np.float32) / 255.0
    tex_float = texture_resized.astype(np.float32) / 255.0
    
    # Apply multiply blending per channel
    # Multiply naturally preserves text because black*anything=black
    result = np.zeros_like(doc_float)
    
    for i in range(3):
        # Multiply: output = base * texture
        # White areas (255) get multiplied by texture, showing grain clearly
        # Black areas (0) stay black regardless of texture, preserving text perfectly
        multiplied = doc_float[:, :, i] * tex_float[:, :, i]
        
        # Blend between original and textured version based on blend_strength
        # blend_strength controls how visible the texture/crumples are
        result[:, :, i] = doc_float[:, :, i] * (1 - blend_strength) + multiplied * blend_strength
    
    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)


def generate_warp_points(width: int, height: int,
                         min_offset: int = WARP_MIN_OFFSET,
                         max_offset: int = WARP_MAX_OFFSET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate perspective warp points for realistic camera angle simulation.
    
    Creates trapezoid effect by offsetting image corners to simulate:
    - Phone camera held at angle over document
    - Document scanned from handheld device (not perfectly perpendicular)
    - Real-world capture conditions (not perfect scanner)
    
    Warp Effect:
    - Source: Perfect rectangle (scanner/document image)
    - Destination: Trapezoid (simulating camera perspective)
    - Top edges move inward → perspective narrowing due to camera angle
    - Creates natural looking document captures
    
    Args:
        width: Image width in pixels (1240 for A4)
        height: Image height in pixels (1754 for A4)
        min_offset: Minimum corner displacement in pixels (30 = subtle)
        max_offset: Maximum corner displacement in pixels (60 = aggressive)
        
    Returns:
        Tuple of (src_points, dst_points) as float32 arrays
        - src_points: Original rectangle corners
        - dst_points: Warped trapezoid corners
        Ready for cv2.getPerspectiveTransform()
    """
    # Source points: corners of the perfect rectangle (original image)
    src_points = np.float32([
        [0, 0],                    # Top-left
        [width - 1, 0],            # Top-right
        [width - 1, height - 1],   # Bottom-right
        [0, height - 1]            # Bottom-left
    ])
    
    # Destination points: trapezoid corners (warped perspective)
    # Random offsets for each corner create varied warp effects
    # Generate with minimum guarantee, and validate result
    offsets = np.random.randint(min_offset, max_offset + 1, size=(4, 2))
    
    dst_points = src_points.copy()
    
    # Top-left: can move right and down
    dst_points[0, 0] += offsets[0, 0]
    dst_points[0, 1] += offsets[0, 1]
    
    # Top-right: can move left and down
    dst_points[1, 0] -= offsets[1, 0]
    dst_points[1, 1] += offsets[1, 1]
    
    # Bottom-right: can move left and up
    dst_points[2, 0] -= offsets[2, 0]
    dst_points[2, 1] -= offsets[2, 1]
    
    # Bottom-left: can move right and up
    dst_points[3, 0] += offsets[3, 0]
    dst_points[3, 1] -= offsets[3, 1]
    
    # Validate corners have moved (ensure warp will be visible)
    max_movement = np.max(np.abs(dst_points - src_points))
    if max_movement < min_offset * 0.8:
        # If insufficient movement, try again
        return generate_warp_points(width, height, min_offset, max_offset)
    
    return src_points, dst_points


def apply_perspective_warp(image: np.ndarray,
                           src_points: Optional[np.ndarray] = None,
                           dst_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply perspective warp to simulate phone camera held at angle.
    Crops white borders aggressively from the warped result.
    Resizes output back to A4 (1240×1754) to maintain consistent dimensions.
    
    Process:
    1. Generate or use provided corner point mappings
    2. Calculate perspective transformation matrix
    3. Apply warp using OpenCV perspective transform
    4. Detect and crop white borders from warped image
    5. Resize back to A4 dimensions
    5. Return cropped warped image with minimal white borders
    
    Args:
        image: Input image to warp
        src_points: Source corner points, None to generate randomly
        dst_points: Destination corner points, None to generate randomly
        
    Returns:
        Tuple of (warped_cropped_resized_image, src_points, dst_points)
        Image with perspective warp applied, white borders cropped, and resized to A4
    """
    h, w = image.shape[:2]
    
    if src_points is None or dst_points is None:
        src_points, dst_points = generate_warp_points(w, h)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply warp with white border fill
    warped = cv2.warpPerspective(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    # Crop white borders from warped image - BALANCED
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Find non-white areas (< 245 for aggressive white removal)
    non_white_mask = gray < 245
    
    # Get rows and columns with non-white pixels
    rows = np.where(non_white_mask.any(axis=1))[0]
    cols = np.where(non_white_mask.any(axis=0))[0]
    
    if len(rows) > 0 and len(cols) > 0:
        # Get document boundaries
        y_min = rows[0]
        y_max = rows[-1] + 1
        x_min = cols[0]
        x_max = cols[-1] + 1
        
        # Scan from edges and remove rows/cols with too much white
        # Remove rows/columns that are > 50% white - decreased cropping to preserve more warped content
        
        # Remove white rows from top
        for y in range(y_min, y_max):
            width = x_max - x_min
            if width > 0:
                white_pct = np.sum(gray[y, x_min:x_max] >= 245) / width * 100
                if white_pct > 50:
                    y_min = y + 1
                else:
                    break
        
        # Remove white rows from bottom
        for y in range(y_max - 1, y_min - 1, -1):
            width = x_max - x_min
            if width > 0:
                white_pct = np.sum(gray[y, x_min:x_max] >= 245) / width * 100
                if white_pct > 50:
                    y_max = y
                else:
                    break
        
        # Remove white columns from left
        for x in range(x_min, x_max):
            height = y_max - y_min
            if height > 0:
                white_pct = np.sum(gray[y_min:y_max, x] >= 245) / height * 100
                if white_pct > 50:
                    x_min = x + 1
                else:
                    break
        
        # Remove white columns from right
        for x in range(x_max - 1, x_min - 1, -1):
            height = y_max - y_min
            if height > 0:
                white_pct = np.sum(gray[y_min:y_max, x] >= 245) / height * 100
                if white_pct > 50:
                    x_max = x
                else:
                    break
        
        # Crop the warped image
        crop_h = y_max - y_min
        crop_w = x_max - x_min
        if crop_w > 100 and crop_h > 100:
            warped = warped[y_min:y_max, x_min:x_max]
    
    # Resize back to A4 dimensions to maintain consistency
    warped = cv2.resize(warped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
    return warped, src_points, dst_points


def extract_ambient_color(background: np.ndarray) -> np.ndarray:
    """
    Extract average color from background image.
    
    This color represents the ambient environment:
    - Wood desk → brownish shadows
    - White desk → blueish shadows
    - Carpet → reddish shadows
    
    Used to tint all shadows for physical realism.
    
    Args:
        background: Background image (BGR)
        
    Returns:
        Average color as uint8 array (B, G, R)
    """
    # Resize to small size for faster computation
    small = cv2.resize(background, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Calculate mean color
    mean_color = cv2.mean(small)[:3]
    
    return np.array(mean_color, dtype=np.uint8)


def soften_shadow(shadow: np.ndarray, blur_kernel: Optional[int] = None) -> np.ndarray:
    """
    Apply Gaussian blur to soften shadow edges.
    
    Softness simulates light type:
    - Sharp edges (5px): Direct sunlight
    - Soft edges (61px): Diffuse office lighting
    
    Args:
        shadow: Shadow overlay (BGRA)
        blur_kernel: Kernel size in pixels (odd), None for random between 5-61
        
    Returns:
        Softened shadow (BGRA)
    """
    if blur_kernel is None:
        # Random kernel size (must be odd)
        blur_kernel = np.random.randint(SHADOW_BLUR_MIN // 2, SHADOW_BLUR_MAX // 2 + 1) * 2 + 1
    
    # Ensure odd kernel
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    
    # Apply Gaussian blur to all channels including alpha
    blurred = cv2.GaussianBlur(shadow, (blur_kernel, blur_kernel), 0)
    
    return blurred


def tint_shadow(shadow: np.ndarray, opacity: Optional[float] = None,
                ambient_color: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply opacity adjustment and ambient color tinting to shadow overlay.
    
    UPDATED: Shadows are now tinted with ambient environment color to simulate
    real-world physics where shadows are filled by secondary light sources.
    
    Process:
    1. Get alpha channel from shadow (defines shadow shape and intensity)
    2. Tint RGB with ambient color (simulates ambient light filling shadow)
    3. Apply opacity to alpha channel to control shadow strength
    4. Return BGRA shadow ready for compositing
    
    Args:
        shadow: Shadow overlay (BGRA, uint8) - typically black with alpha gradient
        opacity: Shadow opacity factor 0-1 (None = random between SHADOW_OPACITY_MIN/MAX)
                 0.3 = very faint shadow, 0.8 = dark prominent shadow
        ambient_color: Environment color (B, G, R) 0-255, None = keep shadow as-is
        
    Returns:
        Shadow with alpha adjusted and RGB tinted (BGRA, uint8)
    """
    if opacity is None:
        opacity = np.random.uniform(SHADOW_OPACITY_MIN, SHADOW_OPACITY_MAX)
    
    # Create output
    result = shadow.copy().astype(np.float32)
    
    # Get alpha channel (0-1 range) - this defines the shadow shape
    alpha = shadow[:, :, 3].astype(np.float32) / 255.0
    
    # Optionally tint RGB with ambient color
    if ambient_color is not None:
        # Convert ambient color from 0-1 to 0-255 if needed
        ambient_color = ambient_color.astype(np.float32)
        if ambient_color.max() <= 1.0:
            # Normalized 0-1 range, convert to 0-255
            ambient_color_uint8 = (ambient_color * 255.0).astype(np.float32)
        else:
            # Already in 0-255 range
            ambient_color_uint8 = ambient_color
        
        # Blend shadow RGB toward ambient color (creates colored shadows)
        for i in range(3):
            result[:, :, i] = ambient_color_uint8[i]
    
    # Apply opacity to alpha channel to control shadow strength
    result[:, :, 3] = alpha * opacity * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


def composite_shadow(document: np.ndarray, shadow: np.ndarray, 
                     opacity: Optional[float] = None,
                     ambient_color: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Composite physically accurate colored shadow onto document.
    
    CRITICAL FIX: Real shadows are NOT pure black. They are lit by ambient light
    (sky, walls, floor reflections), creating color tints. A hand shadow under a
    warm desk lamp appears blue/purple because the shadow is filled with cool
    ambient light while the lit area is warm.
    
    Algorithm:
    1. Extract shadow shape (alpha channel)
    2. Create shadow color by blending ambient environment color
    3. Apply darkening + color shift to simulate real shadow physics
    4. Return shadowed document, binary mask, and resized shadow for debug
    
    Shadow Effect Details:
    - Shadow color = ambient_fill_strength * ambient_color (not pure black)
    - Darkening effect = multiply blend (reduces brightness)
    - Color tinting = additive ambient fill (simulates secondary light source)
    - Result: Blue shadows under warm light, warm shadows under cool light
    
    Args:
        document: Base document image (BGR, uint8)
        shadow: Shadow with alpha channel (BGRA, uint8)
        opacity: Shadow opacity factor 0-1 (None = random between SHADOW_OPACITY_MIN/MAX)
        ambient_color: Background ambient color (B, G, R) 0-255, None = extract from shadow RGB
        
    Returns:
        Tuple of (shadowed_document, binary_mask, shadow_resized_for_debug)
        - shadowed_document: document with colored shadow (BGR, uint8)
        - binary_mask: where shadow visible = 255 (white), else = 0 (black) (grayscale, uint8)
        - shadow_resized: the shadow that was actually applied (for debug comparison)
    """
    from config import SHADOW_AMBIENT_FILL_MIN, SHADOW_AMBIENT_FILL_MAX
    
    # Random opacity if not specified
    if opacity is None:
        opacity = np.random.uniform(SHADOW_OPACITY_MIN, SHADOW_OPACITY_MAX)
    
    # Resize shadow to match document dimensions using quality interpolation
    shadow_resized = cv2.resize(shadow, (document.shape[1], document.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
    
    # Extract alpha channel (shadow shape)
    alpha = shadow_resized[:, :, 3].astype(np.float32) / 255.0
    
    # Apply opacity to alpha
    alpha_with_opacity = alpha * opacity
    
    # Convert document to float
    doc_float = document.astype(np.float32)
    
    # Determine ambient fill color (simulates secondary light filling the shadow)
    if ambient_color is None:
        # Use neutral gray as fallback
        ambient_color_normalized = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        # Convert ambient color to 0-255 range if needed
        ambient_color = ambient_color.astype(np.float32)
        # Check if already normalized (0-1 range) or uint8 (0-255 range)
        if ambient_color.max() <= 1.0:
            # Already normalized 0-1
            ambient_color_normalized = ambient_color
            ambient_color_uint8 = (ambient_color * 255).astype(np.float32)
        else:
            # Convert from 0-255 to 0-1
            ambient_color_normalized = ambient_color / 255.0
            ambient_color_uint8 = ambient_color
    
    # Random ambient fill strength (how much secondary light fills shadow)
    ambient_fill_strength = np.random.uniform(SHADOW_AMBIENT_FILL_MIN, SHADOW_AMBIENT_FILL_MAX)
    
    # PHYSICAL MODEL: Shadow = Darkening + Ambient Color Fill
    result = np.zeros_like(doc_float)
    
    for i in range(3):
        # Step 1: Darken the document (shadow blocks primary light)
        darkened = doc_float[:, :, i] * (1 - alpha_with_opacity * 0.7)
        
        # Step 2: Add ambient color fill (shadow lit by secondary light source)
        # This creates the blue/purple/warm tints seen in real shadows
        # Use uint8 range (0-255) for the ambient color contribution
        if ambient_color.max() <= 1.0:
            ambient_contribution = (ambient_color_uint8[i] * alpha_with_opacity * ambient_fill_strength)
        else:
            ambient_contribution = (ambient_color[i] * alpha_with_opacity * ambient_fill_strength)
        
        # Combine: darkened base + ambient fill
        result[:, :, i] = darkened + ambient_contribution
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Create binary mask (pure white and black) representing shadow presence
    # This mask is identical to the original shadow overlay's alpha channel
    # where shadow exists = 255 (white), no shadow = 0 (black)
    
    shadow_strength = alpha_with_opacity  # Direct mapping of alpha to shadow presence
    
    # Threshold to create binary mask (0-255 only)
    # If shadow effect > 5%, mark as shadow (255), else background (0)
    mask = np.where(shadow_strength > 0.05, 255, 0).astype(np.uint8)
    
    return result, mask, shadow_resized


def apply_defocus_blur(image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
    """
    Apply Gaussian blur to simulate lens defocus and camera aberration.
    
    Physical Motivation:
    - Cheap smartphone cameras: soft focus at close range (depth of field limited)
    - Budget lenses: spherical aberration causes subtle blur
    - Real documents scanned with phone: always slightly out of focus
    
    Blur Effect:
    - Blurs entire image uniformly
    - Text remains readable but loses crisp edges
    - Shadows become softer, more natural looking
    - Combined with noise, creates realistic capture effect
    
    Args:
        image: Input image (BGR, uint8)
        kernel_size: Blur kernel size in pixels (3-7 for DEFOCUS_BLUR_MIN/MAX)
                    - 3 = 1 pixel blur (very slight)
                    - 5 = 2 pixel blur (noticeable)
                    - 7+ = 3 pixel blur (obvious)
                    None = random within config range
        
    Returns:
        Blurred image (BGR, uint8) with softened features
    """
    if kernel_size is None:
        kernel_size = np.random.randint(DEFOCUS_BLUR_MIN, DEFOCUS_BLUR_MAX + 1)
    
    # Skip if kernel is 0 or 1
    if kernel_size <= 1:
        return image
    
    # Ensure odd kernel for cv2.GaussianBlur
    kernel_size = kernel_size * 2 - 1
    
    # Apply Gaussian blur with adaptive sigma (larger kernel = larger sigma)
    sigma = kernel_size / 2.0
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def add_noise(image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    Add physically accurate intensity-dependent (heteroscedastic) Gaussian noise.
    
    CRITICAL FIX: Real camera sensors have VARIABLE Signal-to-Noise Ratio (SNR).
    Dark areas (shadows) have fewer photons, resulting in much higher noise relative
    to the signal. Bright areas (white paper) have many photons, cleaner signal.
    
    Physical Motivation:
    - Photon shot noise: sqrt(N) noise where N = photon count (Poisson process)
    - Dark areas (low N): High relative noise (grainy)
    - Bright areas (high N): Low relative noise (smooth)
    - Sensor readout noise: constant but visible only in dark areas
    
    Noise Model:
    - Base Gaussian noise scaled by darkness: noise_strength = sigma * (1 - brightness)
    - Where brightness = pixel_value / 255
    - Dark pixels (0): Maximum noise (1.5x sigma)
    - Bright pixels (255): Minimum noise (0.2x sigma)
    
    Args:
        image: Input image (BGR, uint8)
        sigma: Base noise standard deviation (6-8 for NOISE_SIGMA_MIN/MAX)
               - Actual noise per pixel varies by intensity
               None = random within config range
        
    Returns:
        Image with intensity-dependent noise (BGR, uint8)
        
    Effect:
        - Shadows appear very grainy (realistic low-light noise)
        - White paper appears clean (realistic well-lit signal)
        - Model learns to handle variable noise levels
    """
    if sigma is None:
        sigma = np.random.uniform(NOISE_SIGMA_MIN, NOISE_SIGMA_MAX)
    
    # Convert to float for noise calculation
    img_float = image.astype(np.float32) / 255.0
    
    # Calculate noise mask: darker pixels get more noise (inverse of brightness)
    # Dark (0.0) → noise_mask = 1.0 → full noise
    # Bright (1.0) → noise_mask = 0.0 → minimal noise
    noise_mask = 1.0 - img_float
    
    # Generate base Gaussian noise
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    
    # Scale noise by darkness (heteroscedastic noise)
    # Multiply by 1.5 to make shadow noise prominent, add 0.2 base for lit areas
    scaled_noise = noise * (noise_mask * 1.5 + 0.2)
    
    # Add noise to image
    noisy = image.astype(np.float32) + scaled_noise
    
    # Clip to valid uint8 range [0, 255]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_jpeg_artifacts(image: np.ndarray, quality: Optional[int] = None) -> np.ndarray:
    """
    Add JPEG compression artifacts to simulate lossy compression.
    
    Physical Motivation:
    - Smartphone cameras save in JPEG format (lossy compression)
    - Budget phones use lower quality levels to save storage
    - JPEG artifacts: blocking, color banding, detail loss
    - Text edges become softer due to DCT-based compression
    - Shadows show more artifacts (dark areas compress worse)
    
    JPEG Encoding:
    1. Convert to YCbCr color space
    2. Downsample Cb/Cr channels (loss of color detail)
    3. Apply Discrete Cosine Transform (DCT)
    4. Quantize coefficients (controlled by quality)
    5. Encode and decode back to RGB
    
    Args:
        image: Input image (BGR, uint8)
        quality: JPEG quality factor (50-70 for JPEG_QUALITY_MIN/MAX)
                 - 100 = lossless (effectively)
                 - 70 = good quality, slight artifacts
                 - 50 = poor quality, obvious artifacts
                 None = random within config range
        
    Returns:
        Image with JPEG compression artifacts (BGR, uint8)
        
    Effect:
        - Text edges lose sharpness
        - Smooth gradients show banding
        - Color fidelity reduced
        - Shadows appear blockier
    """
    if quality is None:
        quality = np.random.randint(JPEG_QUALITY_MIN, JPEG_QUALITY_MAX + 1)
    
    # Encode to JPEG in memory with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    
    # Decode back to simulate real camera encoding/decoding
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    return decoded


def extract_ambient_color_from_background(background: np.ndarray) -> np.ndarray:
    """
    Extract average ambient color from background image.
    
    Used to tint the degraded image with environmental lighting/white balance.
    
    Args:
        background: Background image (BGR, uint8)
        
    Returns:
        Average color as float32 array (B, G, R) normalized 0-1
    """
    # Resize to small size for faster computation
    small = cv2.resize(background, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Calculate mean color
    mean_color = small.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
    
    return mean_color


def apply_ambient_tint(image: np.ndarray, ambient_color: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Apply ambient color tinting to simulate environmental lighting in background.
    
    Physical motivation:
    - Background lighting affects document appearance
    - Warm lighting (wood desk): adds brownish/reddish tint
    - Cool lighting (white desk): adds blueish tint
    - Neutral (daylight): minimal tint
    
    Args:
        image: Input image (BGR, uint8)
        ambient_color: Average ambient color (B, G, R) normalized 0-1
        strength: Tinting strength 0-1 (0 = no tint, 1 = full ambient color)
        
    Returns:
        Tinted image (BGR, uint8)
    """
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Create tint by blending with ambient color
    # For each pixel: tinted = original * (1 - strength) + ambient_color * strength
    tinted = img_float * (1 - strength) + ambient_color.reshape(1, 1, 3) * strength
    
    # Clip and convert back
    return np.clip(tinted * 255, 0, 255).astype(np.uint8)


def apply_camera_degradation(image: np.ndarray,
                              ambient_color: Optional[np.ndarray] = None,
                              blur_kernel: Optional[int] = None,
                              noise_sigma: Optional[float] = None,
                              jpeg_quality: Optional[int] = None) -> np.ndarray:
    """
    Apply complete camera ISP (Image Signal Processor) degradation pipeline.
    Simulates real smartphone camera effects on document capture.
    
    ISP Pipeline Order (CRITICAL - must be in this order):
    1. Defocus Blur: Simulates lens aberration and soft focus
    2. Gaussian Noise: Adds photon shot and sensor readout noise
    3. JPEG Compression: Encodes with lossy compression (main artifact source)
    
    Physical Justification:
    - Real cameras: sensor captures light → noise injection → ISP processing → JPEG encode
    - Blur BEFORE noise: blurred pixels add noise naturally
    - Noise BEFORE JPEG: JPEG compresses noisy data, creating artifacts
    - This order creates realistic degradation matching real cameras
    
    Effect by Stage:
    - Blur: Softens text edges, makes shadows edges gradual
    - Noise: Grainy texture especially in shadows, added to blur image
    - JPEG: Blocking artifacts, color loss, banding (especially in dark areas)
    
    Args:
        image: Input image (BGR, uint8)
        ambient_color: Ambient color from background (B, G, R) 0-1, None = no tint
        blur_kernel: Defocus kernel size (3-7), None for random
        noise_sigma: Gaussian noise sigma (4-10), None for random
        jpeg_quality: JPEG quality (50-70), None for random
        
    Returns:
        Degraded image (BGR, uint8) simulating real smartphone capture
    """
    # Step 1: Defocus blur (lens aberration)
    result = apply_defocus_blur(image, blur_kernel)
    
    # Step 2: Add Gaussian noise (sensor noise)
    # Note: noise naturally appears stronger in dark areas (shadows) due to lower SNR
    result = add_noise(result, noise_sigma)
    
    # Step 3: JPEG compression (final lossy encoding)
    result = add_jpeg_artifacts(result, jpeg_quality)
    
    # Step 4: Apply ambient color tinting (simulate environmental lighting)
    if ambient_color is not None:
        result = apply_ambient_tint(result, ambient_color, strength=0.1)
    
    return result


def create_ground_truth(document: np.ndarray, texture: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create warped and textured ground truth (target) image.
    
    Args:
        document: Clean document image
        texture: Paper texture image
        
    Returns:
        Tuple of (textured_warped_doc, src_points, dst_points)
    """
    # Step 1: Resize to A4
    resized = resize_to_a4(document)
    
    # Step 2: Apply paper texture
    textured = apply_paper_texture(resized, texture)
    
    # Step 3: Apply perspective warp
    warped, src_pts, dst_pts = apply_perspective_warp(textured)
    
    return warped, src_pts, dst_pts


def create_input_image(ground_truth: np.ndarray, shadow: np.ndarray,
                       ambient_color: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create degraded input image with shadow and camera effects.
    
    Args:
        ground_truth: Clean warped document (target)
        shadow: Shadow overlay (BGRA)
        ambient_color: Ambient color for shadow tinting (B, G, R) 0-255
        
    Returns:
        Tuple of (degraded_input, binary_mask)
    """
    # Step 1: Soften shadow
    softened = soften_shadow(shadow)
    
    # Step 2: Tint shadow with ambient color
    tinted = tint_shadow(softened, ambient_color=ambient_color)
    
    # Step 3: Composite shadow onto document with ambient tinting
    shadowed, mask, _ = composite_shadow(ground_truth, tinted, ambient_color=ambient_color)
    
    # Step 4: Apply camera degradation with ambient tinting
    degraded = apply_camera_degradation(shadowed, ambient_color=ambient_color)
    
    return degraded, mask


if __name__ == "__main__":
    # Quick test of individual functions
    print("Testing image processor functions...")
    
    # Create test image
    test_img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "Test Document", (100, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Test resize
    resized = resize_to_a4(test_img)
    print(f"Resized: {test_img.shape} -> {resized.shape}")
    
    # Test warp points
    src, dst = generate_warp_points(TARGET_WIDTH, TARGET_HEIGHT)
    print(f"Warp src:\n{src}")
    print(f"Warp dst:\n{dst}")
    
    # Test ambient color
    fake_bg = np.full((100, 100, 3), (45, 30, 15), dtype=np.uint8)
    ambient = extract_ambient_color(fake_bg)
    print(f"Ambient color: {ambient}")
    
    print("\n✅ Image processor test passed!")
