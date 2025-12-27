"""
Shadow Removal Dataset Generator - Main Pipeline

This script generates a shadow removal training dataset by:
1. Indexing all document images
2. Creating ground truth (warped + textured documents)
3. Adding realistic shadows with ambient-colored tinting
4. Applying camera degradation (blur, noise, compression)
5. Saving input/target/mask triplets

Usage:
    python dataset_generator.py
    python dataset_generator.py --limit 100  # Process only first 100 docs
    python dataset_generator.py --debug      # Save debug images for first 5 docs
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm

from config import (
    OUTPUT_DIR, TRAIN_DIR, INPUT_DIR, TARGET_DIR, MASK_DIR,
    TARGET_SIZE, TARGET_WIDTH, TARGET_HEIGHT,
    PNG_COMPRESSION, RANDOM_SEED, NUM_WORKERS
)
from asset_manager import AssetManager
from image_processor import (
    resize_to_a4,
    apply_paper_texture,
    apply_perspective_warp,
    composite_shadow,
    apply_camera_degradation,
    extract_ambient_color_from_background
)


class DatasetGenerator:
    """
    Main dataset generation pipeline for shadow removal training data.
    
    Complete workflow:
    1. Index: Scan all documents in Extracted_Images (1022 total)
    2. Loop: For each document:
       a. Load random background, paper texture, shadow overlay
       b. Create TARGET: resize → apply texture → warp (clean ground truth)
       c. Extract ambient color from background
       d. Create INPUT: soften shadow → tint with ambient color → composite → degrade
       e. Create MASK: extract shadow shape (binary)
       f. Save triplet preserving folder structure
    3. Output: Dataset/train/ with input/, target/, mask/ subdirectories
    
    Key features:
    - Realistic shadows tinted with ambient colors (wood, carpet, white desk)
    - Degradation pipeline simulates real camera (blur, noise, JPEG compression)
    - Paper texture and perspective warp add realism
    - Debug mode shows all intermediate processing steps
    """
    
    def __init__(self, seed: Optional[int] = RANDOM_SEED, debug: bool = False, debug_count: int = 5):
        """
        Initialize the dataset generator.
        
        Args:
            seed: Random seed for reproducibility (None = random each run)
            debug: If True, save intermediate images for first debug_count documents
            debug_count: Number of documents to save debug images for (default 5)
            debug_count: Number of debug images to save
        """
        self.seed = seed
        self.debug = debug
        self.debug_count = debug_count
        self.debug_dir = OUTPUT_DIR / "debug"
        
        # Initialize asset manager
        print("=" * 60)
        print("🚀 Shadow Removal Dataset Generator")
        print("=" * 60)
        print()
        
        self.asset_manager = AssetManager(seed=seed)
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.errors = []
    
    def _create_output_dirs(self) -> None:
        """
        Create output directory structure for training dataset.
        
        Creates:
        - Dataset/train/input/: Shadowed + degraded input images
        - Dataset/train/target/: Clean ground truth images
        - Dataset/train/mask/: Binary shadow masks
        - Dataset/debug/: Debug intermediate images (if debug=True)
        """
        print("\n📁 Creating output directories...")
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        TRAIN_DIR.mkdir(exist_ok=True)
        INPUT_DIR.mkdir(exist_ok=True)
        TARGET_DIR.mkdir(exist_ok=True)
        MASK_DIR.mkdir(exist_ok=True)
        
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)
        
        print(f"   Output: {OUTPUT_DIR}")
    
    def _get_output_paths(self, sample_id: int, split: str = "train") -> tuple:
        """
        Generate output paths with sequential naming (0001, 0002, etc).
        
        Maps: sample_id 0 → 0001.png, sample_id 1 → 0002.png, etc.
        Files saved directly without folder structure preservation.
        
        Args:
            sample_id: Sequential sample ID (0-indexed)
            split: Dataset split - "train" or "test"
            
        Returns:
            Tuple of (input_path, target_path, mask_path)
        """
        from config import (
            TEST_INPUT_DIR, TEST_TARGET_DIR, TEST_MASK_DIR
        )
        
        # Generate sequential filename with 4-digit zero-padding
        filename = f"{sample_id + 1:04d}.png"
        
        # Select directories based on split
        if split == "test":
            input_dir = TEST_INPUT_DIR
            target_dir = TEST_TARGET_DIR
            mask_dir = TEST_MASK_DIR
        else:  # train
            input_dir = INPUT_DIR
            target_dir = TARGET_DIR
            mask_dir = MASK_DIR
        
        # Create subdirectories if needed
        input_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = input_dir / filename
        target_path = target_dir / filename
        mask_path = mask_dir / filename
        
        return input_path, target_path, mask_path
    
    def _save_image(self, path: Path, image: np.ndarray) -> None:
        """
        Save image as PNG with lossless compression.
        
        Args:
            path: Output file path
            image: Image to save (BGR or grayscale uint8)
        """
        cv2.imwrite(
            str(path), image,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
        )
    
    def _save_debug_images(self, index: int, doc_path: Path, iteration: int,
                           resized: np.ndarray,
                           warped: np.ndarray,
                           textured: np.ndarray,
                           shadow_applied: np.ndarray,
                           shadowed: np.ndarray,
                           degraded: np.ndarray,
                           mask: np.ndarray) -> None:
        """
        Save intermediate images for debugging and analysis.
        Shows complete pipeline progression.
        
        Debug Images Saved:
        1. 01_resized.png: Document resized to A4 (1240x1754)
        2. 02_warped.png: Perspective warp applied (camera angle distortion)
        3. 03_textured_target.png: Paper texture applied (TARGET image - clean)
        4. 04_shadow_applied.png: Shadow resized to document size (applied shadow, cropped area)
        5. 05_shadowed.png: Shadow composited onto document (darkened, accurate shadow)
        6. 06_degraded_input.png: Camera degradation applied (blur + noise + JPEG)
        7. 07_mask.png: Binary mask of shadow region (for shadow removal training)
        8. info.txt: Metadata for this sample
        
        Process Flow (NEW ORDER):
        Document → [Resize] → [Warp] → [Texture] = TARGET
        Shadow → [Composite with random opacity] → [Degrade] = INPUT
        Shadow Alpha → [Threshold] = MASK
        
        Args:
            index: Document index for organizing debug output
            doc_path: Path to source document
            resized: After A4 resize
            warped: After perspective warp
        Args:
            index: Document index for organizing debug output
            doc_path: Path to source document
            iteration: Variation iteration number (0-4)
            resized: After A4 resize
            warped: After perspective warp
            textured: After texture application (TARGET)
            shadow_applied: The resized shadow that was actually applied (cropped area)
            shadowed: Shadow composited on document
            degraded: After camera degradation (INPUT)
            mask: Binary shadow mask
        """
        debug_subdir = self.debug_dir / f"{index:04d}_{doc_path.stem}_v{iteration}"
        debug_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save each step of the pipeline
        self._save_image(debug_subdir / "01_resized.png", resized)
        self._save_image(debug_subdir / "02_warped.png", warped)
        self._save_image(debug_subdir / "03_textured_target.png", textured)
        self._save_image(debug_subdir / "04_shadow_applied.png", shadow_applied)
        self._save_image(debug_subdir / "05_shadowed.png", shadowed)
        self._save_image(debug_subdir / "06_degraded_input.png", degraded)
        self._save_image(debug_subdir / "07_mask.png", mask)
        
        # Save metadata for analysis
        with open(debug_subdir / "info.txt", "w") as f:
            f.write(f"Document: {doc_path}\n")
    
    def process_single_document(self, sample_id: int, split: str = "train") -> bool:
        """
        Process a single document variation and generate training triplet.
        
        Complete pipeline:
        1. Load RANDOM document from shuffled pool (no repetition per batch)
        2. Load random texture, shadow, background (no repetition per batch)
        3. Create ground truth: resize → texture → warp
        4. Create input: shadow composite → degrade with ambient tint
        5. Create mask from shadow alpha channel
        6. Save triplet with sequential naming (0001, 0002, etc)
        
        Args:
            sample_id: Sequential sample ID for output filename (0 → 0001.png)
            split: Dataset split - "train" or "test"
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load RANDOM document (no repetition within batch)
            doc_path, document = self.asset_manager.get_next_random_document()
            
            # Load random assets (no repetition within batch)
            _, texture = self.asset_manager.get_random_texture()
            _, shadow = self.asset_manager.get_random_shadow()
            _, background = self.asset_manager.get_random_background()
            
            # Extract ambient color from background for environmental tinting
            ambient_color = extract_ambient_color_from_background(background)
            
            
            # =========================================================
            # STEP 1: Create Clean Document (Base)
            # =========================================================
            # This is the clean document with texture and warp but NO shadow/noise
            
            # Resize document to A4 dimensions (1240x1754)
            resized = resize_to_a4(document)
            
            # Apply perspective warp FIRST (simulate camera angle)
            # This warps the original document shape before adding texture
            warped, src_pts, dst_pts = apply_perspective_warp(resized)
            
            # Apply paper texture AFTER warp (texture follows the warped document)
            textured = apply_paper_texture(warped, texture)
            
            # Store textured as our clean base document
            clean_doc = textured.copy()
            
            # =========================================================
            # STEP 2: Create Input with Shadow
            # =========================================================
            # Add physically accurate colored shadows FIRST (before degradation)
            # This ensures shadows are visible under the same camera degradation
            
            # Composite shadow directly onto document with random opacity and ambient color
            # Returns: shadowed document, mask, and resized shadow for debug
            shadowed, mask, shadow_applied = composite_shadow(clean_doc, shadow, ambient_color=ambient_color)
            
            # =========================================================
            # STEP 3: Apply Same Camera Degradation to Both
            # =========================================================
            # Apply identical ISP pipeline to both target and input
            # This ensures they have the same blur, noise, and JPEG compression
            # The ONLY difference is the shadow in the input
            
            # Degrade target (no shadow, but same camera effects)
            target = apply_camera_degradation(clean_doc, ambient_color=ambient_color)
            
            # Degrade input (with shadow + same camera effects)
            degraded = apply_camera_degradation(shadowed, ambient_color=ambient_color)
            
            # This is our INPUT image
            input_image = degraded
            
            # =========================================================
            # STEP 4: Save Output Triplet with Sequential Naming
            # =========================================================
            input_path, target_path, mask_path = self._get_output_paths(sample_id, split=split)
            
            self._save_image(input_path, input_image)
            self._save_image(target_path, target)
            self._save_image(mask_path, mask)
            
            # Save debug images for first documents across all iterations
            if self.debug and sample_id < self.debug_count:
                self._save_debug_images(
                    sample_id, doc_path, 0,
                    resized, warped, textured,
                    shadow_applied,
                    shadowed, degraded, mask
                )
            
            return True
            
        except Exception as e:
            self.errors.append((sample_id, str(e)))
            return False
    
    def run(self, limit: Optional[int] = None, iterations: int = 5, test_split: float = 0.0, test_iterations: Optional[int] = None) -> None:
        """
        Run the complete dataset generation pipeline.
        
        Process:
        1. Create output directory structure
        2. Shuffle all asset pools (documents, textures, shadows, backgrounds)
        3. Generate training dataset: each document gets multiple iterations
           - Loads random different documents (NO repetition across all samples)
           - Each sample has random textures/shadows/backgrounds (NO repetition per batch)
        4. Generate test dataset (if test_split > 0): same iterations as train
           - Loads different random documents from shuffled pool
        5. Sequential naming: 0001, 0002, 0003, etc.
        
        Args:
            limit: Maximum number of documents to process (None = all 1022)
            iterations: Number of iterations per document for training (default 5)
            test_split: Fraction of documents to use for test (0.0-1.0, default 0.0 = no test)
            test_iterations: Number of iterations per document for test (default = same as iterations)
        """
        self._create_output_dirs()
        
        total = self.asset_manager.total_docs
        if limit is not None:
            total = min(total, limit)
        
        # Use same iterations for test if not specified
        if test_iterations is None:
            test_iterations = iterations
        
        # Shuffle all pools once at the start
        self.asset_manager.reshuffle_pools()
        
        # Generate training dataset
        print(f"\n📊 Dataset Generation Plan:")
        print(f"   Total documents available: {total}")
        print(f"   Train iterations per doc: {iterations}")
        train_count = total * iterations
        print(f"   Total train samples: {train_count}")
        print(f"   (All {train_count} samples use DIFFERENT random documents)")
        print(f"   Workers: {NUM_WORKERS} threads")
        
        test_count = 0
        test_docs = 0
        if test_split > 0:
            test_docs = max(1, int(total * test_split))
            test_count = test_docs * test_iterations
            print(f"\n   Test docs: {test_docs} (20% of {total})")
            print(f"   Test iterations per doc: {test_iterations}")
            print(f"   Total test samples: {test_count}")
            print(f"   (All {test_count} test samples use DIFFERENT random documents)")
        
        print("-" * 60)
        
        start_time = time.time()
        
        # =====================================================
        # PHASE 1: Generate Training Dataset
        # =====================================================
        print(f"\n🔄 PHASE 1: Generating TRAIN dataset...")
        print(f"   Processing {NUM_WORKERS} samples in parallel...")
        sample_id = 0
        total_samples = train_count
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            futures = []
            for sample in range(total_samples):
                future = executor.submit(self.process_single_document, sample_id, split="train")
                futures.append(future)
                sample_id += 1
            
            # Process results with progress bar
            for future in tqdm(as_completed(futures), total=total_samples, desc="Training dataset", unit="sample"):
                try:
                    success = future.result()
                    if success:
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                except Exception as e:
                    self.error_count += 1
                    self.errors.append((0, str(e)))
        
        # =====================================================
        # PHASE 2: Generate Test Dataset (if enabled)
        # =====================================================
        if test_split > 0:
            # Reshuffle pools for test set to get different random order
            self.asset_manager.reshuffle_pools()
            print(f"\n🔄 PHASE 2: Generating TEST dataset...")
            print(f"   Processing {NUM_WORKERS} samples in parallel...")
            test_sample_id = 0
            
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Submit all tasks
                futures = []
                for sample in range(test_count):
                    future = executor.submit(self.process_single_document, test_sample_id, split="test")
                    futures.append(future)
                    test_sample_id += 1
                
                # Process results with progress bar
                for future in tqdm(as_completed(futures), total=test_count, desc="Test dataset", unit="sample"):
                    try:
                        success = future.result()
                        if success:
                            self.processed_count += 1
                        else:
                            self.error_count += 1
                    except Exception as e:
                        self.error_count += 1
                        self.errors.append((0, str(e)))
        
        elapsed = time.time() - start_time
        
        # Print summary
        print()
        print("=" * 60)
        print("📊 Generation Complete!")
        print("=" * 60)
        print(f"   Total documents:  {total}")
        print(f"   Train iterations: {iterations}")
        print(f"   Train samples:    {train_count}")
        if test_split > 0:
            print(f"   Test samples:     {test_count}")
            print(f"   Total samples:    {train_count + test_count}")
        print(f"   Processed:        {self.processed_count}")
        print(f"   Errors:           {self.error_count}")
        if self.processed_count > 0:
            print(f"   Time elapsed:     {elapsed:.1f}s ({elapsed/self.processed_count:.2f}s/sample)")
        print(f"   Output directory: {OUTPUT_DIR}")
        print()
        
        # Print errors if any
        if self.errors:
            print("⚠️  Errors encountered:")
            for idx, err in self.errors[:10]:  # Show first 10 errors
                print(f"   - Document {idx}: {err}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more errors")
        
        # Print output structure
        print("\n📁 Output Structure:")
        print(f"   {OUTPUT_DIR}/")
        print(f"   ├── train/")
        print(f"   │   ├── input/    ({train_count} files)")
        print(f"   │   ├── target/   ({train_count} files)")
        print(f"   │   └── mask/     ({train_count} files)")
        if test_split > 0:
            print(f"   └── test/")
            print(f"       ├── input/    ({test_count} files)")
            print(f"       ├── target/   ({test_count} files)")
            print(f"       └── mask/     ({test_count} files)")


def main():
    """
    Main entry point for dataset generation.
    
    Command-line interface:
    - dataset_generator.py                                    : Process all docs, train only
    - dataset_generator.py --limit 100                        : Process first 100 docs
    - dataset_generator.py --debug                            : Show debug images for first 5 docs
    - dataset_generator.py --seed 12345                       : Use specific random seed
    - dataset_generator.py --iterations 3                     : Generate 3 variations per document for train
    - dataset_generator.py --test-split 0.2                   : Use 20% of docs for test set
    - dataset_generator.py --test-iterations 2                : Generate 2 iterations per doc for test
    
    Output:
    - SynDoc_Wild_v1/train/input/   : Shadowed + degraded training inputs (0001.png, 0002.png, etc.)
    - SynDoc_Wild_v1/train/target/  : Clean ground truth for shadow removal
    - SynDoc_Wild_v1/train/mask/    : Binary shadow masks for segmentation
    - SynDoc_Wild_v1/test/input/    : Test set shadowed + degraded inputs (if --test-split > 0)
    - SynDoc_Wild_v1/test/target/   : Test set clean ground truth (if --test-split > 0)
    - SynDoc_Wild_v1/test/mask/     : Test set binary shadow masks (if --test-split > 0)
    
    Key features:
    - All train samples use DIFFERENT random documents (no repetition across all samples)
    - All test samples use DIFFERENT random documents (no repetition across all test samples)
    - Each sample gets random textures, shadows, backgrounds (no repetition per batch)
    
    Examples:
    - python dataset_generator.py                                    (all 1022 docs, 5 train iters, no test)
    - python dataset_generator.py --test-split 0.2                  (20% of docs for test with same iterations)
    - python dataset_generator.py --limit 10 --iterations 2 --test-split 0.2 --test-iterations 3
      (10 docs: 2 train iters per doc, 2 test docs with 3 iters each)
    """
    parser = argparse.ArgumentParser(
        description="Generate shadow removal dataset with train/test splits and no-repetition document loading"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Maximum number of documents to process (default: all 1022)"
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=5,
        help="Number of iterations per document for training (default: 5)"
    )
    parser.add_argument(
        "--test-split", "-t", type=float, default=0.0,
        help="Fraction of documents to use for test set (0.0-1.0, default: 0.0 = no test)"
    )
    parser.add_argument(
        "--test-iterations", type=int, default=None,
        help="Number of iterations per document for test (default: same as --iterations)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Save debug images for first 5 documents (iteration 0 only)"
    )
    
    args = parser.parse_args()
    
    # Validate test_split
    if not 0.0 <= args.test_split <= 1.0:
        print("Error: --test-split must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Run generator
    generator = DatasetGenerator(seed=args.seed, debug=args.debug, debug_count=5)
    generator.run(
        limit=args.limit,
        iterations=args.iterations,
        test_split=args.test_split,
        test_iterations=args.test_iterations
    )


if __name__ == "__main__":
    main()
