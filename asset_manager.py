"""
Asset Manager: Handles loading and pooling of images for the dataset generator.
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from config import (
    EXTRACTED_IMAGES_DIR,
    PAPER_TEXTURE_DIR,
    SHADOW_OVERLAYS_DIR,
    BACKGROUND_DIR,
    SUPPORTED_EXTENSIONS,
    RANDOM_SEED
)


class AssetManager:
    """
    Manages all image assets for the dataset generator.
    
    Responsibilities:
    - Scan and index all document images
    - Pool background, texture, and shadow images
    - Provide random selection from pools
    - Cache loaded images for performance
    """
    
    def __init__(self, seed: Optional[int] = RANDOM_SEED):
        """
        Initialize the asset manager.
        
        Args:
            seed: Random seed for reproducibility. None for random.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Master document list (sorted for reproducibility)
        self.master_docs: List[Path] = []
        
        # Shuffled documents for random access without repetition
        self._shuffled_docs: List[Path] = []
        self._doc_index = 0
        
        # Asset pools
        self.backgrounds: List[Path] = []
        self.textures: List[Path] = []
        self.shadows: List[Path] = []
        
        # Shuffled pools for random selection without repetition
        self._shuffled_textures: List[Path] = []
        self._shuffled_shadows: List[Path] = []
        self._shuffled_backgrounds: List[Path] = []
        
        # Current indices for cycling through pools
        self._texture_index = 0
        self._shadow_index = 0
        self._background_index = 0
        
        # Cache for loaded images (optional, for performance)
        self._background_cache: dict = {}
        self._texture_cache: dict = {}
        self._shadow_cache: dict = {}
        
        # Index all assets
        self._build_master_index()
        self._load_asset_pools()
    
    def _is_valid_image(self, path: Path) -> bool:
        """Check if file has a supported image extension."""
        return path.suffix.lower() in SUPPORTED_EXTENSIONS
    
    def _build_master_index(self) -> None:
        """
        Recursively scan Extracted_Images and build sorted master list.
        """
        print(f"📂 Scanning documents in: {EXTRACTED_IMAGES_DIR}")
        
        if not EXTRACTED_IMAGES_DIR.exists():
            raise FileNotFoundError(f"Extracted images directory not found: {EXTRACTED_IMAGES_DIR}")
        
        # Recursively find all valid images
        all_images = []
        for img_path in EXTRACTED_IMAGES_DIR.rglob("*"):
            if img_path.is_file() and self._is_valid_image(img_path):
                all_images.append(img_path)
        
        # Sort alphabetically for reproducibility
        self.master_docs = sorted(all_images, key=lambda p: str(p))
        
        print(f"   Found {len(self.master_docs)} documents")
    
    def _load_asset_pools(self) -> None:
        """
        Load paths to all texture, shadow, and background images.
        Create shuffled copies for random selection without repetition.
        """
        # Textures
        if PAPER_TEXTURE_DIR.exists():
            self.textures = sorted([
                p for p in PAPER_TEXTURE_DIR.iterdir()
                if p.is_file() and self._is_valid_image(p)
            ])
        print(f"📂 Found {len(self.textures)} paper textures")
        
        # Shadows
        if SHADOW_OVERLAYS_DIR.exists():
            self.shadows = sorted([
                p for p in SHADOW_OVERLAYS_DIR.iterdir()
                if p.is_file() and self._is_valid_image(p)
            ])
        print(f"📂 Found {len(self.shadows)} shadow overlays")
        
        # Backgrounds
        if BACKGROUND_DIR.exists():
            self.backgrounds = sorted([
                p for p in BACKGROUND_DIR.iterdir()
                if p.is_file() and self._is_valid_image(p)
            ])
        print(f"📂 Found {len(self.backgrounds)} background images")
        
        # Validate we have all required assets
        if not self.textures:
            raise FileNotFoundError(f"No textures found in: {PAPER_TEXTURE_DIR}")
        if not self.shadows:
            raise FileNotFoundError(f"No shadows found in: {SHADOW_OVERLAYS_DIR}")
        if not self.backgrounds:
            raise FileNotFoundError(f"No backgrounds found in: {BACKGROUND_DIR}")
        
        # Initialize shuffled pools (copies of original, then shuffled)
        self._shuffle_all_pools()
    
    def _shuffle_all_pools(self) -> None:
        """
        Create shuffled copies of all asset pools and reset indices.
        Call this to randomize the order without repetition within a batch.
        """
        # Shuffle documents
        self._shuffled_docs = self.master_docs.copy()
        random.shuffle(self._shuffled_docs)
        self._doc_index = 0
        
        # Shuffle textures
        self._shuffled_textures = self.textures.copy()
        random.shuffle(self._shuffled_textures)
        self._texture_index = 0
        
        # Shuffle shadows
        self._shuffled_shadows = self.shadows.copy()
        random.shuffle(self._shuffled_shadows)
        self._shadow_index = 0
        
        # Shuffle backgrounds
        self._shuffled_backgrounds = self.backgrounds.copy()
        random.shuffle(self._shuffled_backgrounds)
        self._background_index = 0
    
    @property
    def total_docs(self) -> int:
        """Return total number of documents to process."""
        return len(self.master_docs)
    
    def get_document(self, index: int) -> Tuple[Path, np.ndarray]:
        """
        Load a specific document by index.
        
        Args:
            index: Index in master_docs list
            
        Returns:
            Tuple of (path, image as numpy array BGR)
        """
        if index < 0 or index >= len(self.master_docs):
            raise IndexError(f"Document index {index} out of range [0, {len(self.master_docs)})")
        
        path = self.master_docs[index]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        if img is None:
            raise IOError(f"Failed to load document: {path}")
        
        return path, img
    
    def get_next_random_document(self) -> Tuple[Path, np.ndarray]:
        """
        Load next document from shuffled pool (random without repetition).
        When all documents exhausted, reshuffles the pool.
        
        Returns:
            Tuple of (path, image as numpy array BGR)
        """
        # Reshuffle if we've gone through all
        if self._doc_index >= len(self._shuffled_docs):
            self._shuffled_docs = self.master_docs.copy()
            random.shuffle(self._shuffled_docs)
            self._doc_index = 0
        
        path = self._shuffled_docs[self._doc_index]
        self._doc_index += 1
        
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to load document: {path}")
        
        return path, img
    
    def get_random_texture(self) -> Tuple[Path, np.ndarray]:
        """
        Load next texture from shuffled pool (random without repetition).
        When all textures exhausted, reshuffles the pool.
        
        Returns:
            Tuple of (path, image as numpy array BGR)
        """
        # Reshuffle if we've gone through all
        if self._texture_index >= len(self._shuffled_textures):
            self._shuffled_textures = self.textures.copy()
            random.shuffle(self._shuffled_textures)
            self._texture_index = 0
        
        path = self._shuffled_textures[self._texture_index]
        self._texture_index += 1
        
        # Check cache first
        if path in self._texture_cache:
            return path, self._texture_cache[path].copy()
        
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to load texture: {path}")
        
        # Cache for reuse
        self._texture_cache[path] = img
        return path, img.copy()
    
    def get_random_shadow(self) -> Tuple[Path, np.ndarray]:
        """
        Load next shadow from shuffled pool (random without repetition).
        When all shadows exhausted, reshuffles the pool.
        
        Returns:
            Tuple of (path, image as numpy array BGRA)
        """
        # Reshuffle if we've gone through all
        if self._shadow_index >= len(self._shuffled_shadows):
            self._shuffled_shadows = self.shadows.copy()
            random.shuffle(self._shuffled_shadows)
            self._shadow_index = 0
        
        path = self._shuffled_shadows[self._shadow_index]
        self._shadow_index += 1
        
        # Check cache first
        if path in self._shadow_cache:
            return path, self._shadow_cache[path].copy()
        
        # Load with alpha channel (IMREAD_UNCHANGED)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to load shadow: {path}")
        
        # Ensure we have an alpha channel
        if len(img.shape) == 2:
            # Grayscale - convert to BGRA
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            # BGR - add alpha channel (use intensity as alpha)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = 255 - gray  # Invert: dark areas become opaque
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha
        
        # Cache for reuse
        self._shadow_cache[path] = img
        return path, img.copy()
    
    def get_random_background(self) -> Tuple[Path, np.ndarray]:
        """
        Load next background from shuffled pool (random without repetition).
        When all backgrounds exhausted, reshuffles the pool.
        
        Returns:
            Tuple of (path, image as numpy array BGR)
        """
        # Reshuffle if we've gone through all
        if self._background_index >= len(self._shuffled_backgrounds):
            self._shuffled_backgrounds = self.backgrounds.copy()
            random.shuffle(self._shuffled_backgrounds)
            self._background_index = 0
        
        path = self._shuffled_backgrounds[self._background_index]
        self._background_index += 1
        
        # Check cache first
        if path in self._background_cache:
            return path, self._background_cache[path].copy()
        
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to load background: {path}")
        
        # Cache for reuse
        self._background_cache[path] = img
        return path, img.copy()
    
    def get_relative_path(self, doc_path: Path) -> Path:
        """
        Get the relative path of a document from Extracted_Images root.
        
        Args:
            doc_path: Absolute path to document
            
        Returns:
            Relative path (e.g., "1/pdf1_0.png")
        """
        return doc_path.relative_to(EXTRACTED_IMAGES_DIR)
    
    def clear_cache(self) -> None:
        """Clear all cached images to free memory."""
        self._background_cache.clear()
        self._texture_cache.clear()
        self._shadow_cache.clear()
    
    def reshuffle_pools(self) -> None:
        """
        Reshuffle all asset pools and reset iteration indices.
        Use this before processing a new batch to ensure variety.
        """
        self._shuffle_all_pools()
    
    def preload_assets(self) -> None:
        """
        Preload all auxiliary assets into cache.
        Useful for small datasets where RAM is not a concern.
        """
        print("🔄 Preloading assets into cache...")
        
        for path in self.textures:
            if path not in self._texture_cache:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    self._texture_cache[path] = img
        
        for path in self.shadows:
            if path not in self._shadow_cache:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Ensure alpha channel
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.shape[2] == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        alpha = 255 - gray
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                        img[:, :, 3] = alpha
                    self._shadow_cache[path] = img
        
        print(f"   Cached: {len(self._texture_cache)} textures, {len(self._shadow_cache)} shadows")


if __name__ == "__main__":
    # Quick test
    print("Testing AssetManager...")
    manager = AssetManager()
    
    print(f"\nTotal documents: {manager.total_docs}")
    
    # Test loading
    if manager.total_docs > 0:
        path, doc = manager.get_document(0)
        print(f"First document: {path.name}, shape: {doc.shape}")
    
    tex_path, tex = manager.get_random_texture()
    print(f"Random texture: {tex_path.name}, shape: {tex.shape}")
    
    shd_path, shd = manager.get_random_shadow()
    print(f"Random shadow: {shd_path.name}, shape: {shd.shape}")
    
    print("\n✅ AssetManager test passed!")
