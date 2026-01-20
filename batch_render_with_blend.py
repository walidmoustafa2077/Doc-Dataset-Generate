"""
Batch render documents using blender_generator.blend file.
Changes DocumentPlane main texture and renders 10 times with different documents.
"""

import bpy
import sys
import os
import random
import shutil
from pathlib import Path
from mathutils import Vector

# Setup paths
BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(BASE_PATH))

from config import (
    EXTRACTED_IMAGES_DIR,
    PAPER_TEXTURE_DIR,
    SUPPORTED_EXTENSIONS,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)

# Import shadow caster functions from blender_generator
import importlib.util
spec = importlib.util.spec_from_file_location("blender_generator", BASE_PATH / "blender_generator.py")
blender_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blender_gen)
add_shadow_casters = blender_gen.add_shadow_casters

OUTPUT_DIR = BASE_PATH / "SynDoc_Wild_3D_BlendFile"
TARGET_DIR = OUTPUT_DIR / "train" / "target"
INPUT_DIR = OUTPUT_DIR / "train" / "input"

def scan_directory(directory: Path, extensions: list = None) -> list:
    """Scan directory for supported image files."""
    files = []
    if not directory.exists():
        return files
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                files.append(Path(root) / f)
    return files

def get_document_material(doc_obj):
    """Get or create the document material on DocumentPlane."""
    if not doc_obj.data.materials:
        # Create a new material
        mat = bpy.data.materials.new(name="DocMaterial")
        doc_obj.data.materials.append(mat)
        return mat
    return doc_obj.data.materials[0]

def change_document_texture(doc_obj, doc_path: Path, paper_texture_path: Path = None):
    """Change the main texture and paper texture of DocumentPlane material."""
    mat = get_document_material(doc_obj)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # 1. Change main document texture - find "Image Texture" node
    doc_tex = None
    for node in nodes:
        if node.type == 'TEX_IMAGE' and node.name == "Image Texture":
            doc_tex = node
            break
    
    if not doc_tex:
        print(f"✗ No 'Image Texture' node found in material")
        return
    
    # Load the document image
    try:
        if doc_path.exists():
            doc_tex.image = bpy.data.images.load(str(doc_path))
            print(f"✓ Loaded main texture: {doc_path.name}")
        else:
            print(f"✗ Document path not found: {doc_path}")
    except Exception as e:
        print(f"✗ Error loading main texture: {e}")
    
    # 2. Change paper texture node - find "Image Texture.001" node
    if paper_texture_path:
        paper_tex = None
        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.name == "Image Texture.001":
                paper_tex = node
                break
        
        if paper_tex:
            try:
                if paper_texture_path.exists():
                    paper_tex.image = bpy.data.images.load(str(paper_texture_path))
                    print(f"✓ Loaded paper texture: {paper_texture_path.name}")
                else:
                    print(f"✗ Paper texture path not found: {paper_texture_path}")
            except Exception as e:
                print(f"✗ Error loading paper texture: {e}")
        else:
            print(f"✗ 'Image Texture.001' node not found in material")
    
    # 3. Change Mix node B input to random color
    mix_node = None
    for node in nodes:
        if node.type == 'MIX' and node.name == "Mix":
            mix_node = node
            break
    
    if mix_node:
        try:
            # Generate vibrant random RGBA color (not near white)
            # Use higher saturation by keeping values in 0.1-0.9 range with more contrast
            random_color = (
                random.uniform(0.1, 0.9),  # R
                random.uniform(0.1, 0.9),  # G
                random.uniform(0.1, 0.9),  # B
                1.0  # Alpha
            )
            # Ensure color is not too washed out - boost saturation
            # by scaling away from 0.5 (middle gray)
            r, g, b = random_color[0], random_color[1], random_color[2]
            r = 0.5 + (r - 0.5) * 1.3  # Boost saturation
            g = 0.5 + (g - 0.5) * 1.3
            b = 0.5 + (b - 0.5) * 1.3
            # Clamp to valid range
            random_color = (
                max(0.0, min(1.0, r)),
                max(0.0, min(1.0, g)),
                max(0.0, min(1.0, b)),
                1.0
            )
            # Set the B input to random color (input index 7 for RGBA B)
            mix_node.inputs[7].default_value = random_color
            print(f"✓ Set Mix node B color to RGB({random_color[0]:.2f}, {random_color[1]:.2f}, {random_color[2]:.2f})")
        except Exception as e:
            print(f"✗ Error setting Mix node color: {e}")
    else:
        print(f"✗ 'Mix' node not found in material")

def render_single(output_num: int, doc_path: Path):
    """Copy raw document to target and render with shadows to input."""
    try:
        scene = bpy.context.scene
        
        # 1. Copy raw document to target directory (no rendering needed)
        target_path = TARGET_DIR / f"{output_num:05d}.png"
        shutil.copy2(str(doc_path), str(target_path))
        print(f"  Copied to target: {target_path.name}")
        
        # 2. Add shadows to the scene
        for _ in range(random.randint(2, 3)):
            add_shadow_casters()
        
        # 3. Render input (with shadows)
        input_path = INPUT_DIR / f"{output_num:05d}.png"
        scene.render.filepath = str(input_path)
        bpy.ops.render.render(write_still=True)
        
        print(f"✓ Render #{output_num}: target copied, input rendered")
        return True
    except Exception as e:
        print(f"✗ Render error: {e}")
        return False

def main():
    import argparse
    
    # Parse arguments
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Batch render with blend file")
    parser.add_argument("--num-renders", "-n", type=int, default=10, help="Number of renders (default: 10)")
    parser.add_argument("--debug", action='store_true', help="Print debug info")
    args = parser.parse_args(argv)
    
    print("=" * 70)
    print("🎨 Batch Render with Blender File - DocumentPlane Texture Swapper")
    print("=" * 70)
    
    # Create output directories
    for folder in [TARGET_DIR, INPUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Scan for documents
    doc_files = scan_directory(EXTRACTED_IMAGES_DIR)
    tex_files = scan_directory(PAPER_TEXTURE_DIR)
    
    if not doc_files:
        print(f"✗ No documents found in {EXTRACTED_IMAGES_DIR}")
        return
    
    print(f"\n📂 Found {len(doc_files)} documents")
    print(f"📄 Found {len(tex_files)} paper textures")
    print(f"🎬 Will render {args.num_renders} times with different documents\n")
    
    # Find DocumentPlane in the scene
    doc_obj = bpy.data.objects.get("DocumentPlane")
    if not doc_obj:
        print("✗ DocumentPlane not found in scene. Make sure you opened blender_generator.blend")
        return
    
    print(f"✓ Found DocumentPlane object")
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.resolution_x = TARGET_WIDTH
    scene.render.resolution_y = TARGET_HEIGHT
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    
    # Render loop
    print(f"\n🎬 Starting {args.num_renders} renders...\n")
    
    for i in range(1, args.num_renders + 1):
        # Pick a random document
        doc_path = random.choice(doc_files)
        # Pick a random paper texture
        paper_tex_path = random.choice(tex_files) if tex_files else None
        print(f"[{i}/{args.num_renders}] Doc: {doc_path.name}")
        if paper_tex_path:
            print(f"           Texture: {paper_tex_path.name}")
        
        # Change textures
        change_document_texture(doc_obj, doc_path, paper_tex_path)
        
        # Render
        if not render_single(i, doc_path):
            print(f"  ⚠ Skipping render #{i}")
            continue
    
    print("\n" + "=" * 70)
    print(f"✅ Batch rendering complete! {args.num_renders} renders saved")
    print(f"   Target: {TARGET_DIR}")
    print(f"   Input:  {INPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
