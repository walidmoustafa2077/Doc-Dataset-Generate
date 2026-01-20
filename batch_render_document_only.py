"""
Batch render documents with random objects, textures, and shadows.
Loads document_setup.blend, replaces plan mesh with random obj.
Outputs target (clean) and input (with shadows) images.
"""

import bpy
import sys
import os
import random
import shutil
import importlib.util
from pathlib import Path

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
spec = importlib.util.spec_from_file_location("blender_generator", BASE_PATH / "blender_generator.py")
blender_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blender_gen)
add_shadow_casters = blender_gen.add_shadow_casters

OUTPUT_DIR = BASE_PATH / "SynDoc_Wild_v1"
TARGET_DIR = OUTPUT_DIR / "train" / "target"
INPUT_DIR = OUTPUT_DIR / "train" / "input"
MASK_DIR = OUTPUT_DIR / "train" / "mask"
OBJ_DIR = BASE_PATH / "obj"

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

def replace_plan_with_obj(obj_dir: Path):
    """Replace Document mesh with random obj mesh, preserving transforms, materials, and modifiers."""
    # Store original transforms, materials, and modifiers
    original_location = None
    original_rotation_euler = None
    original_scale = None
    old_materials = []
    old_modifiers_data = []
    
    # Check if Document exists
    if "Document" in bpy.data.objects:
        plane = bpy.data.objects["Document"]
        original_location = plane.location.copy()
        original_rotation_euler = plane.rotation_euler.copy()
        original_scale = plane.scale.copy()
        
        # Store materials
        old_materials = list(plane.data.materials[:])
        
        # Store modifier properties
        for mod in plane.modifiers:
            mod_data = {
                'name': mod.name,
                'type': mod.type,
            }
            # Store common modifier properties
            if mod.type == 'SIMPLE_DEFORM':
                # Store only attributes that exist
                if hasattr(mod, 'deform_method'):
                    mod_data['deform_method'] = mod.deform_method
                if hasattr(mod, 'deform_axis'):
                    mod_data['deform_axis'] = mod.deform_axis
                if hasattr(mod, 'angle'):
                    mod_data['angle'] = mod.angle
                if hasattr(mod, 'factor'):
                    mod_data['factor'] = mod.factor
                if hasattr(mod, 'limits'):
                    mod_data['limits'] = tuple(mod.limits)
            old_modifiers_data.append(mod_data)
        
        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        # Select and delete
        plane.select_set(True)
        bpy.context.view_layer.objects.active = plane
        bpy.ops.object.delete()
    
    # Scan for obj files
    obj_files = scan_directory(obj_dir, extensions=['.obj'])
    if not obj_files:
        return
    
    # Deselect all before import
    bpy.ops.object.select_all(action='DESELECT')
    
    # Pick random obj
    obj_path = random.choice(obj_files)
    
    # Import obj
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    imported_obj = bpy.context.selected_objects[-1]
    imported_obj.visible_shadow = False
    
    # Rename to "Document"
    imported_obj.name = "Document"
    imported_obj.data.name = "Document"
    
    # Apply original materials
    if old_materials:
        imported_obj.data.materials.clear()
        for mat in old_materials:
            imported_obj.data.materials.append(mat)
    
    # Recreate modifiers on new object
    for mod_data in old_modifiers_data:
        new_mod = imported_obj.modifiers.new(name=mod_data['name'], type=mod_data['type'])
        if mod_data['type'] == 'SIMPLE_DEFORM':
            if 'deform_method' in mod_data:
                new_mod.deform_method = mod_data['deform_method']
            if 'deform_axis' in mod_data:
                new_mod.deform_axis = mod_data['deform_axis']
            if 'angle' in mod_data:
                new_mod.angle = mod_data['angle']
            if 'factor' in mod_data:
                new_mod.factor = mod_data['factor']
            if 'limits' in mod_data:
                new_mod.limits = mod_data['limits']
    
    # Apply original transforms
    if original_location is not None:
        imported_obj.location = original_location
    if original_rotation_euler is not None:
        imported_obj.rotation_euler = original_rotation_euler
    if original_scale is not None:
        imported_obj.scale = original_scale
    
    return obj_path.name

def get_document_material(doc_obj):
    """Get or create the document material on plan object."""
    if not doc_obj.data.materials:
        # Create a new material
        mat = bpy.data.materials.new(name="DocMaterial")
        doc_obj.data.materials.append(mat)
        return mat
    return doc_obj.data.materials[0]

def randomize_simple_deform(doc_obj):
    """Randomize SimpleDeform modifier on Document object."""
    # Find SimpleDeform modifier
    simple_deform_mod = None
    for mod in doc_obj.modifiers:
        if mod.type == 'SIMPLE_DEFORM':
            simple_deform_mod = mod
            break
    
    if not simple_deform_mod:
        print(f"✗ No SimpleDeform modifier found on {doc_obj.name}")
        return
    
    # Random deform method (TWIST, BEND, TAPER, STRETCH)
    try:
        methods = ['TWIST', 'BEND', 'TAPER', 'STRETCH']
        deform_method = random.choice(methods)
        simple_deform_mod.deform_method = deform_method
    except Exception as e:
        print(f"  ✗ Deform Method error: {e}")
    
    # Random axis (X, Y, Z)
    try:
        axes = ['X', 'Y', 'Z']
        deform_axis = random.choice(axes)
        simple_deform_mod.deform_axis = deform_axis
    except Exception as e:
        print(f"  ✗ Deform Axis error: {e}")
    
    # Random angle - special handling for TWIST+Z axis
    try:
        if deform_method == 'TWIST' and deform_axis == 'Z':
            # For TWIST+Z, use fixed angles 15 or -15 degrees (converted to radians)
            angle = random.choice([0.2618, -0.2618])  # 15° = 0.2618 rad, -15° = -0.2618 rad
        elif deform_method in ['STRETCH', 'TAPER']:
            angle = random.uniform(-0.5, 0.5)  # Lower range for STRETCH/TAPER
        else:
            angle = random.uniform(-0.9, 0.9)  # Higher range for TWIST/BEND
        simple_deform_mod.angle = angle
    except Exception as e:
        print(f"  ✗ Angle error: {e}")
        
def change_document_texture(doc_obj, doc_path: Path, paper_texture_path: Path = None):
    """Change the main texture and paper texture of plan material."""
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
        return
    
    # Load the document image
    try:
        if doc_path.exists():
            doc_tex.image = bpy.data.images.load(str(doc_path))
    except Exception as e:
        pass
    
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
            except Exception as e:
                pass

def render_single(output_num: int, doc_path: Path, obj_name: str):
    """Copy raw document to target and render with shadows to input."""
    try:
        scene = bpy.context.scene
        
        # 1. Copy raw document to target directory (no rendering needed)
        target_path = TARGET_DIR / f"{output_num:05d}.png"
        shutil.copy2(str(doc_path), str(target_path))
        
        # 2. Add shadows to the scene
        for _ in range(random.randint(2, 3)):
            add_shadow_casters()
        
        # 3. Render input (with shadows)
        input_path = INPUT_DIR / f"{output_num:05d}.png"
        scene.render.filepath = str(input_path)
        bpy.ops.render.render(write_still=True)
        
        print(f"✓ Render #{output_num}: {obj_name} | Doc: {doc_path.name}")
        return True
    except Exception as e:
        print(f"✗ Render error: {e}")
        return False

def main():
    import argparse
    
    # Parse arguments
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Batch render with document_setup.blend")
    parser.add_argument("--num-renders", "-n", type=int, default=5, help="Number of renders (default: 5)")
    args = parser.parse_args(argv)
    
    print("=" * 70)
    print("🎨 Batch Render - document_setup.blend with Random Objects")
    print("=" * 70)
    
    # Load document_setup.blend
    blend_file = BASE_PATH / "document_setup.blend"
    if blend_file.exists():
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
        print(f"✓ Loaded: {blend_file.name}\n")
    else:
        print(f"✗ {blend_file} not found!")
        return
    
    # Create output directories
    for folder in [TARGET_DIR, INPUT_DIR, MASK_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Get next output number
    existing = list(TARGET_DIR.glob("*.png"))
    start_num = len(existing) + 1
    
    # Scan for documents
    doc_files = scan_directory(EXTRACTED_IMAGES_DIR)
    tex_files = scan_directory(PAPER_TEXTURE_DIR)
    
    if not doc_files:
        print(f"✗ No documents found in {EXTRACTED_IMAGES_DIR}")
        return
    
    print(f"📂 Found {len(doc_files)} documents")
    print(f"📄 Found {len(tex_files)} paper textures")
    print(f"🔧 Found {len(list(OBJ_DIR.glob('*.obj')))} objects")
    print(f"🎬 Will render {args.num_renders} times with random objects\n")
    
    # Setup render settings
    scene = bpy.context.scene
    scene.render.resolution_x = TARGET_WIDTH
    scene.render.resolution_y = TARGET_HEIGHT
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    
    # Render loop
    print(f"🎬 Starting {args.num_renders} renders...\n")
    
    for i in range(args.num_renders):
        output_num = start_num + i
        
        # Pick random document and texture
        doc_path = random.choice(doc_files)
        paper_tex_path = random.choice(tex_files) if tex_files else None
        
        # Replace plan with random obj
        obj_name = replace_plan_with_obj(OBJ_DIR)
        
        # Change textures
        doc_obj = bpy.data.objects.get("Document")
        if not doc_obj:
            print(f"✗ Document object not found!")
            continue
        
        change_document_texture(doc_obj, doc_path, paper_tex_path)
        
        # Randomize SimpleDeform modifier
        randomize_simple_deform(doc_obj)
        
        # Render
        if not render_single(output_num, doc_path, obj_name):
            print(f"  ⚠ Skipping render #{output_num}")
            continue
    
    print("\n" + "=" * 70)
    print(f"✅ Batch rendering complete! {args.num_renders} renders saved")
    print(f"   Target: {TARGET_DIR}")
    print(f"   Input:  {INPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
