"""
Blender 4.2 LTS Document Renderer - A4 Output
Compatible with EEVEE-Next API.
"""

import bpy  # type: ignore
import bmesh # type: ignore
import math
import os
import sys
import random
import warnings
from pathlib import Path
from mathutils import Vector   # type: ignore


# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(BASE_PATH))

# Import from your config.py
from config import (
    EXTRACTED_IMAGES_DIR,
    PAPER_TEXTURE_DIR,
    SUPPORTED_EXTENSIONS,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    BACKGROUND_DIR
)

# Output paths
OUTPUT_DIR = BASE_PATH / "SynDoc_Wild_3D"
TARGET_DIR = OUTPUT_DIR / "train" / "target"
OBJ_DIR = BASE_PATH / "obj"

# Render settings
DOC_WIDTH = 1.0
DOC_HEIGHT = 1.414

# =============================================================================
# UTILS
# =============================================================================

def scan_directory(directory: Path, extensions: list = None) -> list:
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

# =============================================================================
# BLENDER SETUP (4.2 LTS)
# =============================================================================

def clear_scene():
    """Purge data using the correct Blender 4.2+ ID naming convention."""
    # 1. Delete all objects in the scene
    if bpy.context.object:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    
    # 2. Correct 4.2 LTS Purge Syntax
    # In 4.0, 4.1, 4.2, and 5.0, the arguments are do_local_ids and do_linked_ids
    bpy.data.orphans_purge(
        do_local_ids=True, 
        do_linked_ids=True, 
        do_recursive=True
    )
    
    # 3. Manual cleanup for safety
    for img in bpy.data.images:
        if img.users == 0:
            bpy.data.images.remove(img)
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)

            
def setup_render_settings():
    scene = bpy.context.scene
    # 4.2+ uses EEVEE_NEXT
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    
    scene.render.resolution_x = TARGET_WIDTH
    scene.render.resolution_y = TARGET_HEIGHT
    scene.render.resolution_percentage = 100
    
    # EEVEE Next settings
    eevee = scene.eevee
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 124

    if hasattr(eevee, "use_temporal_antialiasing"):
        eevee.use_temporal_antialiasing = True

    if hasattr(eevee, "shadow_pool_size"):
        eevee.shadow_pool_size = '256' 

    # This blurs the shadow map pixels into a smooth gradient
    if hasattr(scene.eevee, "use_shadow_jitter"):
        scene.eevee.use_shadow_jitter = True

    # PERFORMANCE
    scene.render.use_persistent_data = True

    # Color Management
    scene.view_settings.exposure = 0
    scene.view_settings.view_transform = 'AgX'
    
    try:
        scene.view_settings.look = 'AgX - High Contrast'
    except TypeError:
        # Fallback in case your specific Blender build uses different naming
        scene.view_settings.look = 'None'
    
    # Image Output Settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'


def setup_camera_and_plane(background_path: Path = None):

    # 1. Setup Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.location = (0, 0, 1.41)
    
    avg_color = (1.0, 1.0, 1.0) # Default white

    # 2. Setup Background and Sample Color
    if background_path and background_path.exists():
        # Add background plane
        bpy.ops.mesh.primitive_plane_add(size=1)
        bg_plane = bpy.context.object
        bg_plane.name = "BackgroundPlane"
        
        bg_mat = bpy.data.materials.new(name="BG_Material")
        bg_mat.use_nodes = True
        nodes = bg_mat.node_tree.nodes
        links = bg_mat.node_tree.links
        
        # Load Image
        bg_img = bpy.data.images.load(str(background_path))
        
        # --- COLOR SAMPLING LOGIC ---
        # Sample pixels to get the average scene tint
        pixels = list(bg_img.pixels) # This is [R,G,B,A, R,G,B,A...]
        if pixels:
            # Sample every 100th pixel for performance
            r_vals = pixels[0::400]
            g_vals = pixels[1::400]
            b_vals = pixels[2::400]
            avg_color = (
                sum(r_vals) / len(r_vals),
                sum(g_vals) / len(g_vals),
                sum(b_vals) / len(b_vals)
            )
        
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = bg_img
        links.new(tex_node.outputs['Color'], nodes["Principled BSDF"].inputs['Base Color'])
        
        bg_plane.data.materials.append(bg_mat)
        bg_plane.location = (0, 0, -1)
        bg_plane.scale = (4, 4, 1)

    # 3. Setup Sunlight (Directional)
    sun_data = bpy.data.lights.new(name="Sunlight", type='SUN')
    sun_obj = bpy.data.objects.new(name="Sunlight", object_data=sun_data)
    bpy.context.collection.objects.link(sun_obj)
    

    # INCREASE ENERGY: 0.2 is too low for EEVEE-Next
    sun_data.energy = random.uniform(1.75, 2.75) 
    
    # SOFTEN SHADOWS: Increase angle for more realistic "phone" shadows
    sun_data.angle = random.uniform(0.02, 0.1) # Radians (~5 degrees)
    
    # Force the light to use Jitter
    if hasattr(sun_data, "use_jitter"):
        sun_data.use_jitter = True
        # Overblur makes things fuzzy. Set to 0.1 or 0 for maximum hardness.
        if hasattr(sun_data, "jitter_overblur"):
            sun_data.jitter_overblur = 0.5 
    
    # Lower values (0.0001) make shadows appear for tiny wrinkles and cracks.
    if hasattr(sun_data, "resolution_limit"):
        sun_data.resolution_limit = 0.00005
    
    # ROTATION: Ensure it points DOWN (X rotation should be around 0 to 0.5)
    sun_obj.rotation_euler = (
        random.uniform(0.1, 0.4),  # Tilt downwards
        random.uniform(-0.2, 0.2), # Slight side tilt
        random.uniform(0, 6.28)    # Random direction
    )
    
    # 4. Setup World Tint (Ambient Light)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (avg_color[0], avg_color[1], avg_color[2], 1)
        # FIX: Start with a LOW strength so Target and Input have the same lighting
        bg_node.inputs['Strength'].default_value = 0.3

    return import_random_obj_mesh(OBJ_DIR)


def import_random_obj_mesh(obj_dir: Path) -> bpy.types.Object:
    obj_files = scan_directory(obj_dir, extensions=['.obj'])
    if not obj_files:
        bpy.ops.mesh.primitive_plane_add(size=1)
        doc = bpy.context.object
        doc.scale = (DOC_HEIGHT, DOC_WIDTH, 1)
        return doc
    
    obj_path = random.choice(obj_files)

    # Using the fast C++ importer available in 4.2
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    imported_obj = bpy.context.selected_objects[-1]
    imported_obj.visible_shadow = False 
    imported_obj.location = (0, 0, random.uniform(0.5, 0.575))
    imported_obj.rotation_euler = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 1.57)
    imported_obj.scale = (DOC_HEIGHT * 0.225, DOC_WIDTH * 0.275, 0.2)
    return imported_obj


def create_document_material(doc_obj, doc_path: Path, texture_path: Path = None):
    mat = bpy.data.materials.new(name="DocMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear() # Start fresh
    
    # 1. Output and BSDF
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')

    # 2. Main Document Texture
    doc_tex = nodes.new('ShaderNodeTexImage')
    doc_tex.image = bpy.data.images.load(str(doc_path))

    # 3. Hue/Saturation Node (Boosts Color)
    # Saturation > 1.0 makes colors (photos/logos) more vivid
    hsv_node = nodes.new('ShaderNodeHueSaturation')
    hsv_node.inputs['Saturation'].default_value = random.uniform(0.8, 1.8) # 1.0 is default
    
    # 4. Brightness/Contrast Node (Darkens Text & Whitens Paper)
    bright_con = nodes.new('ShaderNodeBrightContrast')
    # Contrast > 0.5 makes the black text much "blacker"
    bright_con.inputs['Contrast'].default_value = random.uniform(0.6, 1.2)
    
    # 5. Gamma Node (Final Punch)
    # Gamma > 1.0 makes the midtones (text) deeper and less "grey"
    gamma_node = nodes.new('ShaderNodeGamma')
    gamma_node.inputs['Gamma'].default_value = random.uniform(0.5, 1.1)
    
    # --- LINKING THE ENHANCEMENT CHAIN ---
    # Texture -> HSV -> Bright/Contrast -> Gamma
    links.new(doc_tex.outputs['Color'], hsv_node.inputs['Color'])
    links.new(hsv_node.outputs['Color'], bright_con.inputs['Color'])
    links.new(bright_con.outputs['Color'], gamma_node.inputs['Color'])
    
    final_image_output = gamma_node.outputs['Color']

    # 5. Handle Paper Texture Blending
    if texture_path:
        tex_coord = nodes.new('ShaderNodeTexCoord')
        mapping = nodes.new('ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (1, 1, 1)
        
        paper_tex = nodes.new('ShaderNodeTexImage')
        paper_tex.image = bpy.data.images.load(str(texture_path))
        paper_tex.extension = 'CLIP'
        
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], paper_tex.inputs['Vector'])
        
        # 1. Saturation Node
        paper_hsv = nodes.new('ShaderNodeHueSaturation')
        paper_hsv.inputs['Saturation'].default_value = 1.2  # Increase to 2.0 for very vivid color
        links.new(paper_tex.outputs['Color'], paper_hsv.inputs['Color'])

        # 2. Brightness/Contrast (Acts as Sharpening)
        paper_enhance = nodes.new('ShaderNodeBrightContrast')
        paper_enhance.inputs['Contrast'].default_value = 0.3   # Higher contrast makes fibers/details look "sharper"
        links.new(paper_hsv.outputs['Color'], paper_enhance.inputs['Color'])

        # A. Create a "Cavity Map" (Isolate the black creases)
        # This makes the deep parts of the fold DARKER in the color itself
        rgb_to_bw = nodes.new('ShaderNodeRGBToBW')
        links.new(paper_tex.outputs['Color'], rgb_to_bw.inputs['Color'])
        
        val_ramp = nodes.new('ShaderNodeValToRGB')
        val_ramp.color_ramp.elements[0].position = 0.4  # Darkest folds
        val_ramp.color_ramp.elements[1].position = 0.8  # Flat areas
        
        links.new(rgb_to_bw.outputs['Val'], val_ramp.inputs['Fac'])

        # --- COLOR LOGIC ---
        color_mix = nodes.new('ShaderNodeMix')
        color_mix.data_type = 'RGBA'
        color_mix.blend_type = 'MULTIPLY' 
        color_mix.inputs[0].default_value = random.uniform(0.8, 0.9) 
        
        links.new(final_image_output, color_mix.inputs[6])          # Document image
        links.new(paper_enhance.outputs['Color'], color_mix.inputs[7]) # ENHANCED Texture
        links.new(color_mix.outputs[2], bsdf.inputs['Base Color'])

        # --- PHYSICAL LOGIC (BUMP & ROUGHNESS) ---
        rgb_to_bw = nodes.new('ShaderNodeRGBToBW')
        # Use the enhanced color for the bump so the "sharpening" affects the depth too
        links.new(paper_enhance.outputs['Color'], rgb_to_bw.inputs['Color'])
        
       # Gamma 2.0 crunches the heightmap to make cracks physically deeper
        bump_gamma = nodes.new('ShaderNodeGamma')
        bump_gamma.inputs['Gamma'].default_value = 2.0
        links.new(rgb_to_bw.outputs['Val'], bump_gamma.inputs['Color'])

        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = random.uniform(0.25, 0.45) 
        bump.inputs['Distance'].default_value = random.uniform(0.02, 0.05)
        links.new(bump_gamma.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

        # Roughness Mapping 
        map_rough = nodes.new('ShaderNodeMapRange')
        map_rough.inputs['To Min'].default_value = 0.8
        map_rough.inputs['To Max'].default_value = 1.0
        links.new(rgb_to_bw.outputs['Val'], map_rough.inputs['Value'])
        links.new(map_rough.outputs['Result'], bsdf.inputs['Roughness'])
        
    else:
        links.new(final_image_output, bsdf.inputs['Base Color'])
        bsdf.inputs['Roughness'].default_value = 1.0
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    doc_obj.data.materials.append(mat)


def create_random_blob(name="BigShadowShape"):
    """Creates a smoothed, organic procedural mesh to prevent jagged shadows."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    bm = bmesh.new()
    num_points = random.randint(22, 36)
    base_radius = random.uniform(0.12, 0.28) 
    
    # 1. Create the Wiggly Circle
    verts = []
    for i in range(num_points):
        angle = (2 * math.pi / num_points) * i
        radius = base_radius * random.uniform(0.8, 1.2)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append(bm.verts.new((x, y, 0)))
    
    # 2. Extrude to give it volume (Thin objects cast poor shadows in EEVEE)
    if len(verts) > 2:
        face = bm.faces.new(verts)
        result = bmesh.ops.extrude_face_region(bm, geom=[face])
        verts_to_move = [v for v in result['geom'] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=Vector((0, 0, 0.05)), verts=verts_to_move)

    bm.to_mesh(mesh)
    bm.free()
    
    # A. Shade Smooth: Tells Blender to interpolate light across the surface
    for poly in obj.data.polygons:
        poly.use_smooth = True

    # B. Subdivision Surface: Adds "fake" geometry to round off the edges
    # This makes the shadow silhouette perfectly curved instead of a "pixelated" hexagon
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 2

    # 3. Random Scale
    obj.scale.x = random.uniform(1.2, 2.8)
    obj.scale.y = random.uniform(0.9, 1.8)
    
    return obj


def add_shadow_casters():
    """Adds small procedural blobs that cast dark, irregular shadows."""
    
    # Create the procedural shape
    shadow_caster = create_random_blob()
    
    # 1. HIDE FROM CAMERA
    shadow_caster.visible_camera = False

    # 2. POSITIONING (Randomly scattered above the document)
    shadow_caster.location = (
        random.uniform(-0.4, 0.4), 
        random.uniform(-0.4, 0.4), 
        random.uniform(0.7, 0.9)  # Height relative to document
    )

    # 3. RANDOM ROTATION
    shadow_caster.rotation_euler = (
        random.uniform(0, 6.28),
        random.uniform(0, 6.28),
        random.uniform(0, 6.28)
    )

    return shadow_caster


def prepare_shadow_only_scene(doc_obj):
    """Render a high-contrast Black & White shadow mask."""
    scene = bpy.context.scene
    
    # 1. Color Management: Set to 'Standard' and 'Raw' 
    # This prevents 'AgX' from turning your whites into grey
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    
    # 2. Create the Mask Material (Shader to RGB Technique)
    mask_mat = bpy.data.materials.new(name="Shadow_Mask_Final")
    mask_mat.use_nodes = True
    nodes = mask_mat.node_tree.nodes
    links = mask_mat.node_tree.links
    nodes.clear()
    
    # Nodes: Diffuse -> ShaderToRGB -> ColorRamp -> Emission
    # This captures shadows and converts them to hard black/white data
    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_diff = nodes.new('ShaderNodeBsdfDiffuse')
    node_s2r = nodes.new('ShaderNodeShaderToRGB') # EEVEE Specific
    node_ramp = nodes.new('ShaderNodeValToRGB')
    node_emit = nodes.new('ShaderNodeEmission')
    
    # Configure ColorRamp to "crush" the shadows to pure black
    # Pos 0.0 = Black, Pos 0.1 = White (makes the shadow edges sharp and dark)
    node_ramp.color_ramp.elements[0].position = 0.05
    node_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp.color_ramp.elements[1].position = 0.1
    node_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    
    # --- THE INVERSION LOGIC ---
    # Element 0 (The Shadow/Left Side) -> Set to WHITE
    node_ramp.color_ramp.elements[0].position = 0.05
    node_ramp.color_ramp.elements[0].color = (1, 1, 1, 1) # White
    
    # Element 1 (The Lit/Right Side) -> Set to BLACK
    node_ramp.color_ramp.elements[1].position = 0.1
    node_ramp.color_ramp.elements[1].color = (0, 0, 0, 1) # Black
    # ---------------------------

    # Linking
    links.new(node_diff.outputs['BSDF'], node_s2r.inputs['Shader'])
    links.new(node_s2r.outputs['Color'], node_ramp.inputs['Fac'])
    links.new(node_ramp.outputs['Color'], node_emit.inputs['Color'])
    links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])

    # 3. Apply to Document and Background
    doc_obj.data.materials.clear()
    doc_obj.data.materials.append(mask_mat)
    doc_obj.visible_shadow = False 
    
    bg_plane = bpy.data.objects.get("BackgroundPlane")
    if bg_plane:
        bg_plane.hide_render = False
        bg_plane.data.materials.clear()
        bg_plane.data.materials.append(mask_mat)

    # 4. KILL ALL AMBIENT LIGHT
    # We want ONLY the Sun to provide light
    if scene.world:
        scene.world.use_nodes = True
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs['Strength'].default_value = 0.0
            
    # 5. SUPER POWER SUN
    # This ensures the lit areas are pure 1.0, 1.0, 1.0 White
    for light in bpy.data.lights:
        if light.type == 'SUN':
            light.energy = 5.0 # Extreme power for the mask
            light.color = (1, 1, 1)

    # 6. Disable Compositor (We want the raw mask)
    scene.use_nodes = False


def render_triplet(doc_path: Path, output_counter: int, bg_files: list, tex_files: list,
                   target_dir: Path, input_dir: Path, shadow_dir: Path) -> int:
    """Render a single document triplet (target, input, mask). Returns incremented counter."""

    # Clear Scene for fresh start
    clear_scene()
    
    bg = random.choice(bg_files) if bg_files else None
    tex = random.choice(tex_files) if tex_files else None
    
    doc_obj = setup_camera_and_plane(bg)
    create_document_material(doc_obj, doc_path, tex)
    
    # 1. Target Render
    bpy.context.scene.render.filepath = str(target_dir / f"{output_counter:05d}.png")
    bpy.ops.render.render(write_still=True)
    
    # 2. Add Shadows
    for _ in range(random.randint(2, 3)):
        add_shadow_casters()
    
    # 3. Input Render
    bpy.context.scene.render.filepath = str(input_dir / f"{output_counter:05d}.png")
    bpy.ops.render.render(write_still=True)
    
    # 4. Shadow Mask Render
    prepare_shadow_only_scene(doc_obj)
    # bpy.context.scene.view_settings.exposure = -0.1
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.filepath = str(shadow_dir / f"{output_counter:05d}.png")
    bpy.ops.render.render(write_still=True)
    
    # Reset to RGB for the next document
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    # bpy.context.scene.view_settings.exposure = random.uniform(0.5, 0.9)
    
    return output_counter + 1


def main():
    import argparse
    
    # 1. Parse Arguments
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render documents in triplets (Target, Input, Shadow)")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max documents to process")
    parser.add_argument("--iterations", "-i", type=int, default=1, help="Rendering iterations per document (default: 5)")
    parser.add_argument("--test-split", type=float, default=None, help="Fraction of documents for test set (e.g., 0.2 for 20%)")
    parser.add_argument("--test-iterations", type=int, default=None, help="Iterations for test set (default: same as --iterations)")
    parser.add_argument("--use-test", action='store_true', help="Enable test rendering mode with custom document directory")
    parser.add_argument("--PATH", type=str, default=None, help="Path to documents directory (for train or test mode)")
    parser.add_argument("--start-num", type=int, default=1, help="Starting number for output filenames (default: 1 = 00001.png)")
    args = parser.parse_args(argv)
    
    # Validate test-split
    if args.test_split is not None:
        if not (0 < args.test_split < 1):
            print("❌ --test-split must be between 0 and 1")
            return
    
    # Validate --use-test and --test-split are mutually exclusive
    if args.use_test and args.test_split:
        print("❌ Cannot use both --use-test and --test-split together")
        return
    
    # Validate --start-num is not used with --test-split
    if args.test_split and args.start_num != 1:
        print("❌ Cannot use --test-split with custom --start-num (--test-split requires deterministic naming starting from 1)")
        return
    
    # Validate --start-num
    if args.start_num < 1:
        print("❌ --start-num must be >= 1")
        return

    print("=" * 60)
    print("🎨 Blender Document Renderer - Triplet Dataset Mode")
    print("=" * 60)

    # 2. Define and Create Output Subdirectories
    INPUT_DIR = OUTPUT_DIR / "train" / "input"
    TARGET_DIR = OUTPUT_DIR / "train" / "target"
    SHADOW_DIR = OUTPUT_DIR / "train" / "mask"
    
    # Create test directories if test-split or use-test is specified
    test_input_dir = OUTPUT_DIR / "test" / "input" if (args.test_split or args.use_test) else None
    test_target_dir = OUTPUT_DIR / "test" / "target" if (args.test_split or args.use_test) else None
    test_shadow_dir = OUTPUT_DIR / "test" / "mask" if (args.test_split or args.use_test) else None
    
    for folder in [INPUT_DIR, TARGET_DIR, SHADOW_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    
    if args.test_split or args.use_test:
        for folder in [test_input_dir, test_target_dir, test_shadow_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    # 3. Scan for Asset Files (This fixes the Pylance errors)
    # First check if --PATH is provided for training documents
    if args.PATH and not args.use_test:
        print(f"📂 Using custom document path: {args.PATH}")
        doc_files = scan_directory(Path(args.PATH))
        if not doc_files:
            print(f"❌ No documents found in {args.PATH}")
            return
    else:
        # Default to EXTRACTED_IMAGES_DIR
        doc_files = scan_directory(EXTRACTED_IMAGES_DIR)
        if not doc_files:
            print(f"❌ No documents found in {EXTRACTED_IMAGES_DIR}")
            return
    
    bg_files = scan_directory(BACKGROUND_DIR)
    tex_files = scan_directory(PAPER_TEXTURE_DIR)

    print(f"📂 Found {len(doc_files)} documents")
    print(f"🖼️ Found {len(bg_files)} backgrounds")
    print(f"📄 Found {len(tex_files)} textures")

    # Apply limit if requested
    if args.limit:
        doc_files = doc_files[:args.limit]
        print(f"⚠️ Processing limited to first {len(doc_files)} documents")
    
    # Split into train/test if requested
    train_docs = doc_files
    test_docs = []
    
    if args.use_test:
        # Load documents from custom PATH for test set
        print(f"\n📂 Test mode: Loading documents from {args.PATH}")
        test_docs = scan_directory(Path(args.PATH))
        
        if not test_docs:
            print(f"❌ No documents found in {args.PATH}")
            return
        
        # Apply limit to test documents
        if args.limit:
            test_docs = test_docs[:args.limit]
            print(f"⚠️ Processing limited to first {len(test_docs)} test documents")
        
        # Don't render training documents in test mode
        train_docs = []
        test_iter_count = args.test_iterations if args.test_iterations is not None else args.iterations
        print(f"📊 Test Set: {len(test_docs)} documents, {test_iter_count} iterations per document")
    
    elif args.test_split:
        import random as rand_module
        split_idx = int(len(doc_files) * (1 - args.test_split))
        shuffled_docs = doc_files.copy()
        rand_module.shuffle(shuffled_docs)
        train_docs = shuffled_docs[:split_idx]
        test_docs = shuffled_docs[split_idx:]
        test_iter_count = args.test_iterations if args.test_iterations is not None else 1
        print(f"📊 Train/Test Split: {len(train_docs)} train, {len(test_docs)} test")
        print(f"   Training: {args.iterations} iterations per document")
        print(f"   Testing: {test_iter_count} iterations per document")
    else:
        print(f"📊 Processing: {len(train_docs)} documents with {args.iterations} iterations each")

    # 4. Main Rendering Loop
    print(f"\n🎬 Starting Triplet Rendering...")
    
    # Initial Blender Setup
    setup_render_settings()

    # Counter for all outputs (maintains sequential naming across train/test)
    output_counter = args.start_num
    
    # Render training documents - loop by iteration, then document
    for iter_num in range(1, args.iterations + 1):
        for doc_idx, doc_path in enumerate(train_docs, 1):
            bg = random.choice(bg_files) if bg_files else None
            print(f"\n📝 [Train {doc_idx}/{len(train_docs)}, Iter {iter_num}/{args.iterations}] Doc: {doc_path.name}")
            print(f"    BG: {bg.name if bg else 'None'} | Texture: {random.choice(tex_files).name if tex_files else 'None'}")
            
            output_counter = render_triplet(doc_path, output_counter, bg_files, tex_files,
                                           TARGET_DIR, INPUT_DIR, SHADOW_DIR)
    
    # Render test documents if test split is specified
    if (args.test_split or args.use_test) and test_docs:
        print(f"\n🧪 Starting Test Set Rendering...")
        if args.use_test:
            test_iter_count = args.test_iterations if args.test_iterations is not None else args.iterations
        else:
            test_iter_count = args.test_iterations if args.test_iterations is not None else 1
        output_counter = args.start_num  # Reset counter for test set
        for iter_num in range(1, test_iter_count + 1):
            for doc_idx, doc_path in enumerate(test_docs, 1):
                bg = random.choice(bg_files) if bg_files else None
                iter_str = f", Iter {iter_num}/{test_iter_count}" if test_iter_count > 1 else ""
                print(f"\n📝 [Test {doc_idx}/{len(test_docs)}{iter_str}] Doc: {doc_path.name}")
                print(f"    BG: {bg.name if bg else 'None'} | Texture: {random.choice(tex_files).name if tex_files else 'None'}")
                
                output_counter = render_triplet(doc_path, output_counter, bg_files, tex_files,
                                               test_target_dir, test_input_dir, test_shadow_dir)

    print("\n" + "=" * 60)
    print("✅ Dataset Generation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()