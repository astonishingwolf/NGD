import bpy
import os

def load_mesh(filepath):
    """Load a mesh from file and return the object."""
    bpy.ops.wm.obj_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj

def setup_bake_settings():
    """Configure render and bake settings."""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_nodes.clear()
    
    bg = world_nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 1.0
    output = world_nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])
    
    bpy.context.scene.render.bake.use_selected_to_active = False
    bpy.context.scene.render.bake.margin = 16
    bpy.context.scene.cycles.samples = 128
    
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.cycles.max_bounces = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 1
    bpy.context.scene.cycles.volume_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 1
    bpy.context.scene.world.light_settings.distance = 10.0

def setup_bake_material(obj, image_size=1024):
    """Create or modify material for baking."""
    normal_img = bpy.data.images.new(f"NormalMap_{obj.name}", width=image_size, height=image_size)
    ao_img = bpy.data.images.new(f"AOMap_{obj.name}", width=image_size, height=image_size)
    
    normal_img.pixels = [0.5] * (4 * image_size * image_size)
    ao_img.pixels = [1.0] * (4 * image_size * image_size)
    
    if len(obj.data.materials) == 0:
        mat = bpy.data.materials.new(name=f"BakeMaterial_{obj.name}")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    original_nodes = list(nodes)
    original_links = list(links)
    
    nodes.clear()
    
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    normal_node = nodes.new(type="ShaderNodeTexImage")
    ao_node = nodes.new(type="ShaderNodeTexImage")
    
    normal_node.image = normal_img
    ao_node.image = ao_img
    normal_node.image.colorspace_settings.name = 'Non-Color'
    ao_node.image.colorspace_settings.name = 'sRGB'
    
    output.location = (400, 0)
    diffuse.location = (200, 0)
    normal_node.location = (0, 100)
    ao_node.location = (0, -100)
    
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    
    return normal_img, ao_img

def restore_material(mat, original_nodes, original_links):
    """Restore original material nodes and links."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    nodes.clear()
    
    for node in original_nodes:
        nodes.new(type=node.bl_idname)
    
    for link in original_links:
        links.new(link.from_socket, link.to_socket)

def bake_maps(obj, normal_img, ao_img, output_dir, index=0):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'  
    bpy.context.scene.render.bake.use_selected_to_active = False
    bpy.context.scene.render.bake.margin = 16
    bpy.context.scene.cycles.samples = 64  
    bpy.context.scene.cycles.bake_type = 'NORMAL'
    bpy.context.scene.render.bake.normal_space = 'OBJECT'
    normal_img.filepath_raw = f"{output_dir}/NormalMap_{index}.png"
    normal_img.file_format = 'PNG'
    bpy.ops.object.bake(type='NORMAL')
    normal_img.save()
    
    bpy.context.scene.cycles.bake_type = 'AO'
    bpy.context.scene.render.bake.use_pass_direct = True
    bpy.context.scene.render.bake.use_pass_indirect = True
    bpy.context.scene.cycles.samples = 128 
    normal_img.filepath_raw = f"{output_dir}/AOMap_{index}.png"
    normal_img.file_format = 'PNG'
    bpy.ops.object.bake(type='AO')
    normal_img.save()


def process_folder(input_folder, output_folder, image_size=1024):
    """Process all OBJ files in the input folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    setup_bake_settings()
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.obj'):
            filepath = os.path.join(input_folder, filename)
            
            index = filename.split('.')[0].split('_')[-1]
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            
            print(f"Processing: {filename}")
            
            obj = load_mesh(filepath)
            
            normal_img, ao_img = setup_bake_material(obj, image_size)
            
            bake_maps(obj, normal_img, ao_img, output_folder, index)
            
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.images.remove(normal_img, do_unlink=True)
            bpy.data.images.remove(ao_img, do_unlink=True)
            
            print(f"Completed: {filename}")

input_folder = "/hdd_data/nakul/soham/New_Results/Dress4DRecon_Ortho/185/lower/Remeshed/Meshes_with_UV"
output_folder = "/hdd_data/nakul/soham/New_Results/Dress4DRecon_Ortho/185/lower/Remeshed/Maps"

process_folder(input_folder, output_folder, image_size=1024)