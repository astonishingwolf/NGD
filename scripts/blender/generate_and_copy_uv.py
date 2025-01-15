import bpy
import os

def load_mesh(filepath):
    bpy.ops.wm.obj_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj

def copy_materials(source_obj, target_obj):
    target_obj.data.materials.clear()
    
    for mat in source_obj.data.materials:
        if mat.name in bpy.data.materials:
            target_obj.data.materials.append(bpy.data.materials[mat.name])
        else:
            new_mat = mat.copy()
            target_obj.data.materials.append(new_mat)

def copy_uv_layers(source_obj, target_obj):

    while target_obj.data.uv_layers:
        target_obj.data.uv_layers.remove(target_obj.data.uv_layers[0])

    for source_uv in source_obj.data.uv_layers:
        target_uv = target_obj.data.uv_layers.new(name=source_uv.name)
        
        for loop_idx in range(len(target_obj.data.loops)):
            target_uv.data[loop_idx].uv = source_uv.data[loop_idx].uv

def save_mesh(obj, filepath):
    """Save mesh as OBJ file."""
    bpy.ops.object.select_all(action='DESELECT')

    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True
    )

def process_meshes(source_path, target_folder, output_folder):
    """Process all meshes in target folder using source mesh as reference."""
    os.makedirs(output_folder, exist_ok=True)
    
    source_obj = load_mesh(source_path)
    print(f"Loaded source mesh: {source_obj.name}")

    mesh_files = [f for f in os.listdir(target_folder) if f.endswith('.obj')]

    for mesh_file in mesh_files:
        mesh_path = os.path.join(target_folder, mesh_file)

        target_obj = load_mesh(mesh_path)
        print(f"Processing target mesh: {target_obj.name}")
        

        if len(source_obj.data.vertices) != len(target_obj.data.vertices) or \
           len(source_obj.data.polygons) != len(target_obj.data.polygons):
            print(f"Warning: {mesh_file} has different topology than source mesh. Skipping...")
            bpy.data.objects.remove(target_obj, do_unlink=True)
            continue

        copy_materials(source_obj, target_obj)
        copy_uv_layers(source_obj, target_obj)
    
        output_path = os.path.join(output_folder, f"processed_{mesh_file}")
        save_mesh(target_obj, output_path)
        print(f"Saved processed mesh to: {output_path}")

        bpy.data.objects.remove(target_obj, do_unlink=True)
    
    bpy.data.objects.remove(source_obj, do_unlink=True)
    print("Processing complete!")


source_mesh_path = "/hdd_data/nakul/soham/New_Results/Dress4DRecon_Ortho/185/lower/Remeshed/remeshed_template/source_mesh.obj"
target_folder_path = "/hdd_data/nakul/soham/New_Results/Dress4DRecon_Ortho/185/lower/Remeshed/Meshes"
output_folder_path = "/hdd_data/nakul/soham/New_Results/Dress4DRecon_Ortho/185/lower/Remeshed/Meshes_with_UV"

process_meshes(source_mesh_path, target_folder_path, output_folder_path)