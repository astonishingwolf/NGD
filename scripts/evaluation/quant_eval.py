import torch
import trimesh
import pytorch3d
import numpy as np
from typing import List, Tuple
from pytorch3d.structures import Meshes
from natsort import natsorted

def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor):
    
    chamfer_dist, chamfer_details = pytorch3d.loss.chamfer_distance(
        pred, 
        gt,
        batch_reduction='mean',
        point_reduction='mean'
    )
    
    return chamfer_dist

def evaluation_cd(pred_mesh, 
                 gt_mesh, 
                 n_samples: int = 10000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:

    pred_mesh = trimesh.load_mesh(pred_mesh)
    gt_mesh = trimesh.load_mesh(gt_mesh)

    sampled_pred_points = trimesh.sample.sample_surface_even(pred_mesh, n_samples)
    sampled_gt_points = trimesh.sample.sample_surface_even(gt_mesh, n_samples)
    sampled_pred_points -= sampled_pred_points.mean(axis=0)
    sampled_gt_points -= sampled_gt_points.mean(axis=0)
    
    pred_points = torch.tensor(
        sampled_pred_points,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)  

    gt_points = torch.tensor(
        sampled_gt_points,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0) 

    cd_loss = chamfer_distance(pred_points, gt_points)
    
    return cd_loss.item()

def evaluation_ccv(out_meshes) -> float:

    if len(out_meshes) < 2:
        raise ValueError("Need at least 2 meshes to evaluate vertex consistency")
    
    frame_distances = []
    
    # Iterate over consecutive pairs of meshes
    for i in range(len(out_meshes) - 1):

        curr_mesh_path = out_meshes[i]
        next_verts_path = out_meshes[i + 1]
        
        curr_mesh = trimesh.load_mesh(curr_mesh_path)
        next_mesh = trimesh.load_mesh(next_verts_path)

        if curr_verts.shape != next_verts.shape:
            raise ValueError(f"Inconsistent number of vertices between frames {i} and {i+1}")
        
        l2_distances = np.sqrt(np.sum((curr_mesh.vertices - next_mesh.vertices) ** 2, axis=1))
        mean_frame_distance = np.mean(l2_distances)
        frame_distances.append(mean_frame_distance)
    
    mean_consistency = np.mean(frame_distances)
    
    return mean_consistency

if __name__ == "__main__":

    pred_mesh_folder = ''
    gt_mesh_folder = ''

    pred_mesh_files = natsorted(os.listdir(pred_mesh_folder))
    gt_mesh_files = natsoreted(os.listdir(gt_mesh_folder))

    pred_files = [os.path.join(pred_mesh_folder, f) for f in pred_mesh_files]
    gt_files = [os.path.join(gt_mesh_folder, f) for f in gt_mesh_files]

    for pred_mesh, gt_mesh in zip(pred_files, gt_files):
        cd_score = evaluation_cd(pred_mesh, gt_mesh)
        print(f"Chamfer Distance: {cd_score:.6f}")

    # breakpoint()
    # cd_score = evaluation_cd(pred_mesh, gt_mesh)
    # print(f"Chamfer Distance: {cd_score:.6f}")
    
    # # Example for Vertex Consistency evaluation
    # mesh_sequence = [
    #     trimesh.load(f'path_to_mesh_frame_{i}.obj') 
    #     for i in range(num_frames)
    # ]
    # ccv_score = evaluation_ccv(mesh_sequence)
    # print(f"Vertex Consistency Score: {ccv_score:.6f}")