import torch
import torch.nn as nn
import numpy as np
import trimesh 

from scripts.utils.smpl import LBS

DEVICE = 'cuda'
def garment_skinning_interpolation(orig_verts, orig_skinning_weights, new_vetices, k = 3):

    distances = torch.cdist(orig_verts, new_vetices)
    k_distances, k_indices = distances.topk(k, dim = 1, largest = False)

    eps = 1e-10
    # breakpoint()
    weights = 1 / (k_distances + eps)
    weights = weights / torch.sum(weights, dim = 1, keepdim = True)

    weights = weights.unsqueeze(-1)
    # breakpoint()
    nearest_features = orig_skinning_weights[k_indices]
    interpolated_features = torch.sum(nearest_features*weights, dim = 1)

    return interpolated_features

def garment_skinning_function(v_garment_unskinned, pose, shape, body, skin_weight, translation = None):
    """
    Inputs:
        pose: body pose [batch_size, num_frame, 72]
        shape: body shape parameters [batch_size, num_frame, 10]
        trans_vel: translation velocity [batch_size, num_frame, 3]
        translation: [batch_size, num_frame, 3]
    """
    num_frames = pose.shape[1]

    pose_flat = pose.view(-1, 72)
    shape_flat = shape.view(-1, 10)

    # 
    _, joint_transforms = body(shape=shape_flat, pose=pose_flat)
    
    v_garment_skinning_cur = LBS()(v_garment_unskinned, joint_transforms, skin_weight)
    ## Have to change this 
    if translation is not None:
        v_garment_skinning_cur += translation.view(-1, 1, 3)
    # 
    # v_garment_skinning = v_garment_skinning_cur.view(-1, num_frames, v_garment_unskinned.shape[1], 3)

    return v_garment_skinning_cur
    
def load_garment(gament_model):
    
    mesh = trimesh.load(gament_model)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)

    return vertices, faces


def compute_rbf_skinning_weight(garment_template, smpl):
    
    def gaussian_rbf(distances, min_dist):
        
        """
        Compute weights using a Gaussian RBF.
        Returns:
                [num_vert_garment, num_vert_body]
        """

        # Compute the weights using the Gaussian RBF formula
        sigma = min_dist + 1e-7
        k = 0.25

        weights = torch.exp(
            -(distances - min_dist.unsqueeze(1)) ** 2 / (k * sigma ** 2).unsqueeze(1))
        # Normalize the weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights

    garment_body_distance = torch.cdist(garment_template.to(DEVICE), smpl.template_vertices.to(DEVICE))
    min_dist = garment_body_distance.min(dim=1)[0]  
    weight = gaussian_rbf(garment_body_distance, min_dist)
    w_skinning = torch.matmul(weight, smpl.skinning_weights.to(DEVICE))
    v_weights = w_skinning.to(torch.float32)
    
    return v_weights


def compute_k_nearest_skinning_weights(garment_template, smpl, k=5):
    # squared distance
    garment_body_distance = torch.cdist(garment_template.to(DEVICE), smpl.template_vertices.to(DEVICE))

    closest_indices_body = torch.argsort(garment_body_distance, dim=1)
    contribution = torch.zeros([garment_template.shape[0], smpl.num_vertices]).to(DEVICE)
    B = closest_indices_body[:, :k]
    updates = torch.ones_like(B, dtype=contribution.dtype).to(DEVICE) / k

    rows = torch.arange(B.shape[0], device=DEVICE).view(-1, 1).expand(-1, k)
    
    indices = torch.stack([rows, B], dim=-1).view(-1, 2)
    updates_flat = updates.view(-1)
    
    contribution.index_put_(
        (indices[:, 0], indices[:, 1]),
        updates_flat,
        accumulate=True
    )

    w_skinning = torch.matmul(contribution, smpl.skinning_weights.to(DEVICE))
    v_weights = w_skinning.to(torch.float32)

    return v_weights