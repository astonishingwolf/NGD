###Getting the camera parameters from the peoplpe snapshot dataset and rendering them on the image space

import torch 
import torchvision.transforms as transforms
import numpy as np
import argparse
import yaml
import os
from PIL import Image
import nvdiffrast.torch as dr
import imageio
import torch.nn.functional as F
import shutil
import sys
import pickle
from smplx import SMPL
import pyrender
import trimesh 
# import smpl

def local_to_global_homogenous(joint_rotations, joint_positions, parents):
    """
    Computes absolute joint locations given pose.

    Args:
        joint_rotations: batch_size x K x 3 x 3 rotation tensor of K joints
        joint_positions: batch_size x K x 3, joint locations before posing
        parents: list or tensor of size K holding the parent id for each joint

    Returns:
        joint_transforms: `Tensor`: batch_size x K x 4 x 4 relative joint transformations for LBS.
        joint_positions_posed: batch_size x K x 3, joint locations after posing
    """
    joint_rotations = joint_rotations.unsqueeze(0)
    batch_size = joint_rotations.shape[0]
    num_joints = len(parents)
    
    # 
    def make_affine(rotation, translation):
        '''
        Args:
            rotation: batch_size x 3 x 3
            translation: batch_size x 3 x 1
        '''
        # 
        rotation_homo = F.pad(rotation, (0, 0, 0, 1), value=0.0)
        translation_homo = torch.cat([translation, torch.ones([batch_size, 1, 1], device=rotation.device)], dim=1)
        affine_transform = torch.cat([rotation_homo, translation_homo], dim=2)
        # 
        return affine_transform

    # Expand joint_positions to have an additional dimension for homogenous coeordinates
    joint_positions = joint_positions.unsqueeze(-1)
    # 
    # Initialize root rotation and transform
    root_rotation = joint_rotations[:, 0, :, :]
    root_transform = make_affine(root_rotation, joint_positions[:, 0])
    # 
    # Traverse joints to compute global transformations
    transforms = [root_transform]
    for joint, parent in enumerate(parents[1:], start=1):
        position = joint_positions[:, joint] - joint_positions[:, parent]
        # position = joint_positions[:, joint]
        transform_local = make_affine(joint_rotations[:, joint], position)
        transform_global = torch.matmul(transforms[parent], transform_local)
        transforms.append(transform_global)
    
    # Stack all transforms into a tensor
    transforms = torch.stack(transforms, dim=1)

    # Extract joint positions from the transforms
    joint_positions_posed = transforms[:, :, :3, 3]

    # Compute affine transforms relative to initial state (t-pose)
    zeros = torch.zeros([batch_size, num_joints, 1, 1], device=joint_rotations.device)
    joint_rest_positions = torch.cat([joint_positions, zeros], dim=2)
    init_bone = torch.matmul(transforms, joint_rest_positions)
    init_bone = F.pad(init_bone, (3, 0))
    joint_transforms = transforms - init_bone

    return joint_transforms, joint_positions_posed

    
def axisangle_to_rotationmatrix(axis_angle):
    """
    Converts rotations in axis-angle representation to rotation matrices.
    
    Args:
        axis_angle: tensor of shape (batch_size, 3)
        
    Returns:
        rotation_matrix: tensor of shape (batch_size, 3, 3)
    """
    axis_angle = axis_angle.view(-1, 3)
    batch_size = axis_angle.size(0)
    angle = torch.norm(axis_angle, dim=1, keepdim=True)  
    axis = axis_angle / angle  
    cos = torch.cos(angle.unsqueeze(-1))  
    sin = torch.sin(angle.unsqueeze(-1))  
    
    outer = torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(1))  
    def skew_symmetric(v):
        zero = torch.zeros_like(v[:, 0])
        
        return torch.stack([
            zero, -v[:, 2], v[:, 1],
            v[:, 2], zero, -v[:, 0],
            -v[:, 1], v[:, 0], zero
        ], dim=1).view(-1, 3, 3)
    
    skew = skew_symmetric(axis) 
    eyes = torch.eye(3, device=axis_angle.device).unsqueeze(0).repeat(batch_size, 1, 1) 
    # 
    R = cos * eyes + (1 - cos) * outer + sin * skew
    
    return R 
    

def load_garment(gament_model):
    
    mesh = trimesh.load(gament_model)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)

    return vertices, faces

def compute_k_nearest_skinning_weights(garment_template, smpl, k= 3):

    garment_body_distance = torch.cdist(garment_template, smpl.v_template, p=2)

    closest_indices_body = torch.argsort(garment_body_distance, dim=1)

    contribution = torch.zeros(garment_template.shape[0], smpl.v_template.shape[0], dtype=torch.float32)

    B = closest_indices_body[:, :k]

    updates = torch.ones_like(B, dtype=contribution.dtype) * (1 / k)

    rows = torch.arange(B.shape[0], device=B.device).unsqueeze(1)
    rows_repeated = rows.repeat(1, B.shape[1])

    contribution = contribution.scatter(1, B, updates)

    w_skinning = torch.matmul(contribution, smpl.lbs_weights)

    v_weights = w_skinning.to(torch.float32)

    return v_weights
    
def compute_rbf_skinning_weight(garment_template, smpl):
    
    def gaussian_rbf(distances, min_dist):
        """
        Compute weights using a Gaussian RBF.
        Returns:
                [num_vert_garment, num_vert_body]
        """

        # Compute the weights using the Gaussian RBF formula
        sigma = min_dist + 1e-7
        k = 200000000

        weights = torch.exp(
            -(distances - min_dist.unsqueeze(1)) ** 2 / (k * sigma ** 2).unsqueeze(1))
        # Normalize the weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights


    garment_body_distance = torch.cdist(garment_template, smpl.v_template)
    
    min_dist = garment_body_distance.min(dim=1)[0]  
    
    weight = gaussian_rbf(garment_body_distance, min_dist)

    w_skinning = torch.matmul(weight, smpl.lbs_weights)
    
    v_weights = w_skinning.to(torch.float32)
    
    return v_weights
    
    
def pyrenderer_to_open_gl(proj):
    proj_gl = proj
    proj_gl[2] *= -1
    proj_gl[2, 2] = (proj_gl[2, 2] + proj_gl[3, 2]) / 2
    proj_gl[2, 3] =  (proj_gl[2, 3] + proj_gl[3, 3]) / 2
    proj_gl[3, 2] = -1
    proj_gl[3, 3] = 0
    
    return proj_gl

def save_image(img, path):
    img = img.cpu().numpy()
    img = img * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

# def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
#     pos_clip    = transform_pos(mtx, pos)
#     rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
#     color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
#     color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
#     return color

def render(glctx, proj, mv, pos, pos_idx, res: int = 1080):
    mvp = proj @ mv
    pos_clip    = transform_pos(mvp, pos)
    # print(pos_idx.dtype)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, (res,res))
    alpha = torch.clamp(rast_out[..., [-1]], max=1)

    return alpha

def pkl_to_dict(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        return data

# def weakprojection(mtx, pos):

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = self.translation[1] * self.scale[1]
        P[2, 2] = -1
        # 
        return P
    
def get_projection_matrix(scale, translation, width=None, height=None):
    
    P = np.eye(4)
    
    P[0, 0] = scale[0]*1000
    P[1, 1] = scale[1]*1000
    P[0, 3] = translation[0] * scale[0]
    P[1, 3] = -translation[1] * scale[1]
    P[2, 2] = -1
    
    return P

def skinning(vertices, joint_rotations, skinning_weights):
    
    skinning_weights = torch.tensor(skinning_weights)
    vertices = vertices.unsqueeze(0)
    batch_size = vertices.shape[0]
    num_joints = skinning_weights.shape[-1]
    num_vertices = vertices.shape[-2]

    W = skinning_weights
    if len(skinning_weights.shape) < len(vertices.shape):
        W = W.unsqueeze(0).repeat(batch_size, 1, 1)

    A = joint_rotations.view(-1, num_joints, 16)
    T = torch.matmul(W, A)
    T = T.view(-1, num_vertices, 4, 4)

    ones = torch.ones(batch_size, num_vertices, 1, device=vertices.device)
    # 
    vertices_homo = torch.cat([vertices, ones], dim=2)
    skinned_homo = torch.matmul(T, vertices_homo.unsqueeze(-1))
    skinned_vertices = skinned_homo[:, :, :3, 0]

    return skinned_vertices
    
    
def main(args):
    
    # image_dir = args.image_dir
    smpl_pkl = args.smpl_pkl
    smpl_data = pkl_to_dict(smpl_pkl)
    smpl_model_path = '/home/nakul/soham/3d/fixnormalndepth/SMPL_NEUTRAL.pkl'
    garment_model = '/home/nakul/soham/3d/fixnormalndepth/dataset/Fashion3D/tshirt.obj'
    smpl = SMPL(smpl_model_path, device = 'cpu')
    t = 100
    # proj = get_projection_matrix(smpl_data[1]['orig_cam'][0][0:2], smpl_data[1]['orig_cam'][0][2:4])
    camera = WeakPerspectiveCamera(
            scale=[smpl_data[1]['orig_cam'][t][0], smpl_data[1]['orig_cam'][t][1]],
            translation=[smpl_data[1]['orig_cam'][t][2], smpl_data[1]['orig_cam'][t][3]],
            zfar=1000.
        )
    # 
    mv = np.eye(4)
    # 
    
    garment_vertices, garment_faces = load_garment(garment_model)
    # garment_vertices[:,1] = -garment_vertices[:,1]
    # garment_skinning = compute_rbf_skinning_weight(garment_vertices,smpl)
    garment_skinning = compute_k_nearest_skinning_weights(garment_vertices,smpl)
    mv = torch.from_numpy(mv).float().to('cpu')
    betas = torch.from_numpy(smpl_data[1]['betas'][t]).float().to('cpu').unsqueeze(0)
    pose = torch.from_numpy(smpl_data[1]['pose'][t]).float().to('cpu').unsqueeze(0)
    # 
    proj = camera.get_projection_matrix(width=1080, height=1080)
    # proj = torch.from_numpy(proj).float()
    proj = torch.from_numpy(proj).float().to('cpu')
    print(proj)
    # proj_new = pyrenderer_to_open_gl(proj)
    # print(proj_new)
    # Create a translation vector
    # translation = torch.tensor([0.1, 0, 0], dtype=torch.float32)

    # Apply the translation when calling smpl.forward()
    
    # body = smpl.SMPL(os.path.join(ROOT_DIR, config['smpl_path']))
    
    kin_tree = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])

    
    smpl_output = smpl.forward(
        betas=betas, 
        body_pose = pose[:, 3:], 
        global_orient=pose[:, :3]
    )
    
    
    joints_location_local = smpl_output.joints[:,:24,:]
    joints_rotation_local_mat = axisangle_to_rotationmatrix(pose)
    joints,_= local_to_global_homogenous(joints_rotation_local_mat,joints_location_local,kin_tree) 
    new_vertices = skinning(garment_vertices, joints, garment_skinning)
    
    
    vertices = new_vertices  
    faces = garment_faces
    
    # 
    faces_tensor = faces.to(torch.int32)
    glctx = dr.RasterizeCudaContext()
    shil = render(glctx, proj.cuda(), mv.cuda(), vertices[0].cuda(), faces_tensor.cuda())

    shil = shil[0].detach()
    shil =  shil[...,[-1,-1,-1]]
    save_image(shil, 'mask.png')
    # 
    mesh = trimesh.Trimesh(vertices = new_vertices[0].detach().cpu().numpy(), faces = garment_faces.detach().cpu().numpy())
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'mesh_{t}.obj'))
    # 
    mesh = trimesh.Trimesh(vertices = smpl_output.vertices[0].detach().cpu().numpy(), faces = smpl.faces)
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'smpl_{t}.obj'))
    mesh = trimesh.Trimesh(vertices = garment_vertices.detach().cpu().numpy(), faces = garment_faces.detach().cpu().numpy())
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'mesh.obj'))
    # 
    mesh = trimesh.Trimesh(vertices = smpl.v_template.detach().cpu().numpy(), faces = smpl.faces)
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'smpl.obj'))
    # mask_array = shil[0].detach().cpu().numpy()
    
    # mask_array = (mask_array * 255).astype(np.uint8)
    # # 
    # mask_image = Image.fromarray(mask_array)

    # mask_image.save('mask.png')
    # 
    # # camera_file = args.camera_pkl
    
    # with open(camera_file, 'rb') as f:
    #     camera_data = pkl_to_dict(f)

    # 
    # mtx = camera_data['mtx']
    # pos = smpl_data['pos']
    
    
if __name__=='__main__' :
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/image")
    parser.add_argument("--smpl_pkl", type=str, default="/hdd_data/nakul/soham/people_snapshot/people_snapshot_public/male-1-sport/smpl_fitting/sample_video/vibe_output.pkl")
    parser.add_argument("--camera_pkl", type=str, default="/hdd_data/nakul/soham/people_snapshot/people_snapshot_public/male-1-sport/camera.pkl")
    args = parser.parse_args()
    main(args)
    
