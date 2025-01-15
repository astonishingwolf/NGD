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
import smpl
from smplx import SMPL
import pyrender
import trimesh 
from smpl import LBS
# import smpl
import joblib

def garment_skinning_function(v_garment_unskinned, pose, shape, body, skin_weight):
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

    smpl_vertices, joint_transforms = body(shape=shape_flat, pose=pose_flat)

    v_garment_skinning_cur = LBS()(v_garment_unskinned, joint_transforms, skin_weight)
    # v_garment_skinning_cur += translation.view(-1, 1, 3)
    # 
    # v_garment_skinning = v_garment_skinning_cur.view(-1, num_frames, v_garment_unskinned.shape[1], 3)

    return smpl_vertices,v_garment_skinning_cur
    
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
        k = 0.25

        weights = torch.exp(
            -(distances - min_dist.unsqueeze(1)) ** 2 / (k * sigma ** 2).unsqueeze(1))
        # Normalize the weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights


    garment_body_distance = torch.cdist(garment_template, smpl.template_vertices)
    
    min_dist = garment_body_distance.min(dim=1)[0]  
    
    weight = gaussian_rbf(garment_body_distance, min_dist)

    w_skinning = torch.matmul(weight, smpl.skinning_weights)
    
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

def render(glctx, proj, mv, pos, pos_idx, res: int = 1080):
    mvp = proj @ mv
    pos_clip    = transform_pos(mvp, pos)
    # print(pos_idx.dtype)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, (res,res))
    alpha = torch.clamp(rast_out[..., [-1]], max=1)

    return alpha

def pkl_to_dict(pkl_path):
    # with open(pkl_path, 'rb') as f:
    #     data = pickle.load(f)
    #     return data
    data = joblib.load(pkl_path)

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
  
def main(args):
    
    smpl_pkl = args.smpl_pkl
    smpl_data = pkl_to_dict(smpl_pkl)
    # 
    smpl_model_path = '/home/nakul/soham/3d/fixnormalndepth/SMPL_NEUTRAL.pkl'
    garment_model = '/home/nakul/soham/3d/fixnormalndepth/dataset/Fashion3D/tshirt.obj'
    smpl_og = SMPL(smpl_model_path, device = 'cpu')
    t = 65
    camera = WeakPerspectiveCamera(
            scale=[smpl_data[1]['orig_cam'][t][0], smpl_data[1]['orig_cam'][t][1]],
            translation=[smpl_data[1]['orig_cam'][t][2], smpl_data[1]['orig_cam'][t][3]],
            zfar=1000.
        )

    mv = np.eye(4)
    mv[2,2] = -1
    body = smpl.SMPL('/home/nakul/soham/3d/fixnormalndepth/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    
    garment_vertices, garment_faces = load_garment(garment_model)
    garment_skinning = compute_rbf_skinning_weight(garment_vertices,body)
    
    mv = torch.from_numpy(mv).float().to('cpu')
    betas = torch.from_numpy(smpl_data[1]['betas'][t]).float().to('cpu').unsqueeze(0)
    pose = torch.from_numpy(smpl_data[1]['pose'][t]).float().to('cpu').unsqueeze(0)
    proj = camera.get_projection_matrix(width=1080, height=1080)
    proj = torch.from_numpy(proj).float().to('cpu')
    print(proj)
    print(smpl_data[1]['pose'][t])
    smpl_vertices,new_vertices = garment_skinning_function(garment_vertices.unsqueeze(0), pose, betas, body, garment_skinning)
    
    vertices = new_vertices  
    faces = garment_faces
    faces_tensor = faces.to(torch.int32)
    smpl_faces = torch.from_numpy(smpl_og.faces.astype(np.int32)).to(torch.int32).to('cuda')
    glctx = dr.RasterizeCudaContext()
    shil = render(glctx, proj.cuda(), mv.cuda(), vertices[0].cuda(), faces_tensor.cuda())
    shil_2 = render(glctx, proj.cuda(), mv.cuda(),  smpl_vertices[0].cuda(), smpl_faces.cuda())
    ## Saving the results
    shil = shil[0].detach()
    shil =  shil[...,[-1,-1,-1]]
    save_image(shil, f'mask_{t}.png')
    shil_2 = shil_2[0].detach()
    shil_2 =  shil_2[...,[-1,-1,-1]]
    save_image(shil_2, f'mask_smpl_{t}.png')
    mesh = trimesh.Trimesh(vertices = new_vertices[0].detach().cpu().numpy(), faces = garment_faces.detach().cpu().numpy())
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'mesh_{t}.obj'))
    # 
    mesh = trimesh.Trimesh(vertices = smpl_vertices[0].detach().cpu().numpy(), faces = smpl_og.faces)
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'smpl_{t}.obj'))
    mesh = trimesh.Trimesh(vertices = garment_vertices.detach().cpu().numpy(), faces = garment_faces.detach().cpu().numpy())
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'mesh.obj'))
    # 
    mesh = trimesh.Trimesh(vertices = smpl.v_template.detach().cpu().numpy(), faces = smpl.faces)
    mesh.export(os.path.join('/home/nakul/soham/3d/fixnormalndepth/dataset/tmp',f'smpl.obj'))
    
if __name__=='__main__' :
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/home/nakul/soham/3d/fixnormalndepth/dataset/monocular/jumpingjacks/train/image")
    parser.add_argument("--smpl_pkl", type=str, default="/home/nakul/soham/3d/4D-Humans/outputs/results/demo_male-1-sport.pkl")
    parser.add_argument("--camera_pkl", type=str, default="/hdd_data/nakul/soham/people_snapshot/people_snapshot_public/male-1-sport/camera.pkl")
    args = parser.parse_args()
    main(args)
    
