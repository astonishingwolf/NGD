import pickle

import numpy as np

# from script.losses.layers import FaceNormals

import torch
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F

class SMPL(nn.Module):
    
    def __init__(self, model_path, name="smpl"):
        super(SMPL, self).__init__()

        with open(model_path, 'rb') as f:
            dd = pickle.load(f, encoding='latin1')

        self.num_shapes = dd['shapedirs'].shape[-1]
        self.num_vertices = dd["v_template"].shape[-2]
        self.num_faces = dd["f"].shape[-2]
        self.num_joints = dd["J_regressor"].shape[0]

        self.skinning_weights = torch.tensor(
            dd["weights"],
            dtype=torch.float32
        )

        self.template_vertices = torch.tensor(
            dd["v_template"],
            dtype=torch.float32
        )

        self.faces = torch.from_numpy(
            np.array(dd["f"]).astype(np.int32)
        ).to(torch.int32)

        shape = np.array(dd["shapedirs"]).reshape([-1, self.num_shapes]).T

        self.shapedirs = torch.tensor(
            shape,
            dtype=torch.float32
        )

        self.posedirs = torch.tensor(
            np.array(dd["posedirs"]).reshape([-1, np.array(dd['posedirs']).shape[-1]]).T,
            dtype=torch.float32
        )

        self.joint_regressor = torch.tensor(
            np.array(dd["J_regressor"].T.todense()),
            dtype=torch.float32
        )

        self.kintree_table = torch.tensor(
            np.array(dd['kintree_table'][0]).astype(np.int32),
            dtype=torch.int32
        )


    def forward(self, shape=None, pose=None, translation=None):
        # Add shape blendshape
        shape_blendshape = torch.matmul(shape, self.shapedirs).reshape(-1, self.num_vertices, 3)
        vs = self.template_vertices + shape_blendshape

        if pose is None:
            return vs, torch.zeros((self.num_joints, 4, 4))

        # Compute local joint locations and rotations
        pose = pose.reshape(-1, self.num_joints, 3)
        joint_rotations_local = AxisAngleToMatrix()(pose)

        joint_locations_local = torch.stack(
            [
                torch.matmul(vs[:, :, 0], self.joint_regressor),
                torch.matmul(vs[:, :, 1], self.joint_regressor),
                torch.matmul(vs[:, :, 2], self.joint_regressor)
            ],
            dim=2
        )

        # Add pose blendshape
        pose_feature = (joint_rotations_local[:, 1:, :, :] - torch.eye(3).to(pose.device)).reshape(-1, 9 * (self.num_joints - 1))

        pose_blendshape = torch.matmul(pose_feature, self.posedirs).reshape(-1, self.num_vertices, 3)
        vp = vs + pose_blendshape

        # Compute global joint transforms
        joint_transforms, joint_locations = PoseSkeleton()(
            joint_rotations_local,
            joint_locations_local,
            self.kintree_table
        )

        # Apply linear blend skinning
        v = LBS()(vp, joint_transforms, self.skinning_weights)

        # Apply translation
        if translation is not None:
            v += translation[:, None, :]

        return v, joint_transforms



class AxisAngleToMatrix(nn.Module):
    
    def __init__(self):
        super(AxisAngleToMatrix, self).__init__()

    def forward(self, axis_angle):
        """
        Converts rotations in axis-angle representation to rotation matrices

        Args:
            axis_angle: tensor of shape batch_size x 3

        Returns:
            rotation_matrix: tensor of shape batch_size x 3 x 3
        """
        initial_shape = axis_angle.shape

        axis_angle = axis_angle.reshape(-1, 3)
        batch_size = axis_angle.shape[0]

        # Compute angle and axis
        angle = torch.norm(axis_angle + 1e-8, dim=1, keepdim=True)
        axis = axis_angle / angle

        cos = torch.cos(angle).unsqueeze(-1)
        sin = torch.sin(angle).unsqueeze(-1)

        # Outer product of the axis
        outer = torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))

        # Identity matrix for batch
        eyes = torch.eye(3).to(axis.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Skew-symmetric matrix of axis
        skew_axis = Skew()(axis)

        # Rodrigues' rotation formula
        R = cos * eyes + (1 - cos) * outer + sin * skew_axis

        # Reshape back to original dimensions with 3x3 matrix
        R = R.reshape(*initial_shape[:-1], 3, 3)

        return R


class Skew(nn.Module):
    def __init__(self):
        super(Skew, self).__init__()

    def forward(self, vec):
        """
        Returns the skew-symmetric version of each 3x3 matrix in a vector.

        Args:
            vec: tensor of shape batch_size x 3

        Returns:
            rotation_matrix: tensor of shape batch_size x 3 x 3
        """
        batch_size = vec.shape[0]

        # Create skew-symmetric matrices
        skew_matrix = torch.zeros((batch_size, 3, 3), device=vec.device)
        
        skew_matrix[:, 0, 1] = -vec[:, 2]
        skew_matrix[:, 0, 2] = vec[:, 1]
        skew_matrix[:, 1, 0] = vec[:, 2]
        skew_matrix[:, 1, 2] = -vec[:, 0]
        skew_matrix[:, 2, 0] = -vec[:, 1]
        skew_matrix[:, 2, 1] = vec[:, 0]

        return skew_matrix


class PoseSkeleton(nn.Module):
    def __init__(self):
        super(PoseSkeleton, self).__init__()

    def forward(self, joint_rotations, joint_positions, parents):
        """
        Computes absolute joint locations given pose.

        Args:
            joint_rotations: batch_size x K x 3 x 3 rotation matrix of K joints
            joint_positions: batch_size x K x 3, joint locations before posing
            parents: vector of size K holding the parent id for each joint

        Returns:
            joint_transforms: Tensor of shape batch_size x K x 4 x 4 relative joint transformations for LBS.
            joint_positions_posed: Tensor of shape batch_size x K x 3, joint locations after posing
        """
        # 
        batch_size = joint_rotations.shape[0]
        num_joints = len(parents)

        def make_affine(rotation, translation):
            """
            Args:
                rotation: batch_size x 3 x 3
                translation: batch_size x 3 x 1
            Returns:
                affine_transform: batch_size x 4 x 4
            """
            rotation_homo = torch.cat([rotation, torch.zeros((batch_size, 1, 3), device=rotation.device)], dim=1)
            translation_homo = torch.cat([translation, torch.ones((batch_size, 1, 1), device=rotation.device)], dim=1)
            affine_transform = torch.cat([rotation_homo, translation_homo], dim=2)
            return affine_transform

        joint_positions = joint_positions.unsqueeze(-1)
        root_rotation = joint_rotations[:, 0, :, :]
        root_transform = make_affine(root_rotation, joint_positions[:, 0])

        # Traverse joints to compute global transformations
        transforms = [root_transform]
        for joint, parent in enumerate(parents[1:], start=1):
            position = joint_positions[:, joint] - joint_positions[:, parent]
            transform_local = make_affine(joint_rotations[:, joint], position)
            transform_global = torch.matmul(transforms[parent], transform_local)
            transforms.append(transform_global)

        transforms = torch.stack(transforms, dim=1)

        # Extract joint positions
        joint_positions_posed = transforms[:, :, :3, 3]

        # Compute affine transforms relative to initial state (i.e., t-pose)
        zeros = torch.zeros([batch_size, num_joints, 1, 1], device=joint_positions.device)
        joint_rest_positions = torch.cat([joint_positions, zeros], dim=2)
        init_bone = torch.matmul(transforms, joint_rest_positions)
        # init_bone = torch.cat([init_bone, torch.zeros_like(init_bone[..., :3])], dim=-1)
        init_bone = F.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0))
        # 
        joint_transforms = transforms - init_bone

        return joint_transforms, joint_positions_posed


class LBS(nn.Module):
    def __init__(self):
        super(LBS, self).__init__()

    def forward(self, vertices, joint_rotations, skinning_weights):
        # vertices = vertices.unsqueeze(0)
        batch_size = vertices.shape[0]
        num_joints = skinning_weights.shape[-1]
        num_vertices = vertices.shape[-2]

        W = skinning_weights
        if skinning_weights.dim() < vertices.dim():
            W = skinning_weights.unsqueeze(0).repeat(batch_size, 1, 1)

        A = joint_rotations.reshape(-1, num_joints, 16)  # Flatten rotation matrices to 4x4
        T = torch.matmul(W, A)
        T = T.reshape(-1, num_vertices, 4, 4)

        # Add homogeneous coordinate
        ones = torch.ones([batch_size, num_vertices, 1], device=vertices.device)
        # 
        vertices_homo = torch.cat([vertices, ones], dim=2)
        skinned_homo = torch.matmul(T, vertices_homo.unsqueeze(-1))
        skinned_vertices = skinned_homo[:, :, :3, 0]

        return skinned_vertices
