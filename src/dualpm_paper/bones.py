"""
Copyright 2025 University of Oxford
Author: Ben Kaye
Licence: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import einops
from dualpm_paper.skin import OneMeshGltf
from scipy.spatial import KDTree


def to_transform(
    scale_factor: float, rotation: torch.Tensor, translation: torch.Tensor
):
    matrix = torch.zeros(*rotation.shape[:-2], 4, 4, device=rotation.device)
    matrix[..., :, :] = torch.eye(4, device=rotation.device)

    matrix[..., :3, 3] = translation
    matrix[..., :3, :3] = scale_factor * rotation[..., :3, :3]
    return matrix


def __transpose(tensor: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(tensor, "... c r -> ... r c")


def __transform(transform: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return vector @ __transpose(transform[..., :3, :3]) + transform[..., :3, 3]


def kabsch_umeyama(
    a: torch.Tensor, b: torch.Tensor, rotation: bool, scale: bool, reflection: bool
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    calculates the optimal rigid transform from a to b
    b = [sR | t] a

    umeyama's algorithm
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    a: (N, 3)
    b: (N, 3)
    rotation: bool - compute rotation?
    scale: bool - compute scale?
    reflection: bool - allow reflection?

    returns:
        scale_factor: float - scale factor
        rotation: (3, 3) - rotation matrix
        translation: (3,) - translation vector
    """

    assert a.shape == b.shape, "a,b must have the same shape"

    *_, n, m = a.shape

    a_mean = a.mean(axis=-2)
    b_mean = b.mean(axis=-2)

    A = a - a_mean
    B = b - b_mean

    # normalized covariance <(3, n) @ (n, 3)>
    covariance = (__transpose(B) @ A) / n

    U, D, Vh = torch.linalg.svd(covariance)

    # correct signs, no reflection
    Signs = torch.diag(
        [
            1,
            1,
            torch.sign(torch.linalg.det(U) * torch.linalg.det(Vh))
            if not reflection
            else 1,
        ]
    )

    rotation = U @ Signs @ Vh if rotation else torch.eye(3)

    var_A = (A**2).sum(axis=-1).mean(axis=-1)
    scale_factor = torch.trace(torch.diag(D) @ Signs) / var_A if scale else 1

    translation = b_mean - scale_factor * rotation @ a_mean
    return scale_factor, rotation, translation


def get_point_correspondence(
    predicted_canon: torch.Tensor,
    vertex_lookup: callable,
):
    """
    get the correspondence from the canonical prediction and canonical-space mesh

    args:
        predicted_canon: (N, 3) - canonical vertices of the predicted mesh
        vertex_lookup: callable - lookup function that returns the indices of the closest vertices in the target canonical vertices

    returns:
        canonical_indices: (P,) - indices of the closest vertices in the target canonical vertices
            note: P <= N, == sum(inlier_mask)
        inlier_mask: (N,) - boolean mask of the valid vertices
    """
    with torch.no_grad():
        canonical_indices, inlier_mask = vertex_lookup(predicted_canon)

    return canonical_indices, inlier_mask


def estimate_joints(
    predicted_canon: torch.Tensor,
    predicted_pose: torch.Tensor,
    skinned_mesh: OneMeshGltf,
    vertex_lookup: callable,
    joint_threshold: float,
    scale: bool,
):
    """
    1. get the correspondence from the canonical prediction and canonical-space mesh
    2. estimate the global joint transforms from the prior mesh (inverse_bind @ skinned_mesh.vertices) to posed point cloud
    3. apply to the prior mesh, and return
    """

    assert predicted_canon.shape[-1] == 3, "must be shape (P 3)"
    assert predicted_pose.shape[-1] == 3, "must be shape (P 3)"
    assert len(predicted_canon.shape) == 2, "must be shape (P 3)"
    assert len(predicted_pose.shape) == 2, "must be shape (P 3)"
    assert torch.allclose(
        skinned_mesh.inverse_bind_matrices[:, 3, :3],
        torch.zeros(
            skinned_mesh.inverse_bind_matrices.shape[0],
            1,
            3,
            device=skinned_mesh.inverse_bind_matrices.device,
        ),
    ), "must be column major"

    # strip gradients from the canonical points
    predicted_canon = predicted_canon.detach()

    canonical_indices, inlier_mask = get_point_correspondence(
        predicted_canon, vertex_lookup
    )

    skin_points = skinned_mesh.vertices[canonical_indices]
    predicted_pose = predicted_pose[inlier_mask]

    global_joint_transforms, success_mask = _estimate_global_joints(
        skin_points,
        predicted_pose,
        canonical_indices,
        skinned_mesh.joints,
        skinned_mesh.inverse_bind_matrices,
        skinned_mesh.vertex_joints,
        skinned_mesh.vertex_weights,
        joint_threshold,
        scale,
    )

    return global_joint_transforms, success_mask


def _estimate_global_joints(
    template_pointcloud: torch.Tensor,
    reconstruction_pointcloud: torch.Tensor,
    canonical_indices: torch.Tensor,
    mesh_joints: torch.Tensor,
    inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool,
):
    """
    estimate the global joint transforms from the prior mesh (inverse_bind @ skinned_mesh.vertices) to posed point cloud

    args:
        template_pointcloud: (N, 3) - canonical vertices of the prior mesh
        reconstruction_pointcloud: (P, 3) - posed vertices of the reconstruction
        canonical_indices: (P,) - indices of the closest vertices in the target canonical vertices
        mesh_joints: (J,) - indices of the joints in the prior mesh
        inverse_bind: (J, 4, 4) - inverse bind matrices of the prior mesh
        mesh_vertex_joints: (V,) - indices of the joints for each vertex in the prior mesh
        mesh_vertex_weights: (V,) - weights for each vertex in the prior mesh
        joint_threshold: float - threshold for the joint weights

    returns:
        joint_transforms_est: (J, 4, 4) - estimated global joint transforms
    """

    _device = template_pointcloud.device

    joint_transforms_est = torch.zeros_like(inverse_bind, device=_device)
    joint_transforms_est[..., :4, :4] = torch.eye(4, device=_device)

    success_mask = torch.zeros(inverse_bind.shape[0], dtype=torch.bool, device=_device)

    for bone_index in mesh_joints:
        # get the vertex weights for the prior mesh
        transform = _single_joint_transform_est(
            bone_index,
            template_pointcloud,
            reconstruction_pointcloud,
            canonical_indices,
            inverse_bind,
            mesh_vertex_joints,
            mesh_vertex_weights,
            joint_threshold,
            scale,
        )

        if transform is None:
            continue

        joint_transforms_est[bone_index] = transform
        success_mask[bone_index] = True

    return joint_transforms_est, success_mask


def _single_joint_transform_est(
    joint_index: int,
    template_pointcloud: torch.Tensor,
    reconstruction_pointcloud: torch.Tensor,
    canonical_indices: torch.Tensor,
    joint_inverse_bind: torch.Tensor,
    mesh_vertex_joints: torch.Tensor,
    mesh_vertex_weights: torch.Tensor,
    joint_threshold: float,
    scale: bool,
) -> torch.Tensor:
    joint_mask = (mesh_vertex_joints == joint_index).nonzero(as_tuple=True)
    vert_weights = mesh_vertex_weights[joint_mask]

    # drop vertices that have weights below threshold
    thresholded_mask = tuple(j[vert_weights > joint_threshold] for j in joint_mask)

    if len(vert_weights) == 0:
        return None

    cloud_to_bone, mesh_to_bone = (
        thresholded_mask[0][None, :]
        .eq(canonical_indices[:, None])
        .nonzero(as_tuple=True)
    )

    bone_posed = reconstruction_pointcloud[cloud_to_bone]

    if len(bone_posed) == 0:
        return None

    bone_canon = template_pointcloud[cloud_to_bone]

    canon_local = __transform(joint_inverse_bind[joint_index], bone_canon)

    scale_factor, rotation, translation = kabsch_umeyama(
        canon_local,
        bone_posed,
        scale=scale,
        reflection=False,
    )

    transform = to_transform(scale_factor, rotation, translation)
    return transform


def make_lookup(
    target_canonical_vertices: torch.Tensor, distance_threshold: float | None
):
    """
    make a lookup function that returns the indices of the closest vertices in the target canonical vertices
    optionally apply a threshold on the distance to reject outliers

    args:
        target_canonical_vertices: (N, 3) - canonical vertices of the target mesh
        distance_threshold: float | None - threshold on the distance to reject outliers
            if None, all vertices are returned
            if not None, only vertices within the threshold are returned

    returns:
        lookup: callable
            lookup(query_vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
            returns:
                indices: (N,) - indices of the closest vertices in the target canonical vertices
                mask: (N,) - boolean mask of the valid vertices
    """

    kdtree = KDTree(target_canonical_vertices.cpu())

    def lookup(query_vertices: torch.Tensor):
        res = kdtree.query(query_vertices.cpu())
        dists, indices = (torch.tensor(t, device=query_vertices.device) for t in res)
        if distance_threshold is not None:
            mask = dists <= distance_threshold
        else:
            mask = torch.ones_like(dists, dtype=torch.bool)

        return indices[mask], mask

    return lookup
