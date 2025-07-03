import torch
import dualpm_paper.bones as bones
from dualpm_paper.utils import OneMeshGltf
from copy import deepcopy


def estimate_skeleton(
    canon_predictions: torch.Tensor,
    pose_predictions: torch.Tensor,
    mesh: OneMeshGltf,
    weight_threshold=0.5,
    distance_threshold=0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate the skeleton of a mesh from its predictions.

    Args:
        canon_predictions: (N, 3) canonical predictions
        pose_predictions: (N, 3) pose predictions
        mesh: (OneMeshGltf) mesh
        weight_threshold: (float) threshold for the weight of the joints
        distance_threshold: (float) threshold for the distance of the vertices

    Returns:
        reconstruction_in_canonical: (N, 3) reconstruction in the canonical space
        joints_est: (J, 4, 4) joint transforms
        success: (J,) boolean mask of the valid joints
        error: (N,) error between the canonical reconstruction and canonical predictions
    """

    lookup = bones.make_lookup(mesh.vertices, distance_threshold)

    aug_mesh = deepcopy(mesh)
    aug_mesh.inverse_bind_matrices = bones.__transpose(aug_mesh.inverse_bind_matrices)
    joints_est, success = bones.estimate_joints(
        canon_predictions,
        pose_predictions,
        aug_mesh,
        lookup,
        joint_threshold=weight_threshold,
        scale=False,
    )

    reconstruction_in_canonical, error = bones.reconstruction_to_canonical(
        pose_predictions,
        canon_predictions,
        joints_est,
        bones.__transpose(mesh.inverse_bind_matrices),
        mesh.vertex_joints,
        mesh.vertex_weights,
        success,
        lookup,
    )
    return reconstruction_in_canonical, joints_est, success, error
