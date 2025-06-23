"""
Copyright 2025 University of Oxford
Author: Tom Jakab
Editor: Ben Kaye
Licence: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields

import numpy as np
import pygltflib
import torch
from pygltflib import GLTF2, Node
from torch import Tensor


@dataclass
class OneMeshGltf:
    vertices: Tensor
    faces: Tensor
    joints: Tensor
    vertex_joints: Tensor
    vertex_weights: Tensor
    inverse_bind_matrices: Tensor
    nodes_parents_list: dict[int, list[int]]
    local_joint_transforms: Tensor
    node_names: list[str]

    def __iter__(self):
        for field in dataclass_fields(self):
            yield getattr(self, field.name)


@dataclass
class SkinnedMesh(OneMeshGltf):
    posed_vertices: Tensor
    global_joint_transforms: Tensor
    skinning_matrices: Tensor


# https://github.com/KhronosGroup/glTF-Tutorials/issues/21#issuecomment-704437553
DATA_URI_HEADER_OCTET_STREAM = "data:application/octet-stream;base64,"
DATA_URI_HEADER_GLTF_BUFFER = "data:application/gltf-buffer;base64,"

# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types
NUMBER_OF_COMPONENETS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
NUMBER_OF_BYTES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}
COMPONENT_TYPE = {
    5120: "BYTE",
    5121: "UNSIGNED_BYTE",
    5122: "SHORT",
    5123: "UNSIGNED_SHORT",
    5125: "UNSIGNED_INT",
    5126: "FLOAT",
}
COMPONENT_TYPE_TO_NUMPY = {
    "BYTE": np.int8,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.int16,
    "UNSIGNED_SHORT": np.uint16,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.float32,
}


def read_data(gltf: GLTF2, accessor_id: int) -> np.ndarray:
    """

    For more info, see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types
    """
    # Get accesor
    accessor = gltf.accessors[accessor_id]

    # Read joint indices data from the buffer view
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]

    # monkey patch header constant needed for gltf.get_data_from_buffer_uri
    if DATA_URI_HEADER_OCTET_STREAM in buffer.uri:
        pygltflib.DATA_URI_HEADER = DATA_URI_HEADER_OCTET_STREAM
    elif DATA_URI_HEADER_GLTF_BUFFER in buffer.uri:
        pygltflib.DATA_URI_HEADER = DATA_URI_HEADER_GLTF_BUFFER

    # read data as bytes
    data = gltf.get_data_from_buffer_uri(buffer.uri)

    # decode data to numpy array
    data_offset = buffer_view.byteOffset + accessor.byteOffset
    element_size = (
        NUMBER_OF_COMPONENETS[accessor.type] * NUMBER_OF_BYTES[accessor.componentType]
    )
    if buffer_view.byteStride is None:
        step = element_size
    else:
        step = buffer_view.byteStride
    array = []
    for i in range(accessor.count):
        start = data_offset + i * step
        end = start + element_size
        array += [
            np.frombuffer(
                data[start:end],
                dtype=COMPONENT_TYPE_TO_NUMPY[COMPONENT_TYPE[accessor.componentType]],
            )
        ]
    array = np.stack(array)

    return array


def extract_gltf_data(
    gltf: GLTF2,
) -> OneMeshGltf:
    """
    Skin defines skeleton using joints and inverse bind matrices.
    Joints are indices to the nodes in the scene.
    Inverse bind matrices are indices to the accessor and they define the inverse of the global joint transform in its initial position.

    The vertices have to be transformed with the current global transform of the joint node.

    IMPORTANT: Vertex skinning in other contexts often involves a matrix that is called "Bind Shape Matrix". This matrix is supposed to transform the geometry of the skinned mesh into the coordinate space of the joints. In glTF, this matrix is omitted, and it is assumed that this transform is either premultiplied with the mesh data, or postmultiplied to the inverse bind matrices. Source: https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_020_Skins.html#vertex-skinning-implementation
    """

    assert len(gltf.meshes) == 1, (
        f"Only one mesh is supported. {len(gltf.meshes)} found."
    )
    assert len(gltf.skins) == 1, f"Only one skin is supported. {len(gltf.skins)} found."

    mesh_index = 0
    skin_index = 0
    mesh = gltf.meshes[mesh_index]
    skin = gltf.skins[skin_index]

    assert len(mesh.primitives) == 1, "Only one primitive is supported."
    primitive = mesh.primitives[0]

    # Read triangle indices
    assert primitive.mode == 4, "Only triangles are supported."
    indices = torch.tensor(read_data(gltf, primitive.indices), dtype=torch.int64)
    indices = indices.reshape(-1, 3)

    # Read attributes
    attributes = primitive.attributes

    # assert that is has only one attribute matching JOINTS_* and WEIGHTS_*
    assert len([k for k in dir(attributes) if "JOINTS_" in k]) == 1, (
        "Only one attribute matching JOINTS_* is supported."
    )
    assert len([k for k in dir(attributes) if "WEIGHTS_" in k]) == 1, (
        "Only one attribute matching WEIGHTS_* is supported."
    )

    # Read vertices, joint indices and weights
    vertices = torch.tensor(read_data(gltf, attributes.POSITION), dtype=torch.float64)
    vertex_joints = torch.tensor(
        read_data(gltf, attributes.JOINTS_0), dtype=torch.int64
    )
    vertex_weights = torch.tensor(
        read_data(gltf, attributes.WEIGHTS_0), dtype=torch.float64
    )

    # Read inverse bind matrices
    inverse_bind_matrices = torch.tensor(
        read_data(gltf, skin.inverseBindMatrices), dtype=torch.float64
    )

    # Note: matrices are column-major (traslation is the last row)
    # e.g.:
    #     2.0,    0.0,    0.0,    0.0,
    #     0.0,    0.866,  0.5,    0.0,
    #     0.0,   -0.25,   0.433,  0.0,
    #    10.0,   20.0,   30.0,    1.0
    inverse_bind_matrices = inverse_bind_matrices.reshape(-1, 4, 4)

    # Traverse the scene graph and get the parents of each node
    # glTF indicates only childern, so we need to traverse the scene graph from the root node
    nodes_parents_list = get_nodes_parents(gltf.nodes)

    # Compute the local joint transforms
    local_nodes_transforms = torch.tensor(
        get_local_nodes_transforms(gltf.nodes), dtype=torch.float64
    )

    # If pytorch is True, convert to torch tensors
    joints = torch.tensor(skin.joints, dtype=torch.int64)

    node_names = [node.name for node in gltf.nodes]

    return OneMeshGltf(
        vertices,
        indices,
        joints,
        vertex_joints,
        vertex_weights,
        inverse_bind_matrices,
        nodes_parents_list,
        local_nodes_transforms,
        node_names,
    )


def get_local_nodes_transforms(nodes: list[Node]) -> np.ndarray:
    transforms = np.zeros((len(nodes), 4, 4), dtype=np.float32)
    for node_index, node in enumerate(nodes):
        transforms[node_index] = get_node_transform(node)
    return transforms


def get_nodes_parents(nodes: list[Node]) -> dict[int, list[int]]:
    """
    Traverse the scene graph and list all the parents of each node.

    Args:
        nodes: list of gltf nodes

    Returns:
        Dictionary where keys are node indices and values are lists of their parent indices
    """

    # Create a dictionary where keys are nodes and values are their parents
    nodes_parents = {}

    # Recursively traverse the scene graph and fill the nodes_parents dictionary
    def traverse_scene_graph(node_index, parent):
        # If the node was already visited and has a parent, return
        if node_index in nodes_parents and nodes_parents[node_index] is not None:
            return
        nodes_parents[node_index] = parent
        for child in nodes[node_index].children:
            traverse_scene_graph(child, node_index)

    # Start the traversal from the root node (we don't know the root node, so we do it for all nodes that were not visited yet)
    for node_index in range(len(nodes)):
        traverse_scene_graph(node_index, None)

    # Now for each node create a list of all its parents
    nodes_parents_list = {}
    for node_index, parent in nodes_parents.items():
        nodes_parents_list[node_index] = []
        while parent is not None:
            nodes_parents_list[node_index].append(parent)
            parent = nodes_parents[parent]

    return nodes_parents_list


def get_node_transform(node: Node) -> np.ndarray:
    """
    Convention followed by glTF (also Pytorch3D and OpenGL?):

        M = [
            [Rxx, Ryx, Rzx, 0],
            [Rxy, Ryy, Rzy, 0],
            [Rxz, Ryz, Rzz, 0],
            [Tx,  Ty,  Tz,  1],
        ]
    """
    if node.matrix is not None:
        return np.array(node.matrix).reshape(4, 4)
    else:
        # compute the transform from translation, rotation and scale
        transform = np.eye(4)
        # first scale, then rotate, then translate
        if node.scale is not None:
            scale = np.array(node.scale)
            transform = transform @ np.diag(
                np.concatenate([scale, np.ones(1, dtype=scale.dtype)])
            )
        if node.rotation is not None:
            # rotation is a quaternion
            rotation_matrix = quaternion_to_matrix_numpy(node.rotation)
            transform = transform @ rotation_matrix
        if node.translation is not None:
            translation_matrix = np.eye(4)
            # column-major order
            translation_matrix[3, :3] = node.translation
            transform = transform @ translation_matrix
        return transform


def quaternion_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    # gltf uses quaternions in the order [x, y, z, w], but we need to convert it to [w, x, y, z] for Pytorch3D
    quaternion = torch.tensor(
        np.concatenate([quaternion[3:], quaternion[:3]]), dtype=torch.float64
    )
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = quaternion_to_matrix(quaternion).numpy().transpose()
    return rotation_matrix


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    From https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return o.reshape(*quaternions.shape[:-1], 3, 3)


def compute_global_nodes_transforms_batched(
    local_nodes_transforms: Tensor,
    nodes_indices: list[int],
    node_parents_dict: dict[int, list[int]],
) -> Tensor:
    global_nodes_transforms = []
    for _local_nodes_transforms, _nodes_indices, _nodes_parents_list in zip(
        local_nodes_transforms, nodes_indices, node_parents_dict, strict=False
    ):
        global_nodes_transforms.append(
            compute_global_nodes_transforms(
                _local_nodes_transforms, _nodes_indices, _nodes_parents_list
            )
        )
    return torch.stack(global_nodes_transforms)


def compute_global_nodes_transforms(
    local_nodes_transforms: Tensor,
    nodes_indices: list[int],
    node_parents_dict: dict[int, list[int]],
) -> Tensor:
    """
    Follows the node hierarchy to compute the global node transforms in PyTorch.

    Parameters:
    local_nodes_transforms: torch.Tensor
        Local transformations of nodes, shape (num_nodes, 4, 4)
    nodes_indices: list of int
        Indices of nodes for which to compute the global transformations.
    nodes_parents_list: list of list of int
        Each sublist contains the indices of parent nodes for the corresponding node.

    Returns:
    global_nodes_transforms: torch.Tensor
        Global transformations of nodes, shape (len(nodes_indices), 4, 4)
    """
    global_nodes_transforms = []
    for node_index in nodes_indices:
        global_node_transform = local_nodes_transforms[node_index]
        # Iterate over all parents and multiply their matrices to the current node matrix
        for parent_index in node_parents_dict[int(node_index)]:
            parent_matrix = local_nodes_transforms[parent_index]
            global_node_transform = torch.matmul(global_node_transform, parent_matrix)
        global_nodes_transforms.append(global_node_transform)

    # Stack the list of tensors into a single tensor
    global_nodes_transforms = torch.stack(global_nodes_transforms)

    return global_nodes_transforms


def compute_skinning_matrices(
    global_joint_transforms: Tensor,
    inverse_bind_matrices: Tensor,
    joints: Tensor,
    weights: Tensor,
) -> Tensor:
    """
    Computes skinning matrices for each vertex, with an optional batch dimension.

    Parameters:
    global_joint_transforms: torch.Tensor of shape (batch_size, num_joints, 4, 4) or (num_joints, 4, 4)
        Global joint transformations for each joint.
    inverse_bind_matrices: torch.Tensor of shape (batch_size, num_joints, 4, 4) or (num_joints, 4, 4)
        Inverse bind matrices for each joint.
    joints: torch.Tensor of shape (batch_size, num_vertices, 4) or (num_vertices, 4)
        Indices of joints affecting each vertex.
    weights: torch.Tensor of shape (batch_size, num_vertices, 4) or (num_vertices, 4)
        Weights of the influence of each joint on each vertex.

    Returns:
    skinning_matrices: torch.Tensor of shape (batch_size, num_vertices, 4, 4) or (num_vertices, 4, 4)
        The computed skinning matrices for each vertex.
    """
    # Detect if the input is batched
    is_batched = global_joint_transforms.dim() == 4

    # Ensure inputs are batched
    if not is_batched:
        global_joint_transforms = global_joint_transforms.unsqueeze(0)
        inverse_bind_matrices = inverse_bind_matrices.unsqueeze(0)
        joints = joints.unsqueeze(0)
        weights = weights.unsqueeze(0)

    # Compute Joint Matrices by batch matrix multiplication
    joint_matrices = torch.matmul(inverse_bind_matrices, global_joint_transforms)

    # Expand and prepare joints for batched indexing
    batch_size, num_vertices = joints.shape[:2]
    batch_indices = torch.arange(batch_size, device=joints.device)[
        :, None, None
    ].expand(-1, num_vertices, 4)

    # Select and weight joint matrices
    selected_joint_matrices = joint_matrices[batch_indices, joints]
    weighted_joint_matrices = selected_joint_matrices * weights[..., None, None]
    skinning_matrices = weighted_joint_matrices.sum(
        dim=2
    )  # Sum over the joint dimension

    # Remove the batch dimension if it was added
    if not is_batched:
        skinning_matrices = skinning_matrices.squeeze(0)

    return skinning_matrices


def skin_mesh(
    gltf_mesh: OneMeshGltf,
) -> tuple[Tensor, Tensor, Tensor]:
    vertices = gltf_mesh.vertices

    if vertices.dim() == 3:
        global_joint_transforms = compute_global_nodes_transforms_batched(
            gltf_mesh.local_joint_transforms,
            gltf_mesh.joints,
            gltf_mesh.nodes_parents_list,
        )
    else:
        global_joint_transforms = compute_global_nodes_transforms(
            gltf_mesh.local_joint_transforms,
            gltf_mesh.joints,
            gltf_mesh.nodes_parents_list,
        )

    # Compute the skinning matrices
    skinning_matrices = compute_skinning_matrices(
        global_joint_transforms,
        gltf_mesh.inverse_bind_matrices,
        gltf_mesh.vertex_joints,
        gltf_mesh.vertex_weights,
    )

    # Transform the vertices
    transformed_vertices = transform_vertices(vertices, skinning_matrices)

    return transformed_vertices, global_joint_transforms, skinning_matrices


def transform_vertices(vertices: Tensor, transform_transpose: Tensor):
    """
    Transform vertices using the given transform.

    Args:
        vertices: Tensor of shape (..., 3)
        transform_transpose: Tensor of shape (..., 4, 4)
    """
    assert transform_transpose[..., :3, 3].abs().max() < 1e-6, (
        "Error: Transform might not be transposed (col major)"
    )
    return (vertices[..., None, :] @ transform_transpose[..., :3, :3])[
        ..., 0, :
    ] + transform_transpose[..., 3, :3]
