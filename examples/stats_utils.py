import torch
import numpy as np 

#### utils

def mean_pairwise_cosine_similarity(vectors: torch.Tensor) -> torch.Tensor:
    """
    vectors: (N, k, 3)
    returns: (N,)
    """
    eps: float = 1e-8
    # Normalize to unit vectors
    v = vectors / (vectors.norm(dim=-1, keepdim=True) + eps)  # (N, k, 3)
    # Compute cosine similarity matrix for each batch
    sim_matrix = torch.matmul(v, v.transpose(-1, -2))  # (N, k, k)
    k = v.shape[1]
    # Exclude diagonal (self-similarity)
    mask = ~torch.eye(k, dtype=torch.bool, device=vectors.device)
    sim_values = sim_matrix[:, mask].reshape(v.shape[0], k, k - 1)
    # Average over all pairs
    return sim_values.mean(dim=(1, 2))

def largest_eigenvalue_alignment(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    vectors: (N, k, 3)
    returns: (N,)
    """
    # Normalize
    v = vectors / (vectors.norm(dim=-1, keepdim=True) + eps)  # (N, k, 3)
    # Compute M = (1/k) * sum(v v^T)
    # Using batch outer products
    M = torch.einsum('nki,nkj->nij', v, v) / v.shape[1]  # (N, 3, 3)
    # Compute eigenvalues (symmetric matrix)
    eigvals = torch.linalg.eigvalsh(M)  # (N, 3)
    # Return largest eigenvalue
    return eigvals[:, -1]


def quaternion_geodesic_distance(q1, q2, eps=1e-8):
    """
    q1, q2: (..., 4) unit quaternions
    returns: (...,) geodesic angle distance in radians
    """
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + eps)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + eps)

    dot = (q1 * q2).sum(dim=-1).abs().clamp(-1 + eps, 1 - eps)
    return 2 * torch.acos(dot)


######################### Main functions

def per_gaussian_offset_stats(offsets):
    # offsets: (N, 5, 3), N here already takes into account any masks on gaussians
    N = offsets.shape[0]
    # L1 norm per offset → (N, 5)
    l1 = offsets.abs().sum(dim=-1)

    # stats per gaussian → (N,)
    avg = l1.mean(dim=1)
    std = l1.std(dim=1)

    return avg.cpu().numpy(), std.cpu().numpy()

def per_gaussian_scale_stats(neural_scales):
    N = neural_scales.shape[0] // 5
    neural_scales = neural_scales.reshape(N, -1, 3)

    avg, std = per_gaussian_offset_stats(neural_scales)
    return avg, std


def per_gaussian_color_stats(neural_colors):
    # (N*5, 3) → (N, 5, 3)
    N = neural_colors.shape[0] // 5
    colors = neural_colors.reshape(N, -1, 3) #assert in Runner.get_neural_gaussians ensures that using this syntax preserves order as expected

    # mean/std per channel → (N, 3)
    mean_c = colors.mean(dim=1)
    std_c = colors.std(dim=1)

    # average across channels → (N,)
    avg = mean_c.mean(dim=1)
    std = std_c.mean(dim=1)

    return avg.cpu().numpy(), std.cpu().numpy()


def per_gaussian_opacity_stats(neural_opacities):
    # assuming shape (N*5, 1) or (N*5,)
    N = neural_opacities.shape[0] // 5
    op = neural_opacities.reshape(N, -1)

    avg = op.mean(dim=1)
    std = op.std(dim=1)

    return avg.cpu().numpy(), std.cpu().numpy()


def per_gaussian_offset_alignement(offsets):
    # (N*5, 3) → (N, 5, 3)
    #N = offsets.shape[0] // 5
    #offsets = offsets.reshape(N, 5, 3)

    mean_cos = mean_pairwise_cosine_similarity(offsets)
    eig_align = largest_eigenvalue_alignment(offsets)

    return mean_cos.cpu().numpy(), eig_align.cpu().numpy()


def per_gaussian_quaternion_geodesic(quaternions):
    """
    quaternions: (N*5, 4)
    returns:
        mean_dist: (N,)
        std_dist:  (N,)
    """
    N = quaternions.shape[0] // 5
    q = quaternions.view(N, 5, 4)

    # normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    # fix sign ambiguity per group (important!)
    ref = q[:, :1, :]
    dot = (q * ref).sum(dim=-1, keepdim=True)
    q = torch.where(dot < 0, -q, q)

    # pairwise geodesic distances inside each group
    # (N, 5, 5)
    qi = q.unsqueeze(2)  # (N, 5, 1, 4)
    qj = q.unsqueeze(1)  # (N, 1, 5, 4)

    dist = quaternion_geodesic_distance(qi, qj)  # (N, 5, 5)

    # exclude diagonal (self-distances)
    mask = ~torch.eye(5, dtype=torch.bool, device=q.device)
    dist_vals = dist[:, mask]

    mean_dist = dist_vals.mean(dim=1)
    std_dist = dist_vals.std(dim=1)

    return mean_dist.cpu().numpy(), std_dist.cpu().numpy()

import os
from collections import defaultdict
import os
from collections import defaultdict

def measure_compress_kB(compress_dir: str):
    """
    Measures file sizes in kilobytes (kB) per category and total.
    """

    categories = {
        "anchor_feat": "anchor_feat",
        "anchor_xyz": "anchors_",
        "factors": "factors",
        "offsets": "offsets",
        "opacities": "opacities",
        "quats": "quats",
        "scales": "scales",
        "time_features": "time_features",
        "neural_networks": "nets",
        "metadata": "meta",
    }

    stats = defaultdict(float)
    total_kb = 0.0

    for root, _, files in os.walk(compress_dir):
        for fname in files:
            fpath = os.path.join(root, fname)

            if not os.path.isfile(fpath):
                continue

            size_kb = os.path.getsize(fpath) / 1024.0
            total_kb += size_kb

            matched = False
            for key, substr in categories.items():
                if substr in fname:
                    stats[key] += size_kb
                    matched = True
                    break

            if not matched:
                raise ValueError(f"Uncategorized file found: {fname}")

    computed_sum = sum(stats.values())

    # allow small floating-point tolerance
    assert abs(total_kb - computed_sum) < 1e-6, (
        f"Size mismatch: total={total_kb}, sum(categories)={computed_sum}"
    )

    stats["total_kb"] = total_kb

    return dict(stats)

### #### main functions (all functions take in a torch tensor and must return numpy arrays) chat gpt prompt
### 
### def per_gaussian_offset_stats(offsets, means3D=None):
###     #the original offsets are of shape (N*5,3)
###     #reshape the offsets into N,5,3 shape
###     #reshape the means 3D in the same way
###     #get the l1 norm of each offset -> tensor should be of shape N,5 at this point. 
###     #get the std and mean -> tensor should be of shape N at this point
###     #reshape unitary test : substract means3D from offsets and make sure each group of 5 elements has an std of 0 (are all the same) 
###     #TODO
###     avg,std=None,None
###     return avg, std
### 
### def per_gaussian_scale_stats(neural_scales):
###     avg,std, _ = per_gaussian_offset_stats(neural_scales)
###     return avg, std
### 
### 
### def per_gaussian_color_stats(neural_colors):
###     #reshape the neural colors into N,5,3 shape (in the same way as offsets above)
###     #get the std and mean per-chanel -> tensor should be of shape N,3 at this point. 
###     #take the average accross all three chanels (for both std and mean) -> tensor should be of shape N at this point. 
###     #TODO
###     pass
### 
### def per_gaussian_opacity_stats(neural_opacities):
###     #reshape the neural colors into N,5,3 shape (in the same way as offsets above)
###     #get the std and mean per-chanel -> tensor should be of shape N at this point. 
### 
### def per_gaussian_offset_alignement(offsets):
###     #the original offsets are of shape (N*5,3)
###     #reshape the offsets into N,5,3 shape
###     #passes the reshaped tensors into the functions mean_pairwise_cosine_similarity and largest_eigenvalue_alignment and return the output