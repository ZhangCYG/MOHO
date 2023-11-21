import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mcubes
from icecream import ic
from .layers import grid_sample
from .geom_utils import se3_to_matrix
from pytorch3d.transforms import Transform3d
import time


def cat_z_point(points, z):
    '''
    input:
        points: [nScene, nPoints, 3]
        z:      [nScene, nPoints, D]
    return:
        z_p:    [nScene, nPoint, 3+D]
    '''
    nScene, P, D = z.shape
    z_p = torch.cat([points, z], dim=-1)
    z_p = z_p.reshape(nScene, P, 3 + D)
    return z_p
    

def cat_z_hA(z):
    '''
    glb:    [nScene, dim]
    local:  [nScene, H*W, dim], list who has matched with coord3d
    '''
    glb, local, dst_points = z
    out = torch.cat([(glb.unsqueeze(1) + local), dst_points], -1)
    return out

def cat_z(z):
    '''
    glb:    [nScene, dim]
    local:  [nScene, H*W, dim], list who has matched with coord3d
    '''
    glb, local = z
    out = torch.cat([(glb.unsqueeze(1) + local)], -1)
    return out


def apply_transform(geom, trans):
    if not isinstance(trans, Transform3d):
        if trans.ndim == 2:
            trans = se3_to_matrix(trans)
        trans = Transform3d(matrix=trans.transpose(1, 2), device=trans.device)
    verts = trans.transform_points(geom)
    return verts


def get_dist_joint(nPoints, jsTn):
    N, P, _ = nPoints.size()
    num_j = jsTn.size(1)
    nPoints_exp = nPoints.view(N, 1, P, 3).expand(N, num_j, P, 3).reshape(N * num_j, P, 3)
    jsPoints = apply_transform(nPoints_exp, jsTn.reshape(N*num_j, 4, 4)).view(N, num_j, P, 3)
    jsPoints = jsPoints.transpose(1, 2).reshape(N, P, num_j * 3) # N, P, J*3
    return jsPoints


def sample_multi_z(xPoints, z, cTx, cam):
    '''
    input:
        xPoints:    [nScene, n_ray*n_sample, 3]
        z:          [nScene, D_l, H, W]
        cTx:        [nScene, 4, 4]
        cam:        [nScene, 3, 3]
    return:
        zs:         [nScene, n_points, D_l]     (n_points == n_ray*n_sample)
    '''
    n_scene = xPoints.shape[0]
    n_points = xPoints.shape[1]
    dim_local = z.shape[1]
    
    ndcPoints = proj_x_ndc(xPoints, cTx, cam)  # (nScene, 65536, 2)
    ndcPoints[..., 0] = ndcPoints[..., 0] * 2 / (2 * cam[..., None, 0, -1] - 1) - 1
    ndcPoints[..., 1] = ndcPoints[..., 1] * 2 / (2 * cam[..., None, 1, -1] - 1) - 1
    
    torch.clamp(ndcPoints, -1.01, 1.01, out=ndcPoints)
    zs = grid_sample(z, ndcPoints.view(n_scene, n_points, 1, 2))    # [n_scene, D_l, n_points, 1]
    zs = zs.permute(0, 2, 3, 1).view(n_scene, n_points, dim_local)
    
    return zs       # [nScene, n_points, D_l]


def proj_x_ndc(xPoints, cTx, cam):
    device = xPoints.device
    if cam.shape[-1] == 3:
        tmp = torch.zeros(*cam.shape[: -1], 4, dtype=cam.dtype, device=device)
        tmp[..., :3, :3] = cam 
        cam = tmp
    coords3d_hom = torch.cat([xPoints, torch.ones(*xPoints.shape[: -1], 1).to(device)], dim=-1)
    coords2d_hom = cam @ (cTx @ coords3d_hom.transpose(-1, -2))     # [nScene, 3, n_ray*n_sample]
    coords_2d = coords2d_hom / coords2d_hom[..., 2: , :]              # divide depth, to [u, v, 1]
    coords_2d = coords_2d[..., :2, :]
    return coords_2d.transpose(-1, -2).contiguous()                 # [nScene, n_ray*n_sample. 2]


def extract_fields(bound_min, bound_max, resolution, query_func, ref_feature, jsTo, pose, intr):
    '''
    return:
        point cloud:    [n_scenes, resolution, resolution, resolution]
    '''
    device = pose.device
    
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).to(device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).to(device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).to(device).split(N)
    
    if len(pose.shape) < 3:
        pose = pose[None]
        intr = intr[None]
    n_scenes = pose.shape[0]

    u = np.zeros([n_scenes, resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    pts = pts[None].expand(n_scenes, *pts.shape)        # [n_scenes, n_points, 3]
                    glb, local = ref_feature
                    local = sample_multi_z(pts, local, pose, intr)
                    dstPoints = get_dist_joint(pts, jsTo)
                    latent = cat_z_hA((glb, local, dstPoints))
                    points = cat_z_point(pts, latent)
                    points = points.reshape(-1, points.shape[-1])       # [n_scenes*n_points, 3]
                    val = query_func(points).reshape(n_scenes, len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[:, xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, ref_feature, jsTo, pose, intr, resolution, threshold, query_func):
    '''
    return:
        two lists of 'n_scenes' geometries
    '''
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, ref_feature, jsTo, pose, intr)
    
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    vert_list, tria_list = [], []
    
    for idx in range(u.shape[0]):
        us = u[idx]
        vertices, triangles = mcubes.marching_cubes(us, threshold)
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        vert_list.append(vertices)
        tria_list.append(triangles)
    
    if u.shape[0] == 1:
        return vert_list[0], tria_list[0]
    else:
        return vert_list, tria_list


def sample_pdf(bins, weights, n_importance, det=False):
    '''
    input:
        bins(z_vals):               [nScene, batch_size(n_rays), n_samples]
        weights:                    [nScene, batch_size(n_rays), n_samples - 1]
        n_importance:               int
    return:
        new_z_vals:                 [nScene, batch_size(n_rays), n_importance]
    '''
    # This implementation is from NeRF
    device = bins.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)        # [nScene, batch_size, n_samples - 1]
    cdf = torch.cumsum(pdf, -1)                                 # [nScene, batch_size, n_samples - 1]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [nScene, batch_size, n_samples]
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_importance, 1. - 0.5 / n_importance, steps=n_importance)
        u = u.expand(list(cdf.shape[:-1]) + [n_importance]).to(device)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_importance]).to(device)   # [nScene, batch_size, n_importance] 
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)                    # [nScene, batch_size, n_importance, 2]

    matched_shape = [*inds_g.shape[: -1], cdf.shape[-1]]        # [nScene, batch_size, n_importance, n_samples]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        input:
            rays_o, rays_d: [nScene, batch_size(n_rays), 3]
            z_vals:         [nScene, batch_size(n_rays), n_samples]
        """
        n_scenes, batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist * torch.ones_like(dists[..., :1])], -1)
        mid_z_vals = z_vals + dists * 0.5           # n_scenes, batch_size, n_samples

        # Section midpoints
        pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * mid_z_vals[:, :, :, None]  # n_scenes, batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # n_scenes, batch_size, n_samples, 4

        dirs = rays_d[:, :, None, :].expand(n_scenes, batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))          # n_scenes*batch_size*n_samples, 4
        dirs = dirs.reshape(-1, 3)                                  # n_scenes*batch_size*n_samples, 3

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(n_scenes, batch_size, n_samples)) * dists)
        alpha = alpha.reshape(n_scenes, batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-7], -1), -1)[..., :-1]
        sampled_color = sampled_color.reshape(n_scenes, batch_size, n_samples, 3)
        color = (weights[..., None] * sampled_color).sum(dim=-2)        # [n_scenes, batch_size, 3], color for each ray
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,                 # [n_scenes, batch_size, n_samples]
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        input:
            rays_o, rays_d: [nScene, batch_size(n_rays), 3]
            z_vals:         [nScene, batch_size(n_rays), n_samples]
            sdf:            [nScene, batch_size(n_rays), n_samples]
        """
        n_scenes, batch_size, n_samples = z_vals.shape
        pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_vals[:, :, :, None]     # [nScene, batch_size, n_sample, 3]
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)                   # [nScene, batch_size, n_sample]
        inside_sphere = (radius[..., :-1] < 1.0) | (radius[..., 1:] < 1.0)
        sdf = sdf.reshape(n_scenes, batch_size, n_samples)
        prev_sdf, next_sdf = sdf[..., :-1], sdf[..., 1:]
        prev_z_vals, next_z_vals = z_vals[..., :-1], z_vals[..., 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros_like(cos_val[..., :1]), cos_val[..., :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-7], -1), -1)[..., :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, ref_feature, jsTo, pose, intr, sdf, last=False):
        """
        input:
            rays_o, rays_d:         [nScene, batch_size(n_rays), 3]
            z_vals:                 [nScene, batch_size(n_rays), n_samples]
            new_z_vals:             [nScene, batch_size(n_rays), n_importance]
            ref_feature:            clb: [nScene, D_g], local: [nScene, D_l, H, W]
            pose, intr:             [nScene, 4, 4], [nScene, 3, 3]
            sdf:                    [nScenes, n_rays, n_sample]
        """
        device = rays_o.device
        n_scenes, batch_size, n_samples = z_vals.shape
        n_importance = new_z_vals.shape[-1]
        pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * new_z_vals[:, :, :, None]  # [nScene, batch_size, n_importance, 3]
        dirs = rays_d[:, :, None, :].expand(pts.shape)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)      # both [nScene, batch_size(n_rays), n_samples + n_importance]

        if not last:
            pts = pts.reshape(n_scenes, -1, 3)
            dirs = dirs.reshape(n_scenes, -1, 3)
            glb, local = ref_feature
            local = sample_multi_z(pts, local, pose, intr)
            dstPoints = get_dist_joint(pts, jsTo)
            latent = cat_z_hA((glb, local, dstPoints))
            points = cat_z_point(pts, latent)
            new_sdf = self.sdf_network(points.reshape(-1, points.shape[-1]), get_sdf=True)  # forward: [nScene*batch_size*n_importance, dim]
            new_sdf = new_sdf.reshape(n_scenes, batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size, device=device)[None, :, None].expand(n_scenes, 
                                                                               batch_size, 
                                                                               n_samples + n_importance).reshape(n_scenes, -1)
            index = index.reshape(n_scenes, -1)     # xx: index of dim -2; index: index of dim -1
            sdf_res = torch.zeros_like(sdf)
            for idx in range(n_scenes):
                sdf_res[idx] = sdf[idx][(xx[idx], index[idx])].reshape(batch_size, n_samples + n_importance)
        else:
            sdf_res = sdf

        return z_vals, sdf_res

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    pose = None,
                    intr = None,
                    ref_feature = None,
                    jsTo = None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        '''
        input:
            rays_o, rays_d:         [nScene, batch_size(n_rays), 3]
            z_vals:                 [nScene, batch_size(n_rays), n_samples]
            pose, intr:             [nScene, 4, 4], [nScene, 3, 3]
            ref_feature:            global: [nScene, dim], local: [nScene, dim, H, W]
        '''
        device = rays_o.device
        n_scenes, batch_size, n_samples = z_vals.shape

        # Section length
        # calculate distance between two samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # concatenate a last distance
        dists = torch.cat([dists, sample_dist * torch.ones_like(dists[..., :1])], -1)
        # samples in the middle of the original samples
        mid_z_vals = z_vals + dists * 0.5                   # [n_scenes, n_rays, n_samples]

        # Section midpoints
        pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * mid_z_vals[:, :, :, None]  # n_scenes, n_rays, n_samples, 3
        dirs = rays_d[:, :, None, :].expand(pts.shape)

        pts = pts.reshape(n_scenes, -1, 3)
        dirs = dirs.reshape(n_scenes, -1, 3)

        glb, local = ref_feature
        local = sample_multi_z(pts, local, pose, intr)
        dstPoints = get_dist_joint(pts, jsTo)
        latent = cat_z_hA((glb, local, dstPoints))
        points = cat_z_point(pts, latent)
        
        points = points.reshape(-1, points.shape[-1])       # [n_scenes*n_rays*n_samples, 3+dim]
        dirs = dirs.reshape(-1, dirs.shape[-1])             # [n_scenes*n_rays*n_samples, 3]

        sdf_nn_output = sdf_network(points)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network(points, get_gradient=True).squeeze()
        latent_c = cat_z((glb, local))
        points_c = cat_z_point(pts, latent_c)
        points_c = points_c.reshape(-1, points_c.shape[-1])
        sampled_color = color_network(points_c, gradients, dirs, feature_vector)
        sampled_color = sampled_color.reshape(n_scenes, batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]).to(device))[None, :, :1].clip(1e-6, 1e6)     # Single parameter, [1, 1, 1]
        inv_s = inv_s.expand(n_scenes, batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)     # [n_scenes*n_rays*n_samples, 1]

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive, [n_scenes*n_rays*n_samples, 1]

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5    # [n_scenes*n_rays*n_samples, 1]
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5    # [n_scenes*n_rays*n_samples, 1]

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s.reshape(-1, 1))     # [n_scenes*n_rays*n_samples, 1]
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s.reshape(-1, 1))     # [n_scenes*n_rays*n_samples, 1]

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(n_scenes, batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(n_scenes, batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[..., :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[..., n_samples: ]], dim=-1)
            sampled_color = sampled_color * inside_sphere[..., None] + \
                            background_sampled_color[..., :n_samples] * (1.0 - inside_sphere)[..., None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[..., n_samples: ]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-7], -1), -1)[..., :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)     # [n_scenes, batch_size, 1]

        color = (sampled_color * weights[..., None]).sum(dim=-2)    # [n_scenes, batch_size, 3]
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(n_scenes, batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,                                                         # [n_scenes, n_rays, 3]
            'sdf': sdf.reshape(n_scenes, batch_size, n_samples),                    # [n_scenes, n_rays, n_samples] 
            'dists': dists,                                                         # [n_scenes, n_rays, n_samples]
            'gradients': gradients.reshape(n_scenes, batch_size, n_samples, 3),     # [n_scenes, n_rays, n_samples, 3]
            's_val': 1.0 / inv_s,                                                   # [n_scenes, n_rays * n_samples, 1]
            'mid_z_vals': mid_z_vals,                                               # [n_scenes, n_rays, n_samples]
            'weights': weights,                                                     # [n_scenes, n_rays, n_samples]
            'cdf': c.reshape(n_scenes, batch_size, n_samples),                      # [n_scenes, n_rays, n_samples]
            'gradient_error': gradient_error,                                       # [1]
            'inside_sphere': inside_sphere                                          # [n_scenes, n_rays, n_samples]
        }

    def render(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor, 
               pose: torch.Tensor = None, intr: torch.Tensor = None, ref_feature: torch.Tensor = None, jsTo=None, 
               perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        '''
        rays_o, rays_d: [nScene, batch_size, 3]
        near, far:      [nScene, batch_size, 1]
        pose, intr:     [nScene, 4, 4], [nScene, 3, 3]
        ref_feature:    (global: [nScene, D_g], local: [nScene, D_l, H, W])
        
        batch_size: n_rays
        n_sample: n_points for a ray
        '''
        device = rays_o.device
        
        n_scenes = rays_o.shape[0]
        batch_size = rays_o.shape[1]
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, None, :]        # [nScene, batch_size, n_samples]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside).to(device)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([n_scenes, batch_size, 1]) - 0.5).to(device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([n_scenes, batch_size, z_vals_outside.shape[-1]]).to(device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_vals[:, :, :, None]     # [nScene, batch_size, n_sample, 3]
                dirs = rays_d[:, :, None, :].expand(pts.shape)
                pts = pts.reshape(n_scenes, -1, 3)
                dirs = dirs.reshape(n_scenes, -1, 3)                                             # [nScene, batch_size*n_sample, 3]
                glb, local = ref_feature                                                        # [nScene, D_g], [nScene, D_l, H, W]
                local = sample_multi_z(pts, local, pose, intr)
                dstPoints = get_dist_joint(pts, jsTo)
                latent = cat_z_hA((glb, local, dstPoints))
                points = cat_z_point(pts, latent)                                   # [nScenes, n_rays*n_sample, 3+dim]
                sdf = self.sdf_network(points.reshape(-1, points.shape[-1]), get_sdf=True)    # [nScenes*n_rays*n_sample, 3+dim]
                sdf = sdf.reshape(n_scenes, batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  ref_feature,
                                                  jsTo,
                                                  pose, 
                                                  intr,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    pose=pose,
                                    intr=intr,
                                    ref_feature = ref_feature,
                                    jsTo=jsTo,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)     # [n_scenes, batch_size, 1]
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(n_scenes, batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,                                   # [n_scenes, n_rays, 3]
            's_val': s_val,                                             # [n_scenes, n_rays, 1]
            'cdf_fine': ret_fine['cdf'],                                # [n_scenes, n_rays, n_samples]
            'weight_sum': weights_sum,                                  # [n_scenes, n_rays, 1]
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],  # [n_scenes, n_rays, 1]
            'gradients': gradients,                                     # [n_scenes, n_rays, n_samples, 3]
            'weights': weights,                                         # [n_scenes, n_rays, n_samples]
            'gradient_error': ret_fine['gradient_error'],               # [1]
            'inside_sphere': ret_fine['inside_sphere']                  # [n_scenes, n_rays, n_samples]
        }

    def extract_geometry(self, bound_min, bound_max, resolution, ref_feature=None, jsTo=None, pose=None, intr=None, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                ref_feature=ref_feature,
                                jsTo=jsTo,
                                pose=pose,
                                intr=intr,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network(pts, get_sdf=True))