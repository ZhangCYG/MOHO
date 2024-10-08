import torch
import torch.nn.functional as F
from torch.distributed import get_world_size, get_rank
from torch.utils.data import Dataset, Sampler
from torch.utils.data.sampler import RandomSampler, BatchSampler

import math
import cv2 as cv
import numpy as np
import scipy
import os
from pathlib import Path
from glob import glob
import pickle
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import random
from tqdm import tqdm
from torchvision.transforms import Resize


def get_mask_bbox(mask: torch.Tensor, expand=2):
    h_m = torch.where(mask.sum(dim=0) != 0)[0]
    w_m = torch.where(mask.sum(dim=1) != 0)[0]
    x_min = max(h_m.min().item() - expand       , 0)
    x_max = min(h_m.max().item() + expand + 1   , mask.shape[1])
    y_min = max(w_m.min().item() - expand       , 0)
    y_max = min(w_m.max().item() + expand + 1   , mask.shape[0])
    return x_min, x_max, y_min, y_max


class HO3DDataset(Dataset):
    def __init__(self, conf, device='cuda', train_inner_iter=None, ray_batch_size=512):
        print('Load ho3d data: Begin')
        self.device = torch.device(device)
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.cache_dir = conf.get_string('cache_dir')
        self.seg_dir = conf.get_string('seg_dir', default=None)
        if self.seg_dir is None:
            print('seg is None!')
        
        self.cache = self.reidx_cache()
        self.cat = conf.get_string('cat')
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.total_inner_iter = conf.get_int('inner_iter', default=10)
        self.train_inner_iter = self.total_inner_iter if train_inner_iter is None else train_inner_iter
        self.ray_batch_size = ray_batch_size

        self.womask = conf.get_bool('womask', default=False)

        self.ref_mask = conf.get_bool('ref_mask', default=False)

        # only use in `gen_random_rays_at` for training
        self.use_bbox = True

        self.scene_lis = list(self.cache.keys())

        self.n_scenes = len(self.scene_lis)

        self.object_bbox_min = np.array([-0.2, -0.2, -0.2])
        self.object_bbox_max = np.array([0.2, 0.2, 0.2])

        # cache
        self.scene_idx_cache = None

        print('Load data: End')

    def reidx_cache(self):
        cache = pickle.load(open(self.cache_dir, 'rb'))
        new_dict = {}
        for i, index in tqdm(enumerate(cache['index'])):
            if index[1] not in new_dict.keys():
                if os.path.exists(os.path.join(self.data_dir, '{}', '{}', 'rgb', '{}.png').format(*cache['index'][i])):
                    mask_p = os.path.join(self.data_dir, '{}', '{}', 'seg', '{}.jpg').format(*cache['index'][i])
                    mask = cv.imread(mask_p)[:, :, 0] > 100
                    if mask.sum() > 50:
                        new_dict[index[1]] = {
                            'index': [cache['index'][i]],
                            'cad_index': [cache['cad_index'][i]],
                            'hA': [cache['hA'][i]],
                            'hTo': [cache['hTo'][i]],
                            'cTh': [cache['cTh'][i]],
                            'cam': [cache['cam'][i]],
                        }
            else:
                if os.path.exists(os.path.join(self.data_dir, '{}', '{}', 'rgb', '{}.png').format(*cache['index'][i])):
                    mask_p = os.path.join(self.data_dir, '{}', '{}', 'seg', '{}.jpg').format(*cache['index'][i])
                    mask = cv.imread(mask_p)[:, :, 0] > 100
                    if mask.sum() > 50:
                        new_dict[index[1]]['index'].append(cache['index'][i])
                        new_dict[index[1]]['cad_index'].append(cache['cad_index'][i])
                        new_dict[index[1]]['hA'].append(cache['hA'][i])
                        new_dict[index[1]]['hTo'].append(cache['hTo'][i])
                        new_dict[index[1]]['cTh'].append(cache['cTh'][i])
                        new_dict[index[1]]['cam'].append(cache['cam'][i])

        for scene in new_dict.keys():
            print(scene, ': ', len(new_dict[scene]['index']))
        return new_dict

    def check_cache(self, scene, use_ret=False, test=False):
        if self.scene_idx_cache != scene:
            self.scene_idx_cache = scene
            if isinstance(scene, str):
                # debug
                scene_name = scene
            else:
                scene_name = self.scene_lis[scene]
            index = self.cache[scene_name]['index']
            # read image, scene_name for single obj
            images_lis = [os.path.join(self.data_dir, '{}', '{}', 'rgb', '{}.png').format(*idx) for idx in index]
            if not test:
                sample_list = [i for i in range(len(images_lis))]
                sample_list = random.sample(sample_list, self.total_inner_iter)
                images_lis = [images_lis[i] for i in sample_list]
            n_images = len(images_lis)
            images_np = np.stack([cv.imread(im_name) for im_name in images_lis]) / 255.0
            images = torch.from_numpy(images_np.astype(np.float32))  # [inner_iter, H, W, 3]
            H, W = images.shape[1], images.shape[2]
            # obj mask
            masks_lis = [os.path.join(self.data_dir, '{}', '{}', 'seg', '{}.jpg').format(*idx) for idx in index]
            if not test:
                masks_lis = [masks_lis[i] for i in sample_list]
            if self.womask:
                masks_np = np.ones_like(images_np)
            else:
                masks_np = np.stack([(cv.resize(cv.imread(im_name), (W, H), cv.INTER_NEAREST)[:, :, 0] > 100) for im_name in masks_lis])
            # meta info
            cTh_list = self.cache[scene_name]['cTh']
            hTo_list = self.cache[scene_name]['hTo']
            pose_all = [cTh_list[i] @ hTo_list[i] for i in range(len(cTh_list))]
            if not test:
                pose_all = [pose_all[i] for i in sample_list]
            pose_origin_woinv = [pose.float() for pose in pose_all]
            pose_all = [torch.inverse(pose) for pose in pose_all]

            masks = torch.from_numpy(masks_np.astype(np.float32))   # [inner_iter, H, W, 3]

            intr_list = self.cache[scene_name]['cam']
            intr_list = [intr.float() for intr in intr_list]
            if not test:
                intr_list = [intr_list[i] for i in sample_list]
            intr_all = torch.stack(intr_list)
            intr_all_inv = torch.stack([torch.inverse(intr.float()) for intr in intr_list])
            pose_origin_woinv = torch.stack(pose_origin_woinv)    # [inner_iter, 4, 4]
            pose_all = torch.stack(pose_all)                      # [inner_iter, 4, 4]

            # read hand pose and hto from cache
            hA_list = self.cache[scene_name]['hA']
            hTo_list = self.cache[scene_name]['hTo']
            hA_list = [ha.float() for ha in hA_list]
            if not test:
                hA_list = [hA_list[i] for i in sample_list]
            hA_all = torch.stack(hA_list)
            hTo_list = [hto.float() for hto in hTo_list]
            if not test:
                hTo_list = [hTo_list[i] for i in sample_list]
            hTo_all = torch.stack(hTo_list)
            masks = masks.unsqueeze(-1)
            images = images * masks.expand(*images.shape)

            # dino
            dino_lis = [os.path.join(self.data_dir, '{}', '{}', 'pca3_map', '{}.png').format(*idx) for idx in index]
            if not test:
                dino_lis = [dino_lis[i] for i in sample_list]
            dino_np = np.stack([cv.resize(cv.imread(dino_name), (W, H)) for dino_name in dino_lis]) / 255.0
            dinos = torch.from_numpy(dino_np.astype(np.float32))
            dino_image = dinos * masks.expand(*dinos.shape)

            # seg
            if self.seg_dir is not None:
                seg_lis = [os.path.join(self.seg_dir, '{}', '{}', 'seg', '{}.jpg').format(*idx) for idx in index]
                if not test:
                    seg_lis = [seg_lis[i] for i in sample_list]
                seg_np = np.stack([cv.resize(cv.imread(seg_name), (W, H)) for seg_name in seg_lis]) / 255.0
                segs = torch.from_numpy(seg_np.astype(np.float32))[:, :, :, :1]
            else:
                segs = masks.clone()

            self.images_lis = images_lis
            self.n_images = n_images
            self.pose_all = pose_all
            self.pose_origin_woinv = pose_origin_woinv
            self.images = images
            self.masks = masks
            self.intr_inv = intr_all_inv
            self.intr_all = intr_all
            self.H, self.W = H, W
            self.hA = hA_all
            self.hTo = hTo_all
            self.images_lis = images_lis
            self.dino_image = dino_image
            self.segs = segs

        if use_ret:
            return (self.images, self.masks, self.segs, self.pose_all, self.intr_inv, self.H, self.W,      # for ray generation
                    self.pose_origin_woinv, self.hA, self.hTo)  # for ref

    def gen_rays_at(self, scene, view_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        self.check_cache(scene)
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # pixel coord
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        # camera coord (camera as origin point)
        p = torch.matmul(self.intr_inv[view_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # direction vector in camera coord
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[view_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        # world coord origin point's location in camera coord
        rays_o = self.pose_all[view_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        # inverse pose (both R and t, 4x4) is right, map to world coord
        return rays_o.transpose(0, 1).to(self.device), rays_v.transpose(0, 1).to(self.device)

    def gen_random_rays_at(self, scene, view_idx, batch_size, scene_data=None):
        """
        Generate random rays at world space from one camera.
        """
        if scene_data is not None:
            images, masks, segs, pose_all, intr_inv_all, H, W = scene_data
        else:
            self.check_cache(scene)
            images, masks, segs, pose_all, intr_inv_all, H, W = self.images, self.masks, self.segs, self.pose_all, self.intr_inv, self.H, self.W
        if self.use_bbox:
            mask_img = masks[view_idx]          # H, W
            x_min, x_max, y_min, y_max = get_mask_bbox(mask_img)    # inversed x & y
        else:
            x_min, x_max, y_min, y_max = (0, W, 0, H)
        pixels_x = torch.randint(low=x_min, high=x_max - 1, size=[batch_size])
        pixels_y = torch.randint(low=y_min, high=y_max - 1, size=[batch_size])
        color = images[view_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = masks[view_idx][(pixels_y, pixels_x)]      # batch_size, 1
        seg = segs[view_idx][(pixels_y, pixels_x)]      # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(intr_inv_all[view_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(pose_all[view_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose_all[view_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.to(self.device), rays_v.to(self.device), color.to(self.device), mask.to(self.device), seg.to(self.device)], dim=-1).to(self.device)   # batch_size, 10

    def gen_source_img_pose_intr(self, scene, test=False, ref=None):
        '''
        return pose without inverse
        '''
        self.check_cache(scene, test=test)
        if not test:
            ref_idx = np.random.randint(0, self.n_images)
        else:
            ref_idx = ref
        img = self.images[ref_idx].unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        pose = self.pose_origin_woinv[ref_idx]
        intr = self.intr_all[ref_idx]
        hA = self.hA[ref_idx].unsqueeze(0)
        hTo = self.hTo[ref_idx].unsqueeze(0)
        image_p = self.images_lis[ref_idx]
        dino_image = self.dino_image[ref_idx].unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        torch_resize = Resize([120, 160])
        dino_image = torch_resize(dino_image)
        return img.to(self.device), dino_image.to(self.device), pose.to(self.device), intr.to(self.device), hA.to(self.device), hTo.to(self.device), image_p

    def __getitem__(self, scene):
        scene_data = self.check_cache(scene, use_ret=True)  # (images, masks, pose_all, intr_all_inv, H, W, pose_origin_woinv, hA_all, hTo_all)
        view_perm = torch.randperm(self.n_images)
        rays_o_list, rays_d_list, true_rgb_list, mask_list, seg_list = [], [], [], [], []
        near_list, far_list = [], []
        for view_idx in range(self.train_inner_iter):
            data = self.gen_random_rays_at(scene, view_perm[view_idx % len(view_perm)], self.ray_batch_size,
                                           scene_data=scene_data[: 7])
            rays_o, rays_d, true_rgb, mask, seg = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10:]
            near, far = self.near_far_from_sphere(rays_o, rays_d)
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
            true_rgb_list.append(true_rgb)
            mask_list.append(mask)
            seg_list.append(seg)
            near_list.append(near)
            far_list.append(far)
        ref_idx = np.random.randint(0, self.n_images)
        pose_all, hA, hTo = scene_data[7:]
        image = self.images[ref_idx].permute(2, 0, 1).contiguous()
        dino_image = self.dino_image[ref_idx].permute(2, 0, 1).contiguous()
        torch_resize = Resize([120, 160])
        dino_img = torch_resize(dino_image)
        pose = pose_all[ref_idx]
        hA = hA[ref_idx]
        hTo = hTo[ref_idx]
        intr = self.intr_all[ref_idx]             # v*b is equal to 'batch_size' or 'n_rays' outside
        return (torch.cat(rays_o_list, 0), torch.cat(rays_d_list, 0),   # [v*b, 3], [v*b, 3]
                torch.cat(true_rgb_list, 0), torch.cat(mask_list, 0),  # [v*b, 3], [v*b, 1]
                torch.cat(seg_list, 0),
                torch.cat(near_list, 0), torch.cat(far_list, 0),     # [v*b, 1], [v*b, 1] [v, 4, 4]
                image.to(self.device), dino_img.to(self.device), pose.to(self.device), intr.to(self.device), hA.to(self.device), hTo.to(self.device))                                      # [3, H, W], [4, 4], [3, 3]

    def __len__(self):
        return len(self.scene_lis)

    def gen_rays_between(self, scene, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        self.check_cache(scene)
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = rays_v / torch.linalg.norm(self.pose_all[0, :3, :3], dim=(-1, -2))
        pose_0 = self.pose_origin_woinv[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_origin_woinv[idx_1].detach().cpu().numpy()
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).to(self.device)
        trans = torch.from_numpy(pose[:3, 3]).to(self.device)
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        # near = mid - 0.15
        # far = mid + 0.15
        near = 0.2 * torch.ones_like(mid)
        far = 1.2 * torch.ones_like(mid)
        return near, far

    def image_at(self, scene, view_idx, resolution_level):
        self.check_cache(scene)
        # img = cv.imread(self.images_lis[view_idx])
        img = (self.images[view_idx].numpy() * 255).astype(np.uint8)
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
    
    def mask_at(self, scene, view_idx, resolution_level):
        self.check_cache(scene)
        mask = (self.masks[view_idx].numpy() * 255).astype(np.uint8)
        mask = (cv.resize(mask, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
        return np.expand_dims(mask, axis=-1)


class IterationBasedBatchSampler(Sampler):
    """
    Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter
    

def get_iter_sampler(dataset, batch_size, drop_last, max_iteration, start_iteration=0):
    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, max_iteration, start_iteration)
    return batch_sampler