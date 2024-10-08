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


def select_valid_scene(scene_list, views_num, scene_proportion):
    '''
    drop view insufficient scenes.
    '''
    num_dict = {}
    for scene in scene_list:
        if num_dict.get(scene, None) is None:
            num_dict[scene] = 0
        num_dict[scene] += 1
    res_list = []
    for key, value in num_dict.items():
        if value >= views_num:
            res_list.append(key)
    if scene_proportion > 0:
        selected_indices = random.sample(range(len(res_list)), len(res_list) // scene_proportion)
        selected_list = [res_list[i] for i in selected_indices]
        return selected_list
    else:
        return res_list


def get_mask_bbox(mask: torch.Tensor, expand=2):
    h_m = torch.where(mask.sum(dim=0) != 0)[0]
    w_m = torch.where(mask.sum(dim=1) != 0)[0]
    x_min = max(h_m.min().item() - expand       , 0)
    x_max = min(h_m.max().item() + expand + 1   , mask.shape[0])
    y_min = max(w_m.min().item() - expand       , 0)
    y_max = min(w_m.max().item() + expand + 1   , mask.shape[1])
    return x_min, x_max, y_min, y_max


class MultiObjectObmanDataset(Dataset):
    def __init__(self, conf, device='cuda', 
                 train_inner_iter=None, ray_batch_size=512):
        '''
        conf['dataset']:
            'name': multi_object_obman
            'data_dir': str
            'ref_dir': str
            'inner_iter': int, default: 10 (10 views for one scene)
            optional:
                'camera_outside_sphere': bool
                'scale_mat_scale': float
        train_inner_iter:
            view number for one scene, default 'inner_iter'
        '''
        print('Load multi object obman data: Begin')
        self.device = torch.device(device)
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.ref_dir = conf.get_string('ref_dir')
        self.cache_dir = conf.get_string('cache_dir')
        self.cache = self.reidx_cache()
        self.cat = conf.get_string('cat')
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        
        self.total_inner_iter = conf.get_int('inner_iter', default=10)
        self.train_inner_iter = self.total_inner_iter if train_inner_iter is None else train_inner_iter
        self.ray_batch_size = ray_batch_size
        
        self.womask = conf.get_bool('womask', default=False)
        
        self.ref_mask = conf.get_bool('ref_mask', default=False)
        self.scene_proportion = conf.get_int('scene_proportion', default=-1)
        
        # only use in `gen_random_rays_at` for training 
        self.use_bbox = True

        # read scene, retain scenes whose views are equal to inner_iter
        total_image = glob(os.path.join(self.data_dir, 'rgb/*.jpg'))
        self.scene_lis = [Path(img).name[: 8] for img in total_image]
        self.scene_lis = select_valid_scene(self.scene_lis, self.total_inner_iter, self.scene_proportion)
        if self.cat != '0':
            self.scene_lis = self.select_cat(self.cat)
        print(len(self.scene_lis))

        self.n_scenes = len(self.scene_lis)

        # placeholder   
        self.scale_mats_np = [np.eye(4, dtype=np.float32) for _ in range(self.total_inner_iter)]    # [inner_iter, 4, 4]
        # global intr
        self.intrinsics_all = [torch.tensor([[480, 0, 128], [0, 480, 128], [0, 0, 1.]]).float() for _ in range(self.total_inner_iter)]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [inner_iter, 3, 3]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)             # [inner_iter, 3, 3]
        # blender 2 cv
        self.transf_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]])

        self.object_bbox_min = np.array([-1.0, -1.0, -1.0])
        self.object_bbox_max = np.array([ 1.0,  1.0,  1.0])

        # cache 
        self.scene_idx_cache = None

        print('Load data: End')

    def select_cat(self, cat):
        res = []
        for scene_name in tqdm(self.scene_lis):
            meta_p = os.path.join(self.data_dir, 'meta/%s_000.pkl' % scene_name)
            obj_p = pickle.load(open(meta_p, 'rb'))['obj_path']
            if cat in obj_p:
                res.append(scene_name)
        return res

    def reidx_cache(self):
        cache = pickle.load(open(self.cache_dir, 'rb'))
        new_dict = {}
        for i, index in enumerate(cache['index']):
            new_dict[index] = {
                'hA': cache['hA'][i],
                'hTo': cache['hTo'][i]
            }
        return new_dict

    def check_cache(self, scene, use_ret=False):
        if self.scene_idx_cache != scene or use_ret:
            self.scene_idx_cache = scene
            if isinstance(scene, str): 
                # debug
                scene_name = scene
            else:
                scene_name = self.scene_lis[scene]

            # read image, scene_name for single obj
            images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/%s*.jpg' % scene_name)))
            n_images = len(images_lis)
            images_np = np.stack([cv.imread(im_name) for im_name in images_lis]) / 255.0
            # obj mask, segm == 100
            masks_lis = sorted(glob(os.path.join(self.data_dir, 'segm/%s*.png' % scene_name)))
            if self.womask:
                masks_np = np.ones_like(images_np)
            else:
                masks_np = np.stack([cv.imread(im_name) for im_name in masks_lis])
                masks_np = (masks_np[..., 2: 3] == 100).repeat(3, axis=-1)
            # meta info
            metas_lis = sorted(glob(os.path.join(self.data_dir, 'meta/%s*.pkl' % scene_name)))
            pose_all = [self.transf_pose @ pickle.load(open(meta_p, 'rb'))['affine_transform'] \
                    for meta_p in metas_lis]
            pose_origin_woinv = [torch.tensor(pose).float() for pose in pose_all]  # oTc
            pose_all = [torch.inverse(torch.tensor(pose).float()) for pose in pose_all]  # cTo
        
            images = torch.from_numpy(images_np.astype(np.float32)).to(self.device)  # [inner_iter, H, W, 3]
            masks  = torch.from_numpy(masks_np.astype(np.float32)).to(self.device)   # [inner_iter, H, W, 3]
            focal = self.intrinsics_all[0][0, 0]
            pose_origin_woinv = torch.stack(pose_origin_woinv).to(self.device)    # [inner_iter, 4, 4]
            pose_all = torch.stack(pose_all).to(self.device)                      # [inner_iter, 4, 4]
            
            H, W = images.shape[1], images.shape[2]
            image_pixels = H * W
            
            # read ref image
            ref_image = cv.imread(os.path.join(self.ref_dir, 'rgb/%s_000.jpg' % scene_name)) / 255.0
            ref_image = torch.from_numpy(ref_image.astype(np.float32)).to(self.device)
            # dino
            dino_image = cv.imread(os.path.join(self.ref_dir, 'pca3_map/%s_000.png' % scene_name)) / 255.0
            dino_image = torch.from_numpy(dino_image.astype(np.float32)).to(self.device)

            # read hand pose and hto from cache
            hA = self.cache[scene_name]['hA']
            hTo = self.cache[scene_name]['hTo']

            if self.ref_mask:
                ref_image = ref_image * masks[0]
                dino_image = dino_image * masks[0]
                
            self.images_lis = images_lis
            self.n_images = n_images
            self.images_np = images_np
            self.masks_lis = masks_lis
            self.masks_np = masks_np
            self.metas_lis = metas_lis
            self.pose_all = pose_all
            self.pose_origin_woinv = pose_origin_woinv
            self.images = images
            self.masks = masks
            self.focal = focal
            self.H, self.W = H, W
            self.image_pixels = image_pixels
            self.ref_image = ref_image
            self.dino_image = dino_image
            self.ref_hA = torch.from_numpy(hA)
            self.ref_hTo = hTo
                
            if use_ret:
                return (images, masks, pose_all, H, W,      # for ray generation
                        ref_image, dino_image, pose_origin_woinv, torch.from_numpy(hA), hTo)       # for ref

    def gen_rays_mask_at(self, scene, view_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        self.check_cache(scene)
        l = resolution_level
        masks, H, W = self.masks, self.H, self.W
        if self.use_bbox:
            mask_img = masks[view_idx][..., 0].to(self.device)           # H, W
            x_min, x_max, y_min, y_max = get_mask_bbox(mask_img)    # inversed x & y
        else:
            x_min, x_max, y_min, y_max = (0, W, 0, H)
        tx = torch.linspace(x_min, x_max- 1, (x_max-x_min) // l)
        ty = torch.linspace(y_min, y_max - 1, (y_max-y_min) // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # pixel coord
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # camera coord (camera as origin point)
        p = torch.matmul(self.intrinsics_all_inv[view_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # direction vector in camera coord 
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = rays_v / torch.linalg.norm(self.pose_all[view_idx, :3, :3], dim=(-1, -2))
        # roate to the world coord (?) 
        rays_v = torch.matmul(self.pose_all[view_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        # world coord origin point's location in camera coord
        rays_o = self.pose_all[view_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        # inverse pose (both R and t, 4x4) is right, map to world coord
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)
    
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
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # camera coord (camera as origin point)
        p = torch.matmul(self.intrinsics_all_inv[view_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # direction vector in camera coord 
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = rays_v / torch.linalg.norm(self.pose_all[view_idx, :3, :3], dim=(-1, -2))
        # roate to the world coord (?) 
        rays_v = torch.matmul(self.pose_all[view_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        # world coord origin point's location in camera coord
        rays_o = self.pose_all[view_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        # inverse pose (both R and t, 4x4) is right, map to world coord
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, scene, view_idx, batch_size, scene_data=None):
        """
        Generate random rays at world space from one camera.
        """
        if scene_data is not None:
            images, masks, pose_all, H, W = scene_data
        else:
            self.check_cache(scene)
            images, masks, pose_all, H, W = self.images, self.masks, self.pose_all, self.H, self.W
        if self.use_bbox:
            mask_img = masks[view_idx][..., 0].to(self.device)           # H, W
            x_min, x_max, y_min, y_max = get_mask_bbox(mask_img)    # inversed x & y
        else:
            x_min, x_max, y_min, y_max = (0, W, 0, H)
        pixels_x = torch.randint(low=x_min, high=x_max, size=[batch_size]).to(self.device)
        pixels_y = torch.randint(low=y_min, high=y_max, size=[batch_size]).to(self.device)
        view_idx = view_idx.to(self.device)
        color = images[view_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = masks[view_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[view_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = rays_v / torch.linalg.norm(pose_all[view_idx, :3, :3], dim=(-1, -2))
        rays_v = torch.matmul(pose_all[view_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose_all[view_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.to(self.device), rays_v.to(self.device), color, mask[:, 2:3]], dim=-1).to(self.device)   # batch_size, 10

    def gen_source_img_pose_intr(self, scene, test=False):
        '''
        return pose without inverse
        '''
        self.check_cache(scene)
        img = self.ref_image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        pose = self.pose_origin_woinv[0]
        dino_image = self.dino_image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        torch_resize = Resize([64, 64])
        dino_image = torch_resize(dino_image)
        return img.to(self.device), dino_image.to(self.device), pose.to(self.device), self.intrinsics_all[0].to(self.device), self.ref_hA.unsqueeze(0), self.ref_hTo.unsqueeze(0)

    def __getitem__(self, scene):
        scene_data = self.check_cache(scene, use_ret=True)
        view_perm = torch.randperm(self.total_inner_iter)
        rays_o_list, rays_d_list, true_rgb_list, mask_list = [], [], [], []
        near_list, far_list = [], []
        for view_idx in range(self.train_inner_iter):
            data = self.gen_random_rays_at(scene, view_perm[view_idx % len(view_perm)], self.ray_batch_size,
                                           scene_data=scene_data[: 5])
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.near_far_from_sphere(rays_o, rays_d)
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)
            true_rgb_list.append(true_rgb)
            mask_list.append(mask)
            near_list.append(near)
            far_list.append(far)
        image, dino_img, pose_all, hA, hTo = scene_data[5: ]
        image = image.permute(2, 0, 1).contiguous()
        dino_img = dino_img.permute(2, 0, 1).contiguous()
        torch_resize = Resize([64, 64])
        dino_img = torch_resize(dino_img)
        pose = pose_all[0]
        hA = hA
        hTo = hTo
        intr = self.intrinsics_all[0]                # v*b is equal to 'batch_size' or 'n_rays' outside
        return (torch.cat(rays_o_list, 0), torch.cat(rays_d_list, 0),   # [v*b, 3], [v*b, 3]    
                torch.cat(true_rgb_list, 0), torch.cat(mask_list, 0), # [v*b, 3], [v*b, 1]
                torch.cat(near_list, 0), torch.cat(far_list, 0),     # [v*b, 1], [v*b, 1] [v, 4, 4]
                image, dino_img, pose, intr, hA, hTo)                                      # [3, H, W], [4, 4], [3, 3]

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
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, scene, view_idx, resolution_level):
        self.check_cache(scene)
        img = cv.imread(self.images_lis[view_idx])
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