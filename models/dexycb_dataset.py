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
import json
from models import geom_utils
from torchvision.transforms import Resize


def get_mask_bbox(mask: torch.Tensor, expand=2):
    h_m = torch.where(mask.sum(dim=0) != 0)[0]
    w_m = torch.where(mask.sum(dim=1) != 0)[0]
    x_min = max(h_m.min().item() - expand       , 0)
    x_max = min(h_m.max().item() + expand + 1   , mask.shape[1])
    y_min = max(w_m.min().item() - expand       , 0)
    y_max = min(w_m.max().item() + expand + 1   , mask.shape[0])
    return x_min, x_max, y_min, y_max


def cvt_axisang_t_i2o(axisang, trans):
    """+correction: t_r - R_rt_r. inner to outer"""
    trans += get_offset(axisang)

    return axisang, trans


def get_offset(axisang):
    """
    :param axisang: (N, 3)
    :return: trans: (N, 3) = r_r - R_r t_r
    """
    device = axisang.device
    N = axisang.size(0)
    t_mano = torch.tensor([[0.09566994, 0.00638343, 0.0061863]], dtype=torch.float32, device=device).repeat(N, 1)
    rot_r = geom_utils.axis_angle_t_to_matrix(axisang, homo=False)  # N, 3, 3
    delta = t_mano - torch.matmul(rot_r, t_mano.unsqueeze(-1)).squeeze(-1)
    return delta


_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]


_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}


class DexYCBDataset(Dataset):
    def __init__(self, conf, device='cuda', train_inner_iter=None, ray_batch_size=512):
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
        print('Load dexycb data: Begin')
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

        self.object_bbox_min = np.array([-0.3, -0.3, -0.3])
        self.object_bbox_max = np.array([0.3, 0.3, 0.3])

        # cache
        self.scene_idx_cache = None

        print('Load data: End')

    def reidx_cache(self):
        cache = json.load(open(self.cache_dir, 'r'))
        new_dict = {}
        for i, index in tqdm(enumerate(cache['images'])):
            file_name = cache['images'][i]['file_name']
            subject_id = int(file_name.split('_')[0]) - 1
            video_id = '_'.join(file_name.split('_')[1:3])
            scene = str(subject_id) + '_' + video_id
            cam_id = file_name.split('_')[-2]
            frame_id = file_name.split('_')[-1].rjust(6, '0')
            anno = cache['annotations'][i]
            ycb_id = anno['ycb_id']
            fx = anno['fx']
            fy = anno['fy']
            cx = anno['cx']
            cy = anno['cy']
            intr = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float32)
            hA = np.array(anno['hand_poses'], dtype=np.float32)
            hand_trans = np.array(anno['hand_trans'], dtype=np.float32)
            pose = np.array(anno['obj_transform'], dtype=np.float32)
            mask_p = os.path.join(self.data_dir, _SUBJECTS[subject_id], video_id, cam_id, 'labels_' + frame_id + '.npz')
            seg_img = np.load(mask_p)['seg']
            seg_maps = np.zeros((seg_img.shape[0], seg_img.shape[1], 1), dtype=np.float32)
            seg_maps[:, :, 0][np.where(seg_img == ycb_id)] = 1.
            if seg_maps.sum() > 50:
                if scene not in new_dict.keys():
                    new_dict[scene] = {'subject_id': [subject_id], 'video_id': [video_id], 'cam_id': [cam_id], 'frame_id': [frame_id], 'ycb_id': [ycb_id], 'intr': [intr], 'hA': [hA], 'hand_trans': [hand_trans], 'pose': [pose]}
                else:
                    new_dict[scene]['subject_id'].append(subject_id)
                    new_dict[scene]['video_id'].append(video_id)
                    new_dict[scene]['cam_id'].append(cam_id)
                    new_dict[scene]['frame_id'].append(frame_id)
                    new_dict[scene]['ycb_id'].append(ycb_id)
                    new_dict[scene]['intr'].append(intr)
                    new_dict[scene]['hA'].append(hA)
                    new_dict[scene]['hand_trans'].append(hand_trans)
                    new_dict[scene]['pose'].append(pose)
        for scene in new_dict.keys():
            print(scene, ': ', len(new_dict[scene]['subject_id']))  
        return new_dict

    def check_cache(self, scene, use_ret=False, test=False):
        '''
        self.pose_all: inversed pose matrix
        self.pose_origin_woinv: real pose matrix
        '''
        if self.scene_idx_cache != scene:
            self.scene_idx_cache = scene
            if isinstance(scene, str):
                # debug
                scene_name = scene
            else:
                scene_name = self.scene_lis[scene]
            subject_id = self.cache[scene_name]['subject_id']
            video_id = self.cache[scene_name]['video_id']
            cam_id = self.cache[scene_name]['cam_id']
            frame_id = self.cache[scene_name]['frame_id']
            # read image, scene_name for single obj
            images_lis = [os.path.join(self.data_dir, _SUBJECTS[subject_id[i]], video_id[i], cam_id[i], 'color_' + frame_id[i] + '.jpg') for i in range(len(subject_id))]
            if not test:
                sample_list = [i for i in range(len(images_lis))]
                sample_list = random.sample(sample_list, self.total_inner_iter)
                images_lis = [images_lis[i] for i in sample_list]
            n_images = len(images_lis)
            images_np = np.stack([cv.imread(im_name) for im_name in images_lis]) / 255.0
            images = torch.from_numpy(images_np.astype(np.float32))  # [inner_iter, H, W, 3]
            H, W = images.shape[1], images.shape[2]
            # obj mask
            masks_lis = [os.path.join(self.data_dir, _SUBJECTS[subject_id[i]], video_id[i], cam_id[i], 'labels_' + frame_id[i] + '.npz') for i in range(len(subject_id))]
            if not test:
                masks_lis = [masks_lis[i] for i in sample_list]
            ycb_ids = self.cache[scene_name]['ycb_id']
            if not test:
                ycb_ids = [ycb_ids[i] for i in sample_list]
            if self.womask:
                masks_np = np.ones_like(images_np)
            else:
                mask_np_list = []
                for i in range(len(masks_lis)):
                    seg_img = np.load(masks_lis[i])['seg']
                    seg_maps = np.zeros((seg_img.shape[0], seg_img.shape[1], 1), dtype=np.float32)
                    seg_maps[:, :, 0][np.where(seg_img == ycb_ids[i])] = 1.
                    mask_np_list.append(seg_maps)
                masks_np = np.stack(mask_np_list)
            # meta info
            pose_list = self.cache[scene_name]['pose']
            pose_all = [torch.from_numpy(pose_list[i]) for i in range(len(pose_list))]
            if not test:
                pose_all = [pose_all[i] for i in sample_list]
            pose_all = [torch.inverse(pose) for pose in pose_all]
            pose_origin_woinv = [pose.float() for pose in pose_all]

            masks = torch.from_numpy(masks_np.astype(np.float32))   # [inner_iter, H, W, 1]

            intr_list = self.cache[scene_name]['intr']
            intr_list = [torch.from_numpy(intr).float() for intr in intr_list]
            if not test:
                intr_list = [intr_list[i] for i in sample_list]
            intr_all = torch.stack(intr_list)
            intr_all_inv = torch.stack([torch.inverse(intr.float()) for intr in intr_list])
            pose_origin_woinv = torch.stack(pose_origin_woinv)    # [inner_iter, 4, 4]
            pose_all = torch.stack(pose_all)                      # [inner_iter, 4, 4]  cTo

            # read hand pose and hto from cache
            hA_list = self.cache[scene_name]['hA']
            hA_list = [torch.from_numpy(ha).float() for ha in hA_list]
            if not test:
                hA_all_list = [hA_list[i] for i in sample_list]
                hA_pose_list = [hA_list[i][3:] for i in sample_list]
            else:
                hA_all_list = [hA_list[i] for i in range(len(hA_list))]
                hA_pose_list = [hA_list[i][3:] for i in range(len(hA_list))]
            hA_all = torch.stack(hA_pose_list)  # [inner, 45]

            hand_trans_list = self.cache[scene_name]['hand_trans']
            hand_trans_list = [torch.from_numpy(trans).float() for trans in hand_trans_list]
            if not test:
                hand_trans_list = [hand_trans_list[i] for i in sample_list]
            hTo_list = []
            for i in range(len(hA_all_list)):
                ha_pose = hA_all_list[i]
                rot = ha_pose[:3]
                trans = hand_trans_list[i]
                rot, trans = cvt_axisang_t_i2o(rot.unsqueeze(0), trans.unsqueeze(0))
                wTh = geom_utils.axis_angle_t_to_matrix(rot, trans)
                rot = torch.FloatTensor([[[1, 0, 0], [0, -1, 0], [0, 0, -1]]])
                cTw = geom_utils.rt_to_homo(rot, )
                cTh = cTw @ wTh
                cTo = pose_origin_woinv[i]
                hTo = torch.inverse(cTh) @ cTo
                hTo_list.append(hTo)
            hTo_all = torch.cat(hTo_list)

            images = images * masks

            # dino
            dino_lis = [os.path.join(self.data_dir, _SUBJECTS[subject_id[i]], video_id[i], cam_id[i], 'pca3_map_' + frame_id[i] + '.png') for i in range(len(subject_id))]
            if not test:
                dino_lis = [dino_lis[i] for i in sample_list]
            dino_np = np.stack([cv.resize(cv.imread(dino_name), (W, H)) for dino_name in dino_lis]) / 255.0
            dinos = torch.from_numpy(dino_np.astype(np.float32))
            dino_image = dinos * masks.expand(*dinos.shape)

            # seg
            if self.seg_dir is not None:
                seg_lis = [os.path.join(self.seg_dir, _SUBJECTS[subject_id[i]], video_id[i], cam_id[i], 'labels_' + frame_id[i] + '.png') for i in range(len(subject_id))]
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
            self.ycb_ids = ycb_ids
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
        image_p = self.images_lis[ref_idx] + '/' + str(self.ycb_ids[ref_idx])
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
        near = 0.2 * torch.ones_like(mid)
        far = 2.0 * torch.ones_like(mid)
        return near, far

    def image_at(self, scene, view_idx, resolution_level):
        self.check_cache(scene)
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