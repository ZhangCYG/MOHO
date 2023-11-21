import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from shutil import copyfile
from icecream import ic
from tqdm import tqdm, trange
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.obman_dataset_ddp_dino import MultiObjectObmanDataset, get_iter_sampler
from models.ho3d_dataset import HO3DDataset, get_iter_sampler
from models.dexycb_dataset import DexYCBDataset, get_iter_sampler
from models.fields_ho import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer_ho_ddp_hand import NeuSRenderer
from models.enc import ImageSpEnc
from models.hand_utils import ManopthWrapper
import models.geom_utils as geom_utils
import time


def network_dp(net, gpu_num):
    net = torch.nn.DataParallel(net, device_ids=list(range(gpu_num)))
    return net


def knns_dist(xyz1, xyz2, k, device):
    samples = xyz1.shape[1]
    xyz1_xyz1 = torch.bmm(xyz1, xyz1.transpose(2, 1))
    xyz2_xyz2 = torch.bmm(xyz2, xyz2.transpose(2, 1))
    xyz1_xyz2 = torch.bmm(xyz1, xyz2.transpose(2, 1))
    diag_ind_x = torch.arange(0, samples).to(device)
    rx = xyz1_xyz1[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xyz1_xyz2.transpose(2,1))
    ry = xyz2_xyz2[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xyz1_xyz2)
    pair_wise_loss = rx.transpose(2, 1) + ry - 2 * xyz1_xyz2

    top_min_k = torch.topk(pair_wise_loss, k, dim=2, largest=False)

    return top_min_k


def knn_loss(rays_o, normal, k_normal):
    xyz = torch.cat([rays_o, normal], dim=-1)
    device = normal.device
    k = k_normal
    k = k + 1  # a point also includes itself in knn search
    batch = xyz.shape[0]
    num_points = xyz.shape[1]
    channels = xyz.shape[2]

    get_knn = knns_dist(xyz[:, :, 0:3], xyz[:, :, 0:3], k, device)

    k_indices = get_knn.indices

    kp = torch.gather(xyz.unsqueeze(1).expand(-1, xyz.size(1), -1, -1), 2, k_indices.unsqueeze(-1).expand(-1, -1, -1, xyz.size(-1)))

    n_dist = kp[:, :, 0, :].view(batch, num_points, 1, channels)[:, :, :, 3:6] - kp[:, :, 0:k_normal+1, 3:6]
    n_dist = n_dist[:, :, 1:, :]
    normal_neighbor_loss = torch.mean(torch.sum(n_dist ** 2, dim=-1))

    return normal_neighbor_loss


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, device='cuda', gpu_num=2):
        self.device = torch.device(device)

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
            
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        
        self.scene_batch_size = self.conf.get_int('train.scene_batch_size', default=1)
        
        self.total_inner_iter = self.conf.get_int('dataset.inner_iter')
        self.train_inner_iter = self.conf.get_int('train.train_inner_iter', default=self.total_inner_iter)
        self.test_inner_iter = self.conf.get_int('dataset.test_inner_iter', default=self.total_inner_iter)
        assert self.train_inner_iter <= self.total_inner_iter

        self.max_iter_use_bbox = self.conf.get_int('train.max_iter_use_bbox', default=0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        self.normal_smooth_weight = self.conf.get_float('train.normal_smooth_weight')
        self.normal_smooth_thre = self.conf.get_int('train.normal_smooth_thre')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = network_dp(NeRF(**self.conf['model.nerf']).to(self.device), gpu_num)
        self.sdf_network = network_dp(SDFNetwork(**self.conf['model.sdf_network']).to(self.device), gpu_num)
        self.deviation_network = network_dp(SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device), gpu_num)
        self.color_network = network_dp(RenderingNetwork(**self.conf['model.rendering_network']).to(self.device), gpu_num)
        self.enc = network_dp(ImageSpEnc(**self.conf['model.enc']).to(self.device), gpu_num)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.enc.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
        
        self.hand_wrapper = ManopthWrapper()

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # dataset & dataloader
        data_type = self.conf['dataset'].get('name', None)
        if data_type == 'multi_object_obman':
            self.dataset = MultiObjectObmanDataset(self.conf['dataset'], 'cpu', train_inner_iter=self.train_inner_iter, ray_batch_size=self.batch_size)
            # iter_step update in load_checkpoint
            train_sampler = get_iter_sampler(self.dataset, batch_size=self.scene_batch_size,
                                             drop_last=True, max_iteration=self.end_iter,
                                             start_iteration=self.iter_step)
            print('Initial data sampler from iter: %d.' % self.iter_step)
            self.dataloader = DataLoader(self.dataset,
                                         batch_sampler=train_sampler,
                                         num_workers=32)
            self.data_iter = enumerate(self.dataloader)
            print('Loader initialization: finished')
        elif data_type == 'ho3d':
            self.dataset = HO3DDataset(self.conf['dataset'], 'cpu', train_inner_iter=self.train_inner_iter, ray_batch_size=self.batch_size)
            # iter_step update in load_checkpoint
            train_sampler = get_iter_sampler(self.dataset, batch_size=self.scene_batch_size,
                                             drop_last=True, max_iteration=self.end_iter,
                                             start_iteration=self.iter_step)
            print('Initial data sampler from iter: %d.' % self.iter_step)
            self.dataloader = DataLoader(self.dataset,
                                         batch_sampler=train_sampler,
                                         num_workers=32)
            self.data_iter = enumerate(self.dataloader)
            print('Loader initialization: finished')
        elif data_type == 'dexycb':
            self.dataset = DexYCBDataset(self.conf['dataset'], 'cpu', train_inner_iter=self.train_inner_iter, ray_batch_size=self.batch_size)
            # iter_step update in load_checkpoint
            train_sampler = get_iter_sampler(self.dataset, batch_size=self.scene_batch_size,
                                             drop_last=True, max_iteration=self.end_iter,
                                             start_iteration=self.iter_step)
            print('Initial data sampler from iter: %d.' % self.iter_step)
            self.dataloader = DataLoader(self.dataset,
                                         batch_sampler=train_sampler,
                                         num_workers=32)
            self.data_iter = enumerate(self.dataloader)
            print('Loader initialization: finished')
        else:
            raise ValueError('To be continue...')

        self.data_type = data_type
        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
            
        print('Runner initialization: finished')
    
    def get_jsTx(self, hA, hTx):
        hTjs = self.hand_wrapper.pose_to_transform(hA, False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = hTx.unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp  
        return jsTx

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        val_scene_perm = self.get_scene_perm()

        for iter_o in trange(0, res_step):
            # whether use bbox
            if self.iter_step < self.max_iter_use_bbox:
                self.dataset.use_bbox = True
            else:
                self.dataset.use_bbox = False
            
            _, iter_data = self.data_iter.__next__()
            iter_data = list(iter_data)
            for idx, itm_id in enumerate(iter_data):
                iter_data[idx] = itm_id.to(self.device)
            rays_o, rays_d, true_rgb, mask, seg, near, far, image, dino_img, pose, intr, hA, hTo = iter_data
           
            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 1, 3])  # n_scenes, rays, rgb
                
            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
                
            latent = self.enc(image)
            glb, local = latent
            local = torch.cat([local, dino_img], dim=1)
            glb = torch.cat([glb, dino_img.mean(dim=-1).mean(dim=-1)], dim=1)
            latent = (glb, local)
            jsTo = self.get_jsTx(hA.cpu(), hTo.cpu())
            jsTo = jsTo.to(self.device)
            render_out = self.renderer.render(rays_o, rays_d, near, far, 
                                        pose=pose, intr=intr, ref_feature=latent, jsTo=jsTo,
                                        background_rgb=background_rgb,
                                        cos_anneal_ratio=self.get_cos_anneal_ratio())
            
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            mask_sum = mask.sum() + 1e-5
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), seg)

            n_samples = self.renderer.n_samples + self.renderer.n_importance
            normals = render_out['gradients'] * render_out['weights'][:, :, :n_samples, None]
            normals = normals * render_out['inside_sphere'][..., None]
            normals = normals.sum(dim=2)
            n_dot_v = -(normals * rays_d).sum(dim=-1)
            normal_loss = (torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum()

            normal_smooth_loss = knn_loss(rays_o, normals, self.normal_smooth_thre)

            loss = color_fine_loss +\
                eikonal_loss * self.igr_weight +\
                mask_loss * self.mask_weight +\
                normal_loss * self.normal_weight + normal_smooth_loss * self.normal_smooth_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss * self.igr_weight, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss * self.mask_weight, self.iter_step)
            self.writer.add_scalar('Loss/normal_loss', normal_loss * self.normal_weight, self.iter_step)
            self.writer.add_scalar('Loss/normal_smooth_loss', normal_smooth_loss * self.normal_smooth_weight, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[..., :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image(val_scene_perm[self.iter_step % len(val_scene_perm)])

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(val_scene_perm[self.iter_step % len(val_scene_perm)])

            self.update_learning_rate()

    def test_mesh(self):
        scene_perm = self.get_scene_perm()
        pbar = trange(0, self.dataset.n_scenes)
        for iter_o in pbar:
            scene = scene_perm[iter_o]
            pbar.set_description(self.dataset.scene_lis[scene])
            self.validate_mesh(scene, resolution=128, threshold=0.0, test=True, ref=0)
            print(self.dataset.n_images)
            for ref in range(1, self.dataset.n_images):
                self.validate_mesh(scene, resolution=128, threshold=0.0, test=True, ref=ref)

    def get_scene_perm(self):
        return torch.arange(0, self.dataset.n_scenes, 1)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.enc.load_state_dict(checkpoint['enc'])
        # self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'enc': self.enc.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, scene, view_idx=-1, resolution_level=-1, test=False):
        '''
        scene:  int, idx of scene
        view_idx:    int, idx of view in this scene
        '''
        if view_idx < 0:
            view_idx = np.random.randint(self.total_inner_iter)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, view_idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(scene, view_idx, resolution_level=resolution_level)
        rays_o, rays_d = rays_o.to(self.device), rays_d.to(self.device)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        image, dino_img, pose, intr, hA, hTo, _ = self.dataset.gen_source_img_pose_intr(scene)
        image, dino_img, pose, intr = image.to(self.device), dino_img.to(self.device), pose.to(self.device), intr.to(self.device)
        latent = self.enc(image)
        glb, local = latent
        local = torch.cat([local, dino_img], dim=1)
        glb = torch.cat([glb, dino_img.mean(dim=-1).mean(dim=-1)], dim=1)
        latent = (glb, local)
        jsTo = self.get_jsTx(hA.cpu(), hTo.cpu())
        jsTo = jsTo.to(self.device)

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = near.to(self.device), far.to(self.device)
            background_rgb = torch.ones([1, 1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch.unsqueeze(0),
                                              rays_d_batch.unsqueeze(0),
                                              near.unsqueeze(0),
                                              far.unsqueeze(0),
                                              pose=pose.unsqueeze(0),
                                              intr=intr.unsqueeze(0),
                                              ref_feature=latent,
                                              jsTo=jsTo,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'][0].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'][0] * render_out['weights'][0][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][0][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = pose[:3, :3].detach().cpu().numpy()
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        if not test:
            os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>10d}_{}_{}_{}.png'.format(self.iter_step, i, scene, view_idx)),
                            np.concatenate([img_fine[..., i],
                                            self.dataset.image_at(scene, view_idx, resolution_level=resolution_level)]))
                if len(out_normal_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>10d}_{}_{}_{}.png'.format(self.iter_step, i, scene, view_idx)),
                            normal_img[..., i])
        else:
            os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine_test'), exist_ok=True)
            os.makedirs(os.path.join(self.base_exp_dir, 'normals_test'), exist_ok=True)
            scene_name = self.dataset.scene_lis[scene]
            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine_test',
                                            '{:0>10d}_{}_{}_{}.png'.format(self.iter_step, i, scene_name, view_idx)),
                            np.concatenate([img_fine[..., i],
                                            self.dataset.image_at(scene, view_idx, resolution_level=resolution_level)]))
                if len(out_normal_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'normals_test',
                                            '{:0>10d}_{}_{}_{}.png'.format(self.iter_step, i, scene_name, view_idx)),
                            normal_img[..., i])

    def render_novel_image(self, scene, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(scene, idx_0, idx_1, ratio, resolution_level=resolution_level)
        rays_o, rays_d = rays_o.to(self.device), rays_d.to(self.device)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        image, pose, intr = self.dataset.gen_source_img_pose_intr(scene)
        image, pose, intr = image.to(self.device), pose.to(self.device), intr.to(self.device)
        latent = self.enc(image)
        
        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = near.to(self.device), far.to(self.device)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              pose=pose,
                                              intr=intr,
                                              ref_feature=latent,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, scene, world_space=False, resolution=64, threshold=0.0, test=False, ref=None):
        if scene is None:
            scene = np.random.randint(self.dataset.n_scenes)
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32).to(self.device)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32).to(self.device)
        image, dino_img, pose, intr, hA, hTo, image_p = self.dataset.gen_source_img_pose_intr(scene, test=test, ref=ref)
        image, dino_img, pose, intr = image.to(self.device), dino_img.to(self.device), pose.to(self.device), intr.to(self.device)
        
        latent = self.enc(image)
        glb, local = latent
        local = torch.cat([local, dino_img], dim=1)
        glb = torch.cat([glb, dino_img.mean(dim=-1).mean(dim=-1)], dim=1)
        latent = (glb, local)
        jsTo = self.get_jsTx(hA.cpu(), hTo.cpu())
        jsTo = jsTo.to(self.device)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, ref_feature=latent, jsTo=jsTo, 
                                           pose=pose, intr=intr,
                                           resolution=resolution, threshold=threshold)

        if not test:
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes_test'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if not test:
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>10d}.ply'.format(self.iter_step)))
        else:
            if self.data_type == 'ho3d':
                print(image_p)
                scene_n = image_p.split('/')[-3]
                image_n = image_p.split('/')[-1]
                image_n = image_n[:4]
                mesh.export(os.path.join(self.base_exp_dir, 'meshes_test', '{}.ply'.format(scene_n + '_' + image_n)))
            elif self.data_type == 'dexycb':
                print(image_p)
                scene_n = image_p.split('/')[-5] + '_' + image_p.split('/')[-4] + '_' + image_p.split('/')[-3]
                image_n = image_p.split('/')[-2]
                image_n = image_n[:-4] + '_' + image_p.split('/')[-1]
                mesh.export(os.path.join(self.base_exp_dir, 'meshes_test', '{}.ply'.format(scene_n + '_' + image_n)))
            else:
                scene_name = self.dataset.scene_lis[scene]
                mesh.export(os.path.join(self.base_exp_dir, 'meshes_test', '{}.ply'.format(scene_name)))
        logging.info('End')

    # for psnr evalation
    def test_image(self, resolution_level=-1):
        print('Generate test image: iter: {}'.format(self.iter_step))
        test = True 
        
        if self.data_type == 'ho3d':
            ref_file_path = 'cache/dexycb_view_test.txt'
        elif self.data_type == 'dexycb':
            ref_file_path = 'cache/dexycb_view_test.txt'
        with open(ref_file_path, 'r') as f:
            iter_list = [item for item in f.read().split('\n') if item != '']
            ref_list = [[*item.split('\t')] for item in iter_list]
        
        os.makedirs(os.path.join(self.base_exp_dir, 'views_fine_test'), exist_ok=True)
        
        for scene, ref_idx, select_views, img_path in tqdm(ref_list):
            ref_idx = int(ref_idx)
            select_views = [int(sv) for sv in select_views.split(',')]
            
            image, dino_img, pose, intr, hA, hTo, _ = \
                    self.dataset.gen_source_img_pose_intr(scene, test, ref_idx)
            image, dino_img, pose, intr = image.to(self.device), dino_img.to(self.device), \
                                          pose.to(self.device), intr.to(self.device)
            latent = self.enc(image)
            glb, local = latent
            local = torch.cat([local, dino_img], dim=1)
            glb = torch.cat([glb, dino_img.mean(dim=-1).mean(dim=-1)], dim=1)
            latent = (glb, local)
            jsTo = self.get_jsTx(hA.cpu(), hTo.cpu())
            jsTo = jsTo.to(self.device)
            
            ray_batch_size = self.scene_batch_size * self.batch_size * 2
            
            for view_idx in tqdm(select_views, leave=False):
                if view_idx == ref_idx:
                    continue
                
                view_src_path = self.dataset.images_lis[view_idx]
                save_view_path = os.path.join(self.base_exp_dir,
                                    'views_fine_test',
                                    '{}_{}_{}_{}_{}_{}.png'.format(scene, ref_idx, *view_src_path.split('/')[-4: ]))
                if os.path.isfile(save_view_path):
                    continue
                
                if resolution_level < 0:
                    resolution_level = self.validate_resolution_level
                rays_o, rays_d = self.dataset.gen_rays_at(scene, view_idx, resolution_level=resolution_level)
                rays_o, rays_d = rays_o.to(self.device), rays_d.to(self.device)
                H, W, _ = rays_o.shape
                rays_o = rays_o.reshape(-1, 3).split(ray_batch_size)
                rays_d = rays_d.reshape(-1, 3).split(ray_batch_size)

                out_rgb_fine = []

                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                    near, far = near.to(self.device), far.to(self.device)
                    background_rgb = torch.ones([1, 1, 3]) if self.use_white_bkgd else None

                    render_out = self.renderer.render(rays_o_batch.unsqueeze(0),
                                                    rays_d_batch.unsqueeze(0),
                                                    near.unsqueeze(0),
                                                    far.unsqueeze(0),
                                                    pose=pose.unsqueeze(0),
                                                    intr=intr.unsqueeze(0),
                                                    ref_feature=latent,
                                                    jsTo=jsTo,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                    background_rgb=background_rgb)

                    def feasible(key): return (key in render_out) and (render_out[key] is not None)

                    if feasible('color_fine'):
                        out_rgb_fine.append(render_out['color_fine'][0].detach().cpu().numpy())
                    del render_out

                img_fine = None
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
                cv.imwrite(save_view_path, img_fine[..., 0])


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='train', help='split of obman')
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.device, args.gpu_num)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'test':
        runner.test()
    elif args.mode == 'test_mesh':
        runner.test_mesh()
    elif args.mode == 'test_image':
        runner.test_image()