import os
import argparse
from tqdm import tqdm
import lpips
import cv2
import numpy as np
import torch
import json
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


dexycb_dataset_path = 'PATH TO DEXYCB'
ho3d_dataset_path = 'PATH TO HO3D'

dexycb_cache_dir = 'PATH TO cache/dexycb_test_s0.json'

parser = argparse.ArgumentParser(description="Calculate PSNR for rendered images.")
parser.add_argument(
    "--type",
    "-T",
    type=str,
    default="ho3d",
    choices=['dexycb', 'ho3d'],
    help="Dataset type",
)
parser.add_argument(
    "--pred_dir",
    "-P",
    type=str,
    default="",
    help="Generated image directory",
)
parser.add_argument(
    "--wo_ABF10", action='store_true', default=False, help="For ho3d, wo ABF10 scene",
)
parser.add_argument(
    "--lpips_batch_size", type=int, default=4, help="Batch size for LPIPS",
)
args = parser.parse_args()
    
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

def dexycb_reidx_cache(data_dir, cache_dir):
    cache = json.load(open(cache_dir, 'r'))
    new_dict = {}
    for i, index in tqdm(enumerate(cache['images']), total=len(cache['images'])):
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
        mask_p = os.path.join(data_dir, _SUBJECTS[subject_id], video_id, cam_id, 'labels_' + frame_id + '.npz')
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
    

if __name__ == '__main__':
    dataset_type = args.type
    if dataset_type == 'ho3d':
        dataset_root = ho3d_dataset_path
    elif dataset_type == 'dexycb':
        dataset_root = dexycb_dataset_path
        dexycb_cache = dexycb_reidx_cache(dataset_root, dexycb_cache_dir)
    
    pred_root = args.pred_dir
    pred_image_name_list = sorted([itm for itm in os.listdir(pred_root) if '.png' in itm])
    
    pred_list = []
    gt_list = []
    
    print('Calculate PSNR and SSIM...')
    psnr_avg = 0.0
    ssim_avg = 0.0
    for pred_image_name in tqdm(pred_image_name_list):
        # load pred
        pred_image_path = os.path.join(pred_root, pred_image_name)
        if args.wo_ABF10 and 'ABF10' in pred_image_path:
            continue
        pred_image = cv2.imread(pred_image_path)[:, :, ::-1]
        H, W = pred_image.shape[0], pred_image.shape[1]
        
        # load gt with mask
        if dataset_type == 'ho3d':
            gt_image_path = os.path.join(dataset_root, *pred_image_name[: -4].split('_')[-4: ])
            gt_image = cv2.imread(gt_image_path)[:, :, ::-1]
            gt_mask_path = gt_image_path.replace('/rgb/', '/seg/').replace('.png', '.jpg')
            gt_mask = cv2.resize(cv2.imread(gt_mask_path), (W, H), cv2.INTER_NEAREST)[:, :, 0] > 100
        elif dataset_type == 'dexycb':
            name_split = pred_image_name[: -4].split('_')
            gt_image_path = os.path.join(dataset_root,
                                         name_split[-6],
                                         name_split[-5] + '_' + name_split[-4], 
                                         name_split[-3],
                                         name_split[-2] + '_' + name_split[-1]
                                         )
            gt_image = cv2.imread(gt_image_path)[:, :, ::-1]
            gt_scene = '%s_%s_%s' % (name_split[0], name_split[1], name_split[2])
            gt_view_idx = int(name_split[3])
            gt_mask_path = gt_image_path.replace('color_', 'labels_').replace('.jpg', '.npz')
            seg_img = np.load(gt_mask_path)['seg']
            seg_maps = np.zeros((seg_img.shape[0], seg_img.shape[1], 1), dtype=np.float32)
            seg_maps[:, :, 0][np.where(seg_img == dexycb_cache[gt_scene]['ycb_id'][gt_view_idx])] = 1.
            gt_mask = seg_maps[..., 0]
        
        gt_image[gt_mask == 0] = 0
        pred_image[gt_mask == 0] = 0
        
        # to [0, 1]
        pred_image = pred_image.astype(np.float32) / 255.0
        gt_image = gt_image.astype(np.float32) / 255.0
        # calculate psnr and ssim
        psnr = compare_psnr(pred_image, gt_image, data_range=1)
        ssim = compare_ssim(pred_image, gt_image, multichannel=True, data_range=1, channel_axis=-1)
        psnr_avg += psnr
        ssim_avg += ssim
        
        # ready for lpips
        pred_list.append(torch.from_numpy(pred_image).permute(2, 0, 1) * 2.0 - 1.0)
        gt_list.append(torch.from_numpy(gt_image).permute(2, 0, 1) * 2.0 - 1.0)

    psnr_avg = psnr_avg / len(pred_image_name_list)
    ssim_avg = ssim_avg / len(pred_image_name_list)
    print('psnr_avg: %.4f, \tssim_avg: %.4f' % (psnr_avg, ssim_avg))
    
    print('Calculate LPIPS...')
    # load vgg model
    cuda = "cuda:" + str(0)     # cuda: 0
    lpips_vgg = lpips.LPIPS(net="vgg").to(device=cuda)
    # calculate lpips
    preds = torch.stack(pred_list)
    gts = torch.stack(gt_list)
    lpips_all = []
    preds_spl = torch.split(preds, args.lpips_batch_size, dim=0)
    gts_spl = torch.split(gts, args.lpips_batch_size, dim=0)
    with torch.no_grad():
        for predi, gti in tqdm(zip(preds_spl, gts_spl)):
            lpips_i = lpips_vgg(predi.to(device=cuda), gti.to(device=cuda))
            lpips_all.append(lpips_i)
        lpips = torch.cat(lpips_all)
    lpips = lpips.mean().item()
    
    print('lpips_avg: %.4f' % (lpips))
    
    # write result
    out_path = os.path.join(pred_root, '_test_image_results%s.txt' % ('_wo_ABF10' if args.wo_ABF10 else ''))
    out_txt = "psnr {}\nssim {}\nlpips {}".format(psnr_avg, ssim_avg, lpips)
    with open(out_path, "w") as f:
        f.write(out_txt)