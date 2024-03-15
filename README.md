# MOHO: Learning Single-view Hand-held Object Reconstruction with Multi-view Occlusion-Aware Supervision, CVPR 2024
Chenyangguang Zhang, Guanlong Jiao, Yan Di, Gu Wang, Ziqin Huang, Ruida Zhang, Fabian Manhardt, Bowen Fu, Federico Tombari, Xiangyang Ji


## Installation
```
conda activate moho
pip install -r requirements.txt
```

## Data Preparation

To keep the training and testing split with IHOI (https://github.com/JudyYe/ihoi), we use their `cache` file (https://drive.google.com/drive/folders/1v6Pw6vrOGIg6HUEHMVhAQsn-JLBWSHWu?usp=sharing). Unzip it and put under `cache/` folder.
The split we use in DexYCB follows the realsed code of (https://github.com/zerchen/gSDF), which needs to be downloaded from (https://drive.google.com/drive/folders/1qULhMx1PrnXkihrPacIFzLOT5H2FZSj7) and put in the `cache/` folder as `cache/dexycb_test_s0.json` and `cache/dexycb_train_s0.json`.
Moreover, the `xxx_view_test.txt` in the `cache/` folder is for evaluation of novel view synthesis.

`SOMVideo` is downloaded from (https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/zcyg22_mails_tsinghua_edu_cn/Etb0op97f0lOjYLu58ZM7_wBSfu2v0GRo6OKqAaMwzeztg?e=KbVTR4) for `SOMVideo_ref.zip` and (https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/jgl22_mails_tsinghua_edu_cn/EiXuWQMSmbBArOnBH_vPssoBfJpQfz3Nhcq-HZiTnSBOfw?e=Rp6iUz) for `SOMVideo_sup.tar.gz`.
`HO3D` is downloaded from (https://www.tugraz.at/index.php?id=40231) (we use HO3D(v2)).
`DexYCB` is downloaded from (https://dex-ycb.github.io/).

`externals/mano` contains `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`, get them from (https://mano.is.tue.mpg.de/).

We use PCA maps generated from DINO for generic semantic cues for MOHO, these data is also released on our link (https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/zcyg22_mails_tsinghua_edu_cn/Etb0op97f0lOjYLu58ZM7_wBSfu2v0GRo6OKqAaMwzeztg?e=KbVTR4) (`dino_pca.tar.gz`). Users should unzip this file and put it into the corresponding folder of `HO3D` and `DexYCB`.
2D hand coverage maps are released also on the link above (`dexycb_seg.zip` and `ho3d_seg.zip`) for the amodal-mask-weighted supervision when real-world finetuning.

## Configuration
In all config files in `confs/` folder, please make sure the correct `data_dir`, `ref_dir`, `cache_dir` and `seg_dir`.
`data_dir` means the path of supervision images. It is the same as `ref_dir` (the path of reference images) in the real-world finetuning on `HO3D` and `DexYCB`, but remains different on `SOMVideo`.
`seg_dir` is only used in the real-world finetuning for the amodal-mask-weighted supervision.

## Synthetic Pre-training on SOMVideo
```
CUDA_VISIBLE_DEVICES=0,1,2 python exp_runner_ho_dp_hand.py --mode train --conf confs/moo_wmask_dp_hand.conf --case pre_training --gpu_num 3
```

## Real-world Finetuning on HO3D and DexYCB
First, copy the pre-trained checkpoint to the experiment directory of real-world finetuning and rename it as 'ckpt_000000.pth'.
```
CUDA_VISIBLE_DEVICES=0,1,2 python exp_runner_ho_dp_hand.py --mode train --conf confs/ho3d_dino_hand.conf --case finetuning --gpu_num 3 --is_continue
CUDA_VISIBLE_DEVICES=0,1,2 python exp_runner_ho_dp_hand.py --mode train --conf confs/dexycb_dino_hand.conf --case finetuning --gpu_num 3 --is_continue
```

## Test
For mesh inference, 
```
CUDA_VISIBLE_DEVICES=0 python exp_runner_ho_dp_hand.py --mode test_mesh --conf confs/ho3d_dino_hand_test.conf --case finetuning --gpu_num 1 --is_continue
CUDA_VISIBLE_DEVICES=0 python exp_runner_ho_dp_hand.py --mode test_mesh --conf confs/dexycb_dino_hand_test.conf --case finetuning --gpu_num 1 --is_continue
```

For novel view synthesis inference,
```
CUDA_VISIBLE_DEVICES=0 python exp_runner_ho_dp_hand.py --mode test_image --conf confs/ho3d_dino_hand_test.conf --case finetuning --gpu_num 1 --is_continue
CUDA_VISIBLE_DEVICES=0 python exp_runner_ho_dp_hand.py --mode test_image --conf confs/dexycb_dino_hand_test.conf --case finetuning --gpu_num 1 --is_continue
```

## Evaluation
Users need to download `YCBmodels` from https://rse-lab.cs.washington.edu/projects/posecnn/ for mesh evaluation on `HO3D` and `DexYCB`.
For mesh evaluation, 
```
python eval_ho3d_mesh.py --conf confs/ho3d_dino_hand_test.conf --case finetuning --shape_path PATH_TO_YCBmodels
python eval_dexycb_mesh.py --conf confs/dexycb_dino_hand_test.conf --case finetuning --shape_path PATH_TO_YCBmodels
```

For novel view synthesis evaluation,
```
python eval_image.py -T ho3d -P PATH_TO_THE_NOVEL_VIEW_SYNTHESIS_INFERENCE
python eval_image.py -T dexycb -P PATH_TO_THE_NOVEL_VIEW_SYNTHESIS_INFERENCE
```
