# Description: Preprocess the BraTS2021 dataset into numpy arrays
# adpated from https://github.com/AntanasKascenas/DenoisingAE/tree/master

import torch
import random
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F
import cv2

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def normalise_percentile(volume):
    """
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
    for mdl in range(volume.shape[1]):
        v_ = volume[:, mdl, :, :].reshape(-1)
        v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
        p_99 = torch.quantile(v_, 0.99)
        volume[:, mdl, :, :] /= p_99
        # clip the values to [0, 1]
        torch.clamp(volume[:, mdl, :, :], 0, 1)
    return volume

# def center_crop(volume, target_shape):
#     h, w, _ = volume.shape
#     th, tw = target_shape, target_shape
#     x1 = int(round((w - tw) / 2.))
#     y1 = int(round((h - th) / 2.))
#     cropped_volume = volume[y1:y1+th, x1:x1+tw, :]
#     return cropped_volume
    

def process_patient_3d(name, path, target_path, mod, 
                       first=0, last=0, 
                       downsample=False, downsample_size=0):
    
    if name.lower() == 'brats':
        flair = nib.load(path / f"{path.name}_flair.nii.gz").get_fdata()
        t1 = nib.load(path / f"{path.name}_t1.nii.gz").get_fdata()
        t1ce = nib.load(path / f"{path.name}_t1ce.nii.gz").get_fdata()
        t2 = nib.load(path / f"{path.name}_t2.nii.gz").get_fdata()
        labels = nib.load(path / f"{path.name}_seg.nii.gz").get_fdata()
    elif name.lower() == "atlas":
        t1 = nib.load(path / f"{path.name}_T1w.nii.gz").get_fdata()
        labels = nib.load(path / f"{path.name}_mask.nii.gz").get_fdata()
        raise ValueError(f"Dataset {name} not supported.")

    assert mod.lower() in ["all", "flair", "t1", "t1ce", "t2"]

    # volume shape: [1, C, H, W, D]
    if mod == "all":
        volume = torch.stack([torch.from_numpy(x) for x in [flair, t1, t1ce, t2]], dim=0).unsqueeze(dim=0)
    elif mod == "flair":
        volume = torch.stack([torch.from_numpy(x) for x in [flair]], dim=0).unsqueeze(dim=0)
    elif mod == "t1":
        volume = torch.stack([torch.from_numpy(x) for x in [t1]], dim=0).unsqueeze(dim=0)
    elif mod == "t1ce":
        volume = torch.stack([torch.from_numpy(x) for x in [t1ce]], dim=0).unsqueeze(dim=0)
    elif mod == "t2":
        volume = torch.stack([torch.from_numpy(x) for x in [t2]], dim=0).unsqueeze(dim=0)

    # exclude first n and last m slices
    # img: 1 C H W D; mask: H W D
    if first > 0 and last > 0:
        volume = volume[:, :, :, :, first:-last]
        labels = labels[:, :, first:-last]
    elif first > 0 and last < 0:
        volume = volume[:, :, :, :, first:]
        labels = labels[:, :, first:]
    elif first < 0 and last > 0:
        volume = volume[:, :, :, :, :-last]
        labels = labels[:, :, :-last]
        
    # mask: 1 1 H W D
    labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)
    
    # # directory to save the preprocessed data
    # patient_dir = target_path / f"patient_{path.name}"
    # patient_dir.mkdir(parents=True, exist_ok=True)
    
    # normalise the intensity values
    volume = normalise_percentile(volume)

    # remove empty slices
    sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
    fs_dim2 = sum_dim2.argmax()
    ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()
    print(f"Patient {path.name} has {fs_dim2} to {ls_dim2} slices with brain tissue.", flush=True)
    
    for slice_idx in range(fs_dim2, ls_dim2):
        if downsample:
            assert downsample_size > 0
            low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(downsample_size, downsample_size))
            low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(downsample_size, downsample_size))
        else:
            # no downsample
            low_res_x = volume[:, :, :, :, slice_idx]
            low_res_y = labels[:, :, :, :, slice_idx]
        
        gradient_map_x = [torch.from_numpy(gradient_map(low_res_x[0, i, ...].numpy())) for i in range(low_res_x.shape[1])]
        gradient_map_x = torch.stack(gradient_map_x, dim=0).unsqueeze(dim=0)
        
        # print(low_res_x.shape, low_res_y.shape, gradient_map_x.shape, flush=True) [1, C, H, W]
        np.savez_compressed(target_path / f"patient_{path.name}_slice_{slice_idx}", x=low_res_x, y=low_res_y, grad_x=gradient_map_x)

def gradient_map(slice):
    grad_x = cv2.Sobel(slice, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(slice, cv2.CV_64F, 0, 1, ksize=3)
    gradient_map = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_map 

def preprocess(name: str, datapath: Path, mod: str, first=-1, last=-1, shape=128, downsample=False):

    all_imgs = sorted(list((datapath).iterdir()))
    sub_dir = f"preprocessed_data_{mod}_{first}{last}_{shape}"
    splits_path = datapath.parent / sub_dir / "data_splits"

    if not splits_path.exists():

        indices = list(range(len(all_imgs)))
        random.seed(10)
        random.shuffle(indices)

        n_train = int(len(indices) * 0.75)
        n_val = int(len(indices) * 0.05)
        n_test = len(indices) - n_train - n_val
       
        split_indices = {}
        split_indices["train"] = indices[:n_train]
        split_indices["val"] = indices[n_train:n_train + n_val]
        split_indices["test"] = indices[n_train + n_val:]

        for split in ["train", "val", "test"]:
            (splits_path / split).mkdir(parents=True, exist_ok=True)
            with open(splits_path / split / "scans.csv", "w") as f:
                f.write("\n".join([all_imgs[idx].name for idx in split_indices[split]]))

    for split in ["train", "val", "test"]:
        paths = [datapath / x.strip() for x in open(splits_path / split / "scans.csv").readlines()]
        print(f"Patients in {split}]: {len(paths)}", flush=True)

        for source_path in tqdm(paths):
            target_path = datapath.parent / sub_dir / f"np_{split}"
            if not target_path.exists():
                target_path.mkdir(parents=True, exist_ok=True)
            process_patient_3d(name, source_path, target_path, mod, first, last, downsample=downsample, downsample_size=shape)


if __name__ == "__main__":
   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='brats', type=str, help="dataset name")
    parser.add_argument("-s", "--source", default='/data/amciilab/yiming/DATA/BraTS21_training/BraTS21', type=str, help="path to Brats2021 Data directory")
    parser.add_argument("-m", "--mod", default='flair', type=str, help="modelity to preprocess")
    parser.add_argument("--first", default=0, 
                        type=int, help="skip first n slices")
    parser.add_argument("--last", default=0,
                        type=int, help="skip last n slices")
    parser.add_argument("--downsample", default=True, type=str2bool, help="downsample the images")
    parser.add_argument("--downsample-size", default=128, type=int, help="shape of the downsampled images")
    
    args = parser.parse_args()
    datapath = Path(args.source)
    preprocess(args.name, datapath, args.mod, args.first, args.last, downsample=args.downsample, shape=args.downsample_size)
