import glob
import os
from typing import Literal
import cv2
import numpy as np
import torch
from scipy.io import savemat, loadmat
import mat73


def array_info(x: np.ndarray):
    print(f"shape: {x.shape}, dtype: {x.dtype}, min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}, std: {x.std():.4f}")


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
    """
    Get a mask for undersampling, implemented by Korean authors.
    """
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    if type == 'gaussian2d':
        mask = torch.zeros_like(img)
        cov_factor = size * (1.5 / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        else:
            for i in range(batch_size):
                # sample different masks for batch
                samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == 'uniformrandom2d':
        mask = torch.zeros_like(img)
        if fix:
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[:, ...] = mask_b
        else:
            for i in range(batch_size):
                # sample different masks for batch
                mask_vec = torch.zeros([1, size * size])
                samples = np.random.choice(size * size, int(Nsamp))
                mask_vec[:, samples] = 1
                mask_b = mask_vec.view(size, size)
                mask[i, ...] = mask_b
    elif type == 'gaussian1d':
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[... , int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[... , c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, :, int_samples] = 1
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from + Nsamp_center] = 1
    elif type == 'uniform1d':
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from+Nsamp_center] = 1
    elif type == 'grid':
        mask = torch.zeros_like(img)
        mask[..., ::2, ::2] = 1
        # keep center
        c_size = int(size * center_fraction)
        c_from = size // 2 - c_size // 2
        mask[..., c_from:c_from + c_size, c_from:c_from + c_size] = 1
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask


def fft2(x):
    """ FFT with shifting DC to the center of the image"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])

def ifft2(x):
    """ IFFT with shifting DC to the corner of the image prior to transform"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def read_our_images(size: int = 256, split: Literal['train', 'test'] = 'test'):
    src_dir = f'MRI/{split}'
    png_files = glob.glob(f'{src_dir}/*.png')
    png_files.sort()
    imgs = []
    for png_file in png_files:
        img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)  # (N, H, W)
    m1, m2 = imgs.min(), imgs.max()
    imgs = (imgs - m1) / (m2 - m1)  # normalize to [0, 1]
    return imgs


def load_mat(file: os.PathLike):
    try:
        data = loadmat(file)
    except:
        data = mat73.loadmat(file)
    keys = list(data.keys())
    keys_useful = [key for key in keys if not key.startswith('__')]
    key = keys_useful[0]
    data = data.get(key)
    data = np.array(data).T  # (H*W, N) -> (N, H*W)
    print(f'loaded data from {file}, transposed from {data.T.shape} to {data.shape}')
    return data


def save_mat(file, data: np.ndarray):
    # to be consistent with the original data format, data should be (H*W, N)
    try:
        savemat(file, {'data': data})
    except:
        mat73.savemat(file, {'data': data})
    print(f'saved data {data.shape} to {file}')


def read_automap_images(size: int = 256, split: Literal['train', 'test'] = 'test', part: Literal['real', 'imag'] = 'real'):
    src_file = f'data/{split}_x_{part}.mat'
    data = load_mat(src_file)
    w = int(np.sqrt(data.shape[1]))  # 64
    data = data.reshape(-1, w, w)  # (N, 64, 64)

    if size != w:
        data_resized = []
        for img in data:
            img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            data_resized.append(img_resized)
        data = np.stack(data_resized, axis=0)
    
    m1, m2 = data.min(), data.max()
    data = (data - m1) / (m2 - m1)  # normalize to [0, 1]
    return data
