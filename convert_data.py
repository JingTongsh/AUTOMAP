import glob
import os
import cv2
import mat73
from scipy.io import savemat, loadmat
import mat73
import numpy as np
from tqdm import tqdm
import torch


def array_info(x: np.ndarray):
    print(f"shape: {x.shape}, dtype: {x.dtype}, min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}, std: {x.std():.4f}")


def load_mat(file):
    try:
        file = loadmat(file)
    except:
        file = mat73.loadmat(file)
    keys = list(file.keys())
    keys_useful = [key for key in keys if not key.startswith('__')]
    key = keys_useful[0]
    print(key)
    data = file.get(key)
    data = np.array(data).transpose()
    return data


def save_mat(file, data):
    print(file, data.shape)
    try:
        savemat(file, {'data': data})
    except:
        mat73.savemat(file, {'data': data})


def fft_one_image(img):
    img = np.fft.fft2(img)
    # img = np.fft.fftshift(img)
    return img


def fft_1d(img):
    img = img.flatten()
    img = np.fft.fft(img)
    # img = np.fft.fftshift(img)
    return img


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
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
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask


def fft_all_images(src_dir: os.PathLike, dst_dir: os.PathLike, down: bool):
    """
    read .png files, do fft, then save results in .mat
    """
    os.makedirs(dst_dir, exist_ok=True)
    real_images = []
    results = []
    for img_file in tqdm(glob.glob(f'{src_dir}/*.png')):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # normalize to [0, 1]
        result = fft_one_image(img)

        if down:
            mask = get_mask(img, 64, 1, type='uniform1d', acc_factor=2, center_fraction=0.15, fix=False)
        real_part = np.real(result).flatten()
        imag_part = np.imag(result).flatten()
        result = np.concatenate((real_part, imag_part))
        results.append(result)
        real_images.append(img.flatten())
    real_images = np.stack(real_images, axis=0)  # shape=(N, H*W)
    results = np.stack(results, axis=0)  # shape=(N, 2*H*W)
    real_file = os.path.join(dst_dir, 'x_real.mat')
    fft_file = os.path.join(dst_dir, 'x_input.mat')
    real_images = real_images.transpose()
    results = results.transpose()
    print(real_images.shape, results.shape)
    array_info(real_images)
    array_info(results)
    # scale value to [-pi/2, pi/2]
    r = max(abs(results.min()), abs(results.max()))
    scale = np.pi / 2 / r
    results *= scale
    array_info(results)
    savemat(real_file, {'x_real': real_images})
    savemat(fft_file, {'x_fft': results})


def fft_original_data(src_real, src_imag, dst_file):
    """
    read .mat file, do fft, then save result in .mat
    """
    src_real = load_mat(src_real)  # (N, 4096)
    src_imag = load_mat(src_imag)  # (N, 4096)
    src_complex = src_real + 1j * src_imag
    src_complex = src_complex.reshape(-1, 64, 64)
    fft_kspace = np.fft.fft2(src_complex)
    # fft_kspace = np.fft.fftshift(fft_kspace)  # (N, 64, 64)
    fft_real = np.real(fft_kspace).reshape(-1, 4096)  # (N, 4096)
    fft_imag = np.imag(fft_kspace).reshape(-1, 4096)  # (N, 4096)
    fft_concat = np.concatenate((fft_real, fft_imag), axis=1)  # (N, 8192)
    # scale value to [-pi/2, pi/2]
    r = max(abs(fft_concat.min()), abs(fft_concat.max()))
    scale = np.pi / 2 / r
    fft_concat *= scale
    array_info(fft_concat)
    save_mat(dst_file, fft_concat.transpose())
    
    


if __name__ == '__main__':
    out_dir = 'data_our64'
    os.makedirs(out_dir, exist_ok=True)
    fft_all_images('MRI/train', os.path.join(out_dir, 'train'), down=True)
    fft_all_images('MRI/test', os.path.join(out_dir, 'test'), down=True)
    print('Done converting our data.')

    src_real = 'data/train_x_real.mat'
    src_imag = 'data/train_x_img.mat'
    dst_file = 'data/train_fft_down_input.mat'
    fft_original_data(src_real, src_imag, dst_file, down=True)
    src_real = 'data/test_x_real.mat'
    src_imag = 'data/test_x_img.mat'
    dst_file = 'data/test_fft_down_input.mat'
    fft_original_data(src_real, src_imag, dst_file, down=True)
    