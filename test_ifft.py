import numpy as np
import torch
import cv2
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import os
import time
from typing import Literal


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
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask


def fft2(x):
    """ FFT with shifting DC to the center of the image"""
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])

def ifft2(x):
    """ IFFT with shifting DC to the corner of the image prior to transform"""
    # x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2])).squeeze().numpy()


def read_our_images(size: int = 256):
    src_dir = 'MRI/test'
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


def load_mat(file):
    try:
        from scipy.io import loadmat
        file = loadmat(file)
    except:
        import mat73
        file = mat73.loadmat(file)
    keys = list(file.keys())
    keys_useful = [key for key in keys if not key.startswith('__')]
    key = keys_useful[0]
    print(key)
    data = file.get(key)
    data = np.array(data).transpose()  # (H*W, N) -> (N, H*W)
    return data


def read_automap_images(size: int = 256):
    src_file = 'data/test_x_real.mat'
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


def test_ifft(
    which_data: Literal['ours', 'automap'],
    mask_type: Literal['gaussian2d', 'uniformrandom2d', 'gaussian1d', 'uniform1d', 'customized'],
    size: int = 256
):
    """
    For two cases:
    (1) images -> FFT -> kspace -> IFFT -> reconstructed images;
    (2) images -> FFT -> kspace -> mask -> IFFT -> under-sampled reconstructed images;
    visualize the results and compute PSNR and SSIM.
    TODO: add noise to images or kspace.
    """
    time_str = time.strftime('%Y%m%d-%H%M%S')
    dst_dir = 'ifft-output/{}-{}-{}-{}'.format(which_data, size, mask_type, time_str)
    vis_dir  = os.path.join(dst_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    metrics_file = os.path.join(dst_dir, 'metrics.txt')
    print('Output directory:', dst_dir)
    
    full_psnr_list = []
    full_ssim_list = []
    uns_psnr_list = []
    uns_ssim_list = []
    
    all_imgs = read_our_images(size) if which_data == 'ours' else read_automap_images(size)
    print('loaded all images:', all_imgs.shape)
    
    # the same mask for all fft results
    if mask_type == 'customized':
        mask = np.zeros((size, size))
        mask[::2, ::2] = 1
    else:
        mask = torch.zeros(1, 1, size, size)
        mask = get_mask(mask, size, 1, type=mask_type, acc_factor=2, center_fraction=0.15, fix=False).squeeze().numpy()
    cv2.imwrite(f'{dst_dir}/mask.png', mask * 255)
    print(f'mask shape {mask.shape}, sum: {mask.sum()}/{size**2}, saved as {dst_dir}/mask.png')
    
    num = all_imgs.shape[0]
    for k, img in tqdm(enumerate(all_imgs), total=num):
        kspace = fft2(img)
        
        reconstruct = ifft2(kspace).real
        psnr_full = psnr(img, reconstruct)
        ssim_full = ssim(img, reconstruct, data_range=reconstruct.max() - reconstruct.min())
        full_psnr_list.append(psnr_full)
        full_ssim_list.append(ssim_full)
        
        kspace_under = kspace * mask
        
        reconstruct_under = ifft2(kspace_under).real
        h, w = img.shape
        # reconstruct_under = reconstruct_under[:h//scale, :w//scale]
        reconstruct_under = cv2.resize(reconstruct_under, (w, h), interpolation=cv2.INTER_LINEAR)
        psnr_uns = psnr(img, reconstruct_under)
        ssim_uns = ssim(img, reconstruct_under, data_range=reconstruct_under.max() - reconstruct_under.min())
        uns_psnr_list.append(psnr_uns)
        uns_ssim_list.append(ssim_uns)
        
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[1].imshow(reconstruct, cmap='gray')
        axs[1].set_title('Reconstructed,\n PSNR: {:.2f},\n SSIM: {:.2f}'.format(psnr_full, ssim_full))
        axs[2].imshow(reconstruct_under, cmap='gray')
        axs[2].set_title('Reconstructed Under-sampled,\n PSNR: {:.2f},\n SSIM: {:.2f}'.format(psnr_uns, ssim_uns))
        
        plt.savefig(f'{vis_dir}/{k:04d}.png')
        plt.close()
    
    full_psnr = np.mean(full_psnr_list)
    full_ssim = np.mean(full_ssim_list)
    uns_psnr = np.mean(uns_psnr_list)
    uns_ssim = np.mean(uns_ssim_list)
    
    print('Full PSNR: {:.6f}, SSIM: {:.6f}'.format(full_psnr, full_ssim))
    print('Under-sampled PSNR: {:.6f}, SSIM: {:.6f}'.format(uns_psnr, uns_ssim))
    
    with open(metrics_file, 'w') as f:
        f.write(f'Full PSNR: {full_psnr}, SSIM: {full_ssim}\n')
        f.write(f'Under-sampled PSNR: {uns_psnr}, SSIM: {uns_ssim}\n')
    
    return full_psnr, full_ssim, uns_psnr, uns_ssim


def main():
    f = open('ifft-output.txt', 'w')
    for whilch_data in ['ours', 'automap']:
        for mask_type in ['gaussian2d', 'uniformrandom2d', 'gaussian1d', 'uniform1d']:
            for size in [64, 128, 256]:
                full_psnr, full_ssim, uns_psnr, uns_ssim = test_ifft(whilch_data, mask_type, size)
                f.write(f'{whilch_data}, {mask_type}, {size}\n')
                f.write(f'Full PSNR: {full_psnr}, SSIM: {full_ssim}\n')
                f.write(f'Under-sampled PSNR: {uns_psnr}, SSIM: {uns_ssim}\n')
    f.close()


if __name__ == '__main__':
    main()
    