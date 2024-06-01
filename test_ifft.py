import numpy as np
import torch
import cv2
import glob
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import os


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


def fft2(x):
    """ FFT with shifting DC to the center of the image"""
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])

def ifft2(x):
    """ IFFT with shifting DC to the corner of the image prior to transform"""
    # x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2])).squeeze().numpy()


def main():
    src_dir = 'MRI/test'
    size = 64
    mask_type = 'uniform1d'
    # mask_type = 'gaussian1d'
    # mask_type = 'uniformrandom2d'
    # mask_type = 'gaussian2d'
    # mask_type = 'grid'
    import time
    time_str = time.strftime('%Y%m%d-%H%M%S')
    dst_fir = 'output/traditional-ifft-ourdata-{}-{}'.format(mask_type, time_str)
    os.makedirs(dst_fir, exist_ok=True)
    
    full_psnr_list = []
    full_ssim_list = []
    uns_psnr_list = []
    uns_ssim_list = []
    
    png_files = glob.glob(f'{src_dir}/*.png')
    png_files.sort()
    for png_file in tqdm(png_files):
        img = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)
        
        kspace = fft2(img)
        # kspace = np.fft.fftshift(kspace)
        
        reconstruct = ifft2(kspace).real
        # reconstruct = (reconstruct - reconstruct.min()) / (reconstruct.max() - reconstruct.min())
        psnr_full = psnr(img, reconstruct)
        ssim_full = ssim(img, reconstruct, data_range=reconstruct.max() - reconstruct.min())
        full_psnr_list.append(psnr_full)
        full_ssim_list.append(ssim_full)
        
        # kspace_under = kspace * get_mask(torch.from_numpy(img).unsqueeze(0).unsqueeze(0), 256, 1, type=mask_type, acc_factor=2, center_fraction=0.15, fix=False)
        # mask = np.zeros([256, 256])
        # mask[::2] = 1
        # scale = 4
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if i % scale == j % scale == 0:
        #             mask[i, j] = 1
        # mask = get_mask(torch.from_numpy(img).unsqueeze(0).unsqueeze(0), 256, 1, type=mask_type, acc_factor=2, center_fraction=0.15, fix=False).squeeze().numpy()
        
        mask = torch.zeros(1, 1, size, size)
        mask = get_mask(mask, size, 1, type=mask_type, acc_factor=2, center_fraction=0.15, fix=False).squeeze().numpy()
        print(mask.sum())
        kspace_under = kspace * mask
        
        reconstruct_under = ifft2(kspace_under).real
        h, w = img.shape
        # reconstruct_under = reconstruct_under[:h//scale, :w//scale]
        reconstruct_under = cv2.resize(reconstruct_under, (w, h), interpolation=cv2.INTER_LINEAR)
        # reconstruct_under = (reconstruct_under - reconstruct_under.min()) / (reconstruct_under.max() - reconstruct_under.min())
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
        
        plt.savefig(f'{dst_fir}/{png_file.split("/")[-1]}')
        plt.close()
    
    print('Full PSNR: {:.2f}, SSIM: {:.2f}'.format(np.mean(full_psnr_list), np.mean(full_ssim_list)))
    print('Under-sampled PSNR: {:.2f}, SSIM: {:.2f}'.format(np.mean(uns_psnr_list), np.mean(uns_ssim_list)))
          
        

if __name__ == '__main__':
    main()
    