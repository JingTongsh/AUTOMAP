import os
import time
from typing import Literal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from utils import *


def test_ifft(
    which_data: Literal['ours', 'automap'],
    mask_type: Literal['gaussian2d', 'uniformrandom2d', 'gaussian1d', 'uniform1d', 'grid'],
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
    
    all_imgs = read_our_images(size=size) if which_data == 'ours' else read_automap_images(size=size)
    print('loaded all images:', all_imgs.shape)
    
    # the same mask for all fft results
    mask = torch.zeros(1, 1, size, size)
    mask = get_mask(mask, size, 1, type=mask_type, acc_factor=2, center_fraction=0.15, fix=False).squeeze().numpy()
    cv2.imwrite(f'{dst_dir}/mask.png', mask * 255)
    print(f'mask shape {mask.shape}, sum: {mask.sum()}/{size**2}, saved as {dst_dir}/mask.png')
    
    num = all_imgs.shape[0]
    for k, img in tqdm(enumerate(all_imgs), total=num):
        kspace = fft2(img)
        
        reconstruct = ifft2(kspace).numpy().real
        psnr_full = psnr(img, reconstruct)
        ssim_full = ssim(img, reconstruct, data_range=reconstruct.max() - reconstruct.min())
        full_psnr_list.append(psnr_full)
        full_ssim_list.append(ssim_full)
        
        kspace_under = kspace * mask
        
        reconstruct_under = ifft2(kspace_under).numpy().real
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
    f = open('ifft-output.txt', 'a')
    for whilch_data in ['ours', 'automap']:
        for mask_type in ['grid']:
            for size in [64, 128, 256]:
                full_psnr, full_ssim, uns_psnr, uns_ssim = test_ifft(whilch_data, mask_type, size)
                f.write(f'{whilch_data}, {mask_type}, {size}\n')
                f.write(f'Full PSNR: {full_psnr}, SSIM: {full_ssim}\n')
                f.write(f'Under-sampled PSNR: {uns_psnr}, SSIM: {uns_ssim}\n')
    f.close()


def test():
    imgs = read_our_images(256)
    print(imgs.shape)
    fft_separately = []
    for img in imgs:
        fft = fft2(img).squeeze().numpy()
        fft_separately.append(fft)
    fft_separately = np.stack(fft_separately, axis=0)
    print(fft_separately.shape)
    fft_whole = fft2(imgs).squeeze().numpy()
    print(fft_whole.shape)
    print(np.allclose(fft_separately, fft_whole))


if __name__ == '__main__':
    main()
    # test()
    