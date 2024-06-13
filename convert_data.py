import os
import cv2
import numpy as np

from utils import *




def fft_from_images(
    save_dir: os.PathLike,
    which_data: Literal['ours', 'automap'],
    mask_type: Literal['gaussian2d', 'uniformrandom2d', 'gaussian1d', 'uniform1d', 'grid', 'none'],
    size: int = 256,
    noise: bool = False
):
    """
    Load images, do fft, then save results in .mat
    :param save_dir: output directory
    :param which_data: 'ours' or 'automap'
    :param mask_type: 'gaussian2d', 'uniformrandom2d', 'gaussian1d', 'uniform1d', 'grid', 'none'
    :param size: image size
    :param noise: whether to add noise
    """
    os.makedirs(save_dir, exist_ok=True)

    if mask_type != 'none':
        # the same mask for all fft results
        mask = torch.zeros(1, 1, size, size)
        mask = get_mask(mask, size, 1, type=mask_type, acc_factor=2, center_fraction=0.15, fix=False).squeeze().numpy()
        cv2.imwrite(f'{save_dir}/mask.png', mask * 255)
        print(f'mask type {mask_type}, shape {mask.shape}, sum {mask.sum()}/{size**2}, saved as {save_dir}/mask.png')
    else:
        mask = np.ones((size, size))

    for split in ['train', 'test']:
        real_images = read_our_images(size=size, split=split) if which_data == 'ours' else read_automap_images(size=size, split=split)

        fft = fft2(real_images).numpy()
        print(f'fft from real data: {fft.shape}')

        if mask_type != 'none':
            fft *= mask
            print(f'under-sampled: {fft.shape}')
        
        if noise:
            # add noise to fft results
            noise = np.random.normal(0, 0.01, fft.shape)
            fft += noise
            print(f'added noise: {fft.shape}')
        
        # save imgs
        real_images = real_images.reshape(-1, size*size).transpose()  # (size*size, num) to be consistent with the original data
        save_file = os.path.join(save_dir, f'{split}_x_real.mat')
        save_mat(save_file, real_images)
        
        # save fft results
        fft = fft.reshape(-1, size*size).transpose()  # (size*size, num)
        real_part, imag_part = np.real(fft), np.imag(fft)
        fft_concat = np.concatenate([real_part, imag_part], axis=0)  # (2*size*size, num) to be consistent with the original data
        save_file = os.path.join(save_dir, f'{split}_fft.mat')
        save_mat(save_file, fft_concat)


def fft_from_out_test_images_given_mask(mask: np.ndarray, size: int = 256):
    save_dir = f'data-transfer-test'
    os.makedirs(save_dir, exist_ok=True)
    real_test_images = read_our_images(size=size, split='test')
    fft = fft2(real_test_images).numpy()
    fft *= mask
    print(f'under-sampled: {fft.shape}')
    real_test_images = real_test_images.reshape(-1, size*size).transpose()  # (size*size, num) to be consistent with the original data
    save_file = os.path.join(save_dir, f'test_x_real.mat')
    save_mat(save_file, real_test_images)
    fft = fft.reshape(-1, size*size).transpose()  # (size*size, num)
    real_part, imag_part = np.real(fft), np.imag(fft)
    fft_concat = np.concatenate([real_part, imag_part], axis=0)  # (2*size*size, num) to be consistent with the original data
    save_file = os.path.join(save_dir, f'test_fft.mat')
    save_mat(save_file, fft_concat)


def load_mask_from_png(mask_file: os.PathLike, size: int = 256):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = mask / 255
    return mask


def main():
    for which_data in ['ours', 'automap']:
        for mask_type in ['uniform1d']:
            for size in [64, 128]:
                save_dir = f'data-fft/{which_data}_{size}/down_{mask_type}'
                fft_from_images(save_dir, which_data, mask_type, size, noise=False)


def main_expanded():
    mask_file = 'data-fft/automap_64/down_uniform1d/mask.png'
    mask = load_mask_from_png(mask_file, size=64)
    fft_from_out_test_images_given_mask(mask, size=64)
    

if __name__ == '__main__':
    main()
