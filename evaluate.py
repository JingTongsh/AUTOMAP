from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import math
import matplotlib.pyplot as plt


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
    data = np.array(data).transpose()
    return data


def make01(data):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)


def main():
    original = 'data_mri/test/x_real.mat'
    contrast = 'infer_result.mat'
    original = load_mat(original)
    # ori_min, ori_max = original.min(), original.max()
    # original = (original - ori_min) / (ori_max - ori_min)
    contrast = load_mat(contrast)
    # con_min, con_max = contrast.min(), contrast.max()
    # contrast = (contrast - con_min) / (con_max - con_min)

    print(original.shape)
    print(contrast.shape)

    w = math.sqrt(original.shape[1])
    w = int(w)

    psnr_list, ssim_list = [], []

    for ori, con in zip(original, contrast):
        ori = ori.reshape(w, w)
        con = con.reshape(w, w)
        ori, con = make01(ori), make01(con)
        psnr_list.append(psnr(ori, con))
        ssim_list.append(ssim(ori, con, data_range=1))

    psnr_avg = np.mean(psnr_list)
    ssim_avg = np.mean(ssim_list)

    print(f'Average PSNR: {psnr_avg}')
    print(f'Average SSIM: {ssim_avg}')


def visualize(y_fft, x_gt, x_pred, save_file):
    """
    Compare results from ifft, gt and pred.
    FIXME: try 1d ifft instead of 2d
    """
    w = int(math.sqrt(x_gt.shape[0]))
    y_fft_real = y_fft[:w**2].reshape(w, w)
    y_fft_imag = y_fft[w**2:].reshape(w, w)
    y_fft = y_fft_real + 1j * y_fft_imag
    x_ifft = np.fft.ifft2(y_fft).real
    x_gt = x_gt.reshape(w, w)
    x_pred = x_pred.reshape(w, w)
    x_ifft, x_gt, x_pred = make01(x_ifft), make01(x_gt), make01(x_pred)
    
    psnr_pred = psnr(x_gt, x_pred)
    ssim_pred = ssim(x_gt, x_pred, data_range=1)
    psnr_ifft = psnr(x_gt, x_ifft)
    ssim_ifft = ssim(x_gt, x_ifft, data_range=1)
    
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(x_ifft, cmap='gray')
    axs[0].set_title('IFFT, PSNR: {:.2f}, SSIM: {:.2f}'.format(psnr_ifft, ssim_ifft))
    axs[1].imshow(x_gt, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(x_pred, cmap='gray')
    axs[2].set_title('Prediction, PSNR: {:.2f}, SSIM: {:.2f}'.format(psnr_pred, ssim_pred))
    
    plt.savefig(save_file)
    plt.close()

    
def main2():
    import os
    from tqdm import tqdm
    y_fft = 'data/test_input.mat'
    x_gt = 'data/test_x_real.mat'
    x_pred = 'results/64.mat'
    y_fft = 'data_fft64/test/x_input.mat'
    x_gt = 'data_fft64/test/x_real.mat'
    x_pred = 'results/transfer64.mat'
    y_fft = load_mat(y_fft)
    x_gt = load_mat(x_gt)
    x_pred = load_mat(x_pred)
    
    out_dir = 'output/transfer-64'
    os.makedirs(out_dir, exist_ok=True)
    
    for k in tqdm(range(y_fft.shape[0])):
        visualize(y_fft[k], x_gt[k], x_pred[k], f'{out_dir}/{k}.png')


if __name__ == '__main__':
    main2()
    