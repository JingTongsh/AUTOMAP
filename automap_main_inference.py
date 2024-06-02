import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import tensorflow as tf

from data_loader.automap_inference_data_generator import InferenceDataGenerator
from trainers.automap_inferencer import AUTOMAP_Inferencer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils.tools import fft2, ifft2


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    data = InferenceDataGenerator(config)

    # image ground truth
    img_gt = data.output  # (N, H*W)
    img_gt = img_gt.reshape(-1, config.im_h, config.im_w)
    
    # ifft reconstruction
    fft = data.input  # (N, 2*H*W)
    im_h, im_w = config.im_h, config.im_w
    fft_real, fft_imag = fft[:, :im_h * im_w], fft[:, im_h * im_w:]
    fft = fft_real + 1j * fft_imag  # (N, H*W)
    fft = fft.reshape(-1, im_h, im_w)  # (N, H, W)
    ifft = ifft2(fft).numpy().real  # (N, H, W)
    
    # AUTOMAP reconstruction
    model = tf.keras.models.load_model(config.loadmodel_dir)
    inferencer = AUTOMAP_Inferencer(model, data, config)
    output = inferencer.inference()  # (N, H*W)
    output = output.reshape(-1, config.im_h, config.im_w)
    
    # metrics
    psnr_ifft = psnr(img_gt, ifft)
    ssim_ifft = ssim(img_gt, ifft, data_range=ifft.max() - ifft.min())
    print('PSNR IFFT: ', psnr_ifft)
    print('SSIM IFFT: ', ssim_ifft)
    psnr_output = psnr(img_gt, output)
    ssim_output = ssim(img_gt, output, data_range=output.max() - output.min())
    print('PSNR AUTOMAP: ', psnr_output)
    print('SSIM AUTOMAP: ', ssim_output)
    save_dir = os.path.dirname(config.save_inference_output)
    metrics_file = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f'PSNR IFFT: {psnr_ifft}\n')
        f.write(f'SSIM IFFT: {ssim_ifft}\n')
        f.write(f'PSNR AUTOMAP: {psnr_output}\n')
        f.write(f'SSIM AUTOMAP: {ssim_output}\n')
    
    # visualize
    vis_dir = os.path.join(save_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    num = img_gt.shape[0]
    print('visualizing results...')
    for i in tqdm(range(num)):
        img_gt_i = img_gt[i]
        ifft_i = ifft[i]
        output_i = output[i]
        ifft_pnsr, ifft_ssim = psnr(img_gt_i, ifft_i), ssim(img_gt_i, ifft_i, data_range=ifft_i.max() - ifft_i.min())
        output_pnsr, output_ssim = psnr(img_gt_i, output_i), ssim(img_gt_i, output_i, data_range=output_i.max() - output_i.min())

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img_gt_i, cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[1].imshow(ifft_i, cmap='gray')
        axs[1].set_title(f'IFFT Reconstruction,\n PSNR: {ifft_pnsr:.2f},\n SSIM: {ifft_ssim:.2f}')
        axs[2].imshow(output_i, cmap='gray')
        axs[2].set_title(f'AUTOMAP Reconstruction,\n PSNR: {output_pnsr:.2f},\n SSIM: {output_ssim:.2f}')
        plt.savefig(f'{vis_dir}/{i:04d}.png')
        plt.close()



if __name__ == '__main__':
    main()