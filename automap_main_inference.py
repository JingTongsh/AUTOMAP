import glob
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
    ssim_ifft = ssim(img_gt, ifft, data_range=1)
    print('PSNR IFFT: ', psnr_ifft)
    print('SSIM IFFT: ', ssim_ifft)
    psnr_output = psnr(img_gt, output)
    ssim_output = ssim(img_gt, output, data_range=1)
    print('PSNR AUTOMAP: ', psnr_output)
    print('SSIM AUTOMAP: ', ssim_output)
    save_dir = os.path.dirname(config.save_inference_output)
    metrics_file = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('Overall Metrics\n')
        f.write(f'PSNR IFFT: {psnr_ifft}\n')
        f.write(f'SSIM IFFT: {ssim_ifft}\n')
        f.write(f'PSNR AUTOMAP: {psnr_output}\n')
        f.write(f'SSIM AUTOMAP: {ssim_output}\n')
    
    if 'expanded' in config.data_dir or 'transfer' in config.data_dir:
        # for our data, evaluate metrics for each sequence
        print('Evaluating metrics for each sequence...')
        original_dir = 'MRI/test'
        png_files = glob.glob(f'{original_dir}/*.png')
        png_files.sort()
        id_list = ['_'.join(os.path.basename(file).split('_')[:2]) for file in png_files]
        id_list = list(set(id_list))
        id_list.sort()
        
        for i in id_list:
            sequence_idx = [j for j, file in enumerate(png_files) if i in file]
            sequence_gt = img_gt[sequence_idx]
            sequence_ifft = ifft[sequence_idx]
            sequence_output = output[sequence_idx]
            sequence_psnr_ifft = psnr(sequence_gt, sequence_ifft)
            sequence_ssim_ifft = ssim(sequence_gt, sequence_ifft, data_range=1)
            sequence_psnr_output = psnr(sequence_gt, sequence_output)
            sequence_ssim_output = ssim(sequence_gt, sequence_output, data_range=1)
            with open(metrics_file, 'a') as f:
                f.write('-' * 30 + '\n')
                f.write(f'sequence: {i}\n')
                f.write('-' * 30 + '\n')
                f.write(f'PSNR IFFT: {sequence_psnr_ifft}\n')
                f.write(f'SSIM IFFT: {sequence_ssim_ifft}\n')
                f.write(f'PSNR AUTOMAP: {sequence_psnr_output}\n')
                f.write(f'SSIM AUTOMAP: {sequence_ssim_output}\n')
            print(f'sequence: {i}')
            print(f'PSNR IFFT: {sequence_psnr_ifft}')
            print(f'SSIM IFFT: {sequence_ssim_ifft}')
            print(f'PSNR AUTOMAP: {sequence_psnr_output}')
            print(f'SSIM AUTOMAP: {sequence_ssim_output}')
            
    
    # visualize
    vis_dir = os.path.join(save_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    num = img_gt.shape[0]
    print('visualizing results...')
    for i in tqdm(range(num)):
        img_gt_i = img_gt[i]
        ifft_i = ifft[i]
        output_i = output[i]
        ifft_pnsr, ifft_ssim = psnr(img_gt_i, ifft_i), ssim(img_gt_i, ifft_i, data_range=1)
        output_pnsr, output_ssim = psnr(img_gt_i, output_i), ssim(img_gt_i, output_i, data_range=1)

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img_gt_i, cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[1].imshow(ifft_i, cmap='gray')
        axs[1].set_title(f'IFFT Reconstruction,\n PSNR: {ifft_pnsr:.2f},\n SSIM: {ifft_ssim:.2f}')
        axs[2].imshow(output_i, cmap='gray')
        axs[2].set_title(f'AUTOMAP Reconstruction,\n PSNR: {output_pnsr:.2f},\n SSIM: {output_ssim:.2f}')
        plt.savefig(f'{vis_dir}/{i:04d}.png')
        plt.close()
    
    print(f'Inference results saved at {config.save_inference_output}')


if __name__ == '__main__':
    main()
