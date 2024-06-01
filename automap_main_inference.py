import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

from data_loader.automap_inference_data_generator import InferenceDataGenerator
from trainers.automap_inferencer import AUTOMAP_Inferencer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    data = InferenceDataGenerator(config)
    
    model = tf.keras.models.load_model(config.loadmodel_dir)
    
    inferencer = AUTOMAP_Inferencer(model, data, config)
    output = inferencer.inference()  # (N, H*W)
    output = output.reshape(-1, config.im_h, config.im_w)
    img_gt = data.output  # (N, H*W)
    img_gt = img_gt.reshape(-1, config.im_h, config.im_w)
    psnr_output = psnr(img_gt, output)
    ssim_output = ssim(img_gt, output, data_range=output.max() - output.min())
    print('PSNR: ', psnr_output)
    print('SSIM: ', ssim_output)
    save_dir = os.path.dirname(config.save_inference_output)
    metrics_file = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f'PSNR: {psnr_output}\n')
        f.write(f'SSIM: {ssim_output}\n')


if __name__ == '__main__':
    main()