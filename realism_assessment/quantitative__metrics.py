import os
from PIL import Image as im
import numpy as np
import torch
import piq
import lpips
from skimage.metrics import mean_squared_error as mse


# load images and calculate metric --> Not very efficient and quick
def load_image_assess(dirt1, dirt2, name):
    score_all = []
    set1 = os.listdir(dirt1)
    set2 = os.listdir(dirt2)
    if name == 'lpips':
        lpips_metric = lpips.LPIPS(net='vgg')
    for i in range(0, len(set1)):
        image1 = np.expand_dims(np.transpose(np.asarray(im.open(os.path.join(dirt1, set1[i]))), (2, 0, 1)), axis=0)
        image2 = np.expand_dims(np.transpose(np.asarray(im.open(os.path.join(dirt2, set2[i]))), (2, 0, 1)), axis=0)
        if name == 'psnr':
            score = piq.psnr(torch.from_numpy(image1), torch.from_numpy(image2), data_range=255, reduction='none')
            score_all.append(score.numpy()[0])
        elif name == 'mse':
            score = mse(image1, image2)
            score_all.append(score)
        elif name == 'ssim':
            score = piq.ssim(torch.from_numpy(image1), torch.from_numpy(image2), data_range=255)
            score_all.append(score.unsqueeze(0).numpy()[0])
        elif name == 'msssim':
            score = piq.multi_scale_ssim(torch.from_numpy(image1), torch.from_numpy(image2), data_range=255)
            score_all.append(score.unsqueeze(0).numpy()[0])
        elif name == 'lpips':
            # images need to be normalized between [-1, 1]
            image1 = image1.astype('float64')
            image2 = image2.astype('float64')
            image1 = (image1.astype('float64')*(2/255)) - 1
            image2 = (image2.astype('float64')*(2/255)) - 1
            score = lpips_metric(torch.from_numpy(image1.astype('uint8')), torch.from_numpy(image2.astype('uint8')))
            score_all.append(score.data.numpy().min())  # contains only one score, but is stuck in a 5 dim array
        print('Determined score for image:', set1[i])

    return score_all


# Define data directories
dir_GT = '...'      # original images
dir_syn1 = '...'    # synthetic images after training with 0.3% of full train set
dir_syn2 = '...'    # synthetic images after training with 2% of full train set
dir_syn3 = '...'    # synthetic images after training with full train set

# Load data and calculate metric
metric_name = 'ssim'        # insert name of metric to be calculated; psnr, mse, ssim, msssim, lpips
print('Loading data and calculating metric:', metric_name)
scores = load_image_assess(dir_GT, dir_syn1, metric_name)
