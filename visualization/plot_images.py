
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import *
from data.data_processing import onehot_encode

def save_images_with_metrics(true_images, decoded_images, natural_images, true_labels, pred_labels, kl_all, save_dir):
    """ Save plot showing natural image, true iconic image, and decoded iconic image
        with PSNR, SSIM, and KL metrics between the true and decoded iconic image.

    Args: 
        true_images: The true iconic images.
        decoded_images: The decoded iconic images.
        natural_images: The corresponding natural images.
        true_labels: The labels for each image.
        pred_labels: The labels predicted by the model.
        kl_all: KL divergences between every true and decoded iconic image pair.
        save_dir: Path to directory to save the images.

    """
    for i, (img1, img2, img3) in enumerate(zip(natural_images, true_images, decoded_images)):
        
        psnr = compute_psnr(img2, img3)
        ssim = compute_ssim(img2, img3)
        kl_div = kl_all[i]

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(img1)
        ax[1].set_title('Natural')
        ax[1].imshow(img2)
        ax[1].set_title('True')
        ax[2].imshow(img3)
        ax[2].set_title('Decoded')

        true = true_labels[i]
        pred = pred_labels[i]
        correct = (true == pred).astype(int)

        fig.suptitle('PSNR: {:.3f}, SSIM: {:.3f}, KL: {:.3f},\nTrue: {:d}, Pred: {:d} '.format(
            psnr, ssim, kl_div, int(true), int(pred)), fontsize=16)

        img_name = os.path.join(save_dir, 'image{:d}'.format(i))
        img_name = img_name + '_correct' if correct==1 else img_name + '_incorrect'
        plt.savefig(img_name + '.png')
        plt.tight_layout()
        plt.close()

def save_decoded_images(decoded_images, save_dir):
    """ Save decoded iconic images.
    """
    for i, img in enumerate(decoded_images):
        img_name = os.path.join(save_dir, 'image{:d}'.format(i))
        plt.imsave(img_name + '.png', img)
