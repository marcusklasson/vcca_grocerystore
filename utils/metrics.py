
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def compute_mse(img_true, img_recon):
    """ Compute mean-squared-error between images.
    """
    mse = mean_squared_error(img_true, img_recon)
    return mse

def compute_psnr(img_true, img_recon, max_val=1.0):
    """ Compute peak signal-to-noise ratio between images.
    """
    psnr = peak_signal_noise_ratio(img_true, img_recon, data_range=max_val)
    return psnr

def compute_ssim(img_true, img_recon):
    """ Compute structural similarity between images.
    """
    ssim = structural_similarity(img_true, img_recon, multichannel=True,
                                 data_range=np.max(img_true) - np.min(img_true))
    return ssim

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    """ Estimate KL divergence between two 
        Gaussian mixtures using monte-carlo sampling.
    """
    X, _ = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()

def compute_accuracy(true_labels, predicted_labels):
    """ Compute classification accuracy.
    """
    num_correct = np.sum((true_labels == predicted_labels).astype(np.float64))
    return num_correct / len(true_labels) 
