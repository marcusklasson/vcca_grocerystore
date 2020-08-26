
import os
import argparse
import time

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture

from data.get_datasets import get_datasets
from data.import_data import load_iconic_images, load_natural_images
from data.data_processing import onehot_encode
from utils.metrics import *
from classification.evaluate_classifier import evaluate_class_label_decoder, evaluate_softmax_classifier
from visualization.plot_images import save_images_with_metrics, save_decoded_images
from utils.get_latents import add_latents_to_dataset_using_tensors
from visualization.plot_latents import plot_latent_representation

parser = argparse.ArgumentParser()
# Directory arguments
parser.add_argument('--data_path', type=str, default='./data/processed', help='Data directory')
parser.add_argument('--model_dir', type=str, default='./saved_model', help='Saved model directory')
parser.add_argument('--clf_dir', type=str, default='./saved_model/saved_classifier', help='Saved classifier directory')
parser.add_argument('--save_dir', type=str, default='./saved_images_and_metrics', help='For saving images.')
# Model arguments
parser.add_argument('--model_name', type=str, default='vcca_xi', help='Model name',
    choices=[ 'vae_x', 'vcca_xy', 'vcca_xw', 'vcca_xwy', 'vcca_xi', 'vcca_xiy', 'vcca_xiw', 'vcca_xiwy',
              'ae_x', 'splitae_xy', 'splitae_xw', 'splitae_xwy', 'splitae_xi', 'splitae_xiy', 'splitae_xiw', 'splitae_xiwy', ])
parser.add_argument('--K', type=int, default=5, help='Posterior samples when evaluating class label decoder.')
# GMM arguments
parser.add_argument('--n_components', type=int, default=2, help='Number of GMM components.')
parser.add_argument('--mc_samples', type=int, default=100, help='Number of monte carlo samples for approximating kl divergence.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
# Other arguments
parser.add_argument('--feature_extractor_name', type=str, default='densenet', help='Feature extractor. densenet or resnet')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--save_images', action='store_true', default=False, help='Save image with natural, true and decoded iconic images.')
parser.add_argument('--accuracy_file', type=str, default='/home/marcus/iconic_image_metrics.txt', help='File for saving accuracy.')
parser.add_argument('--iconic_image_file', type=str, default='/home/marcus/iconic_image_metrics.txt', 
    help='File for saving metrics for iconic images.')
parser.add_argument('--save_decoded_images', action='store_true', default=False, help='Save decoded iconic images.')

args = parser.parse_args() 

def restore_tf_graph(sess, model_dir):
    """ Restore Tensorflow model and computational graph from meta file.
    """
    graph = tf.get_default_graph()
    meta_file = [f for f in os.listdir(model_dir) if f.endswith('.meta')][0]
    ckpt_file = meta_file.replace('.meta','')

    saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
    saver.restore(sess, os.path.join(model_dir, ckpt_file))
    return graph

def get_decoded_iconic_images(data, sess, input_tensor, target_tensor, batch_size=128):
    """ Get the decoded iconic images corresponding to all features in a dataset.
    """
    print('Get decoded iconic images...')
    features = data['features']
    n_examples = len(features)
    n_batches = int(np.ceil(n_examples/batch_size))

    targets = np.zeros([n_examples, 64, 64 ,3])
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples
        targets[start:end] = sess.run(target_tensor, feed_dict={input_tensor: features[start:end]})
    return targets

def fit_gaussian_mixtures(images, random_seed=0):
    """ Fit Gaussian mixture models for images. 
    """
    gmms = []
    for img in images:
        data = img.reshape(-1, img.shape[-1])
        gmm = GaussianMixture(n_components=args.n_components, random_state=random_seed)
        gmm.fit(data)
        gmms.append(gmm)
    return np.array(gmms)

def compute_kl_matching(gmm_p, gmm_q, true_labels, n_samples=1000):
    """ Compute KL divergence between all GMMs for decoded and 
        the corresponding true iconic images.
    """
    kl = np.zeros(len(gmm_p))
    for i, (gmm, true_label) in enumerate(zip(gmm_p, true_labels)):
        gmm_true = gmm_q[true_label]
        kl[i] = gmm_kl(gmm, gmm_true, n_samples=n_samples)
    return np.mean(kl), kl

def compute_kl_for_all_images(gmm_decoded, gmm_true, n_samples=1000):
    """ Compute KL divergence between all GMMs for decoded and all iconic images.
    """
    kl = np.zeros([len(gmm_decoded), len(gmm_true)]) # [n_examples x n_classes]
    for i, gmm_p in enumerate(gmm_decoded):
        for j, gmm_q in enumerate(gmm_true):
            kl[i, j] = gmm_kl(gmm_p, gmm_q, n_samples=n_samples)
    return kl


# Add argument for use of classifier decoder, iconic image decoder
views = args.model_name.split('_')[-1]
args.use_labels = True if 'y' in views else False
args.use_text = True if 'w' in views else False
args.use_iconic = True if 'i' in views else False
args.use_private = True if 'private' in args.model_name else False

# Create directories and files
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.accuracy_file):
    os.mknod(args.accuracy_file)

config = tf.ConfigProto(device_count = {'GPU': 0}) # only use cpu
sess = tf.Session(config=config)
graph = restore_tf_graph(sess, args.model_dir)

# Get tensors
x = graph.get_tensor_by_name("x:0")
x_recon = graph.get_tensor_by_name("image_feature_recon:0")

# Load datasets
train_data, val_data, test_data = get_datasets(args)
eval_data = test_data

# Classification
if args.use_labels:
    labels = graph.get_tensor_by_name("labels:0")
    scores = graph.get_tensor_by_name("classifier_val_softmax_score:0")
    accuracy = graph.get_tensor_by_name("classifier_val_accuracy:0")
    posterior_samples = graph.get_tensor_by_name("posterior_samples:0")

    tensors = {'x': x, 'labels': labels, 'scores': scores, 'accuracy': accuracy, 
                'posterior_samples': posterior_samples}
    accuracy, accuracy_coarse, predicted_labels = evaluate_class_label_decoder(args, sess, tensors, eval_data)

else:
    print('softmax classifier')
    latent_rep = graph.get_tensor_by_name("latent_rep:0")
    latent_rep_train = sess.run(latent_rep, feed_dict={x: train_data['features']} )
    latent_rep_eval = sess.run(latent_rep, feed_dict={x: eval_data['features']} )
    train_data['latent_features'] = latent_rep_train
    eval_data['latent_features'] = latent_rep_eval

    accuracy, accuracy_coarse, predicted_labels = evaluate_softmax_classifier(args, eval_data, train_data, train=False)

# Write accuracies to file
with open(args.accuracy_file, 'a') as file:
    file.write("Model     Seed    Accuracy   Coarse Accuracy \n {:s}    {:d} {:.3f}    {:.3f}      \n".format(
        args.model_name, args.seed, accuracy, accuracy_coarse))

# Compute iconic image metrics
if args.use_iconic:
    # Create file for saving metrics
    if not os.path.exists(args.iconic_image_file):
        os.mknod(args.iconic_image_file)
    # Get image paths
    iconic_image_paths = np.unique(np.array(train_data['iconic_image_paths']))
    labels_eval = eval_data['labels']
    n_classes = eval_data['n_classes']
    correct_idx = (predicted_labels == labels_eval)
    
    # Get tensors
    iconic_images = graph.get_tensor_by_name("iconic_images:0")
    iconic_image_recon = graph.get_tensor_by_name("iconic_image_recon:0")

    # Get decoded iconic images
    true_iconic = load_iconic_images(eval_data['iconic_image_paths'])
    decoded_iconic = get_decoded_iconic_images(eval_data, sess, x, iconic_image_recon, args.batch_size)

    if args.save_decoded_images:
        new_dir = os.path.join(args.save_dir, 'only_decoded')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        save_decoded_images(decoded_iconic, save_dir=new_dir)
    
    mse = compute_mse(true_iconic, decoded_iconic)
    psnr = compute_psnr(true_iconic, decoded_iconic)
    ssim = compute_ssim(true_iconic, decoded_iconic)
    print('All images - MSE: {:.5f}, PSNR: {:.5f}, SSIM: {:.5f}'.format(mse, psnr, ssim))

    mse_correct = compute_mse(true_iconic[correct_idx], decoded_iconic[correct_idx])
    psnr_correct = compute_psnr(true_iconic[correct_idx], decoded_iconic[correct_idx])
    ssim_correct = compute_ssim(true_iconic[correct_idx], decoded_iconic[correct_idx])
    print('Correctly classified images - MSE: {:.5f}, PSNR: {:.5f}, SSIM: {:.5f}'.format(mse_correct, psnr_correct, ssim_correct))

    mse_incorrect = compute_mse(true_iconic[~correct_idx], decoded_iconic[~correct_idx])
    psnr_incorrect = compute_psnr(true_iconic[~correct_idx], decoded_iconic[~correct_idx])
    ssim_incorrect = compute_ssim(true_iconic[~correct_idx], decoded_iconic[~correct_idx])
    print('Misclassified images - MSE: {:.5f}, PSNR: {:.5f}, SSIM: {:.5f}'.format(mse_incorrect, psnr_incorrect, ssim_incorrect))
    
    # Fit GMMs for true iconic images
    true_iconic = load_iconic_images(iconic_image_paths)
    np.random.seed(args.seed)

    t0 = time.time()
    print('Fit GMMs for computing KL divergences...')
    gmm_true_iconic = fit_gaussian_mixtures(true_iconic, random_seed=args.seed)
    gmm_decoded_iconic = fit_gaussian_mixtures(decoded_iconic, random_seed=args.seed)
    print('Time elapsed: {:.2f} seconds'.format(time.time() - t0))
    
    kl_mean, kl = compute_kl_matching(gmm_decoded_iconic, gmm_true_iconic,
                                     labels_eval, n_samples=args.mc_samples)
    # Save plots of images with corresponding metrics for each image
    if args.save_images:
        print('Saving images...')
        natural_images = load_natural_images(eval_data['natural_image_paths'], [64, 64, 3])
        true_iconic = load_iconic_images(eval_data['iconic_image_paths'])
        correct_idx = correct_idx.astype(int)
        image_path = os.path.join(args.save_dir, 'decoded_iconic_images')
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        save_images_with_metrics(true_iconic, decoded_iconic, natural_images,
                                labels_eval, predicted_labels, kl, image_path)
    # Write all metrics to txt file
    with open(args.iconic_image_file, 'w') as file:
        file.write('MSE {:.4f} \nMSE_correct {:.4f} \nMSE_incorrect {:.4f} \n'
            'PSNR {:.4f} \nPSNR_correct {:.4f} \n PSNR_incorrect {:.4f} \n'
            'SSIM {:.4f} \nSSIM_correct {:.4f} \nSSIM_incorrect {:.4f} \n'
            'KL {:.4f}\naccuracy {:.4f} \ncoarse_acc {:.4f} \n'.format(
            mse, mse_correct, mse_incorrect, psnr, psnr_correct, psnr_incorrect,
            ssim, ssim_correct, ssim_incorrect, kl_mean, accuracy, accuracy_coarse))  

# Plot latent representations using tensors from restored model
tensors = {}
tensors['x'] = graph.get_tensor_by_name("x:0")
tensors['latents'] = graph.get_tensor_by_name("latent_rep:0")
if args.use_private:
    tensors['latents_ux'] = graph.get_tensor_by_name("latent_rep_ux:0")
    if args.use_text:
        tensors['captions'] = graph.get_tensor_by_name("captions:0")
        tensors['latents_uw'] = graph.get_tensor_by_name("latent_rep_uw:0")
    if args.use_iconic:
        tensors['iconic_images'] = graph.get_tensor_by_name("iconic_images:0")
        tensors['latents_ui'] = graph.get_tensor_by_name("latent_rep_ui:0")

# Add latent representations to dataset dictionaries
train_data = add_latents_to_dataset_using_tensors(args, sess, tensors, train_data)
eval_data = add_latents_to_dataset_using_tensors(args, sess, tensors, eval_data)

# Plot latents in 2D using PCA
plot_latent_representation(args, train_data, eval_data, method='pca')