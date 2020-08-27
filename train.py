
import os
import time
import argparse

import numpy as np
import tensorflow as tf

from data.get_datasets import get_datasets
from data.import_data import load_captions, load_iconic_images
from classification.evaluate_classifier import *
from utils.get_model import get_model
from utils.get_training_function import get_training_function
from utils.print_results import print_results, write_accuracy_to_file
from utils.get_latents import add_latents_to_dataset
from visualization.plot_latents import plot_latent_representation

parser = argparse.ArgumentParser()
# Directory arguments
parser.add_argument('--data_path', type=str, default='./data/processed', help='Data directory')
parser.add_argument('--model_dir', type=str, default='./saved_model', help='Saved model directory')
parser.add_argument('--save_dir', type=str, default='./saved_images_and_metrics',
 help='Directory for saving figures and printing to files')
# Training arguments
parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Training optimizer')
# Data arguments
parser.add_argument('--iconic_img_size', type=int, default=64, help='Iconic image size')
# Model arguments
parser.add_argument('--model_name', type=str, default='ae', help='Model name',
    choices=[ 'vae', 'vcca_xy', 'vcca_xw', 'vcca_xwy', 'vcca_xi', 'vcca_xiy', 'vcca_xiw', 'vcca_xiwy', 
              'vcca_private_xw', 'vcca_private_xwy', 'vcca_private_xi', 'vcca_private_xiy',
              'ae', 'splitae_xy', 'splitae_xw', 'splitae_xwy', 'splitae_xi', 'splitae_xiy', 'splitae_xiw', 'splitae_xiwy', ])
parser.add_argument('--z_dim', type=int, default=200, help='Dimension of latent variable')
parser.add_argument('--lambda_x', type=float, default=1.0, help='Likelihood weight for features')
parser.add_argument('--lambda_i', type=float, default=1000.0, help='Likelihood weight for iconic images')
parser.add_argument('--lambda_w', type=float, default=1000.0, help='Likelihood weight for text')
parser.add_argument('--lambda_y', type=float, default=1000.0, help='Likelihood weight for labels')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--n_layers_encoder', type=int, default=1, help='Number of layers in encoder')
parser.add_argument('--n_layers_decoder', type=int, default=1, help='Number of layers in decoder')
parser.add_argument('--n_layers_classifier', type=int, default=1, help='Number of layers in classifier')
parser.add_argument('--use_batchnorm', type=int, default=0, help='Use BatchNorm in encoder and decoder')
parser.add_argument('--K', type=int, default=5, help='Posterior samples when evaluating class label decoder.')
# Visualization arguments 
parser.add_argument('--visualize_latents', action='store_true', default=False, help='PCA plot of latents in 2D.')
parser.add_argument('--visualization_method', type=str, default='pca', help='Method for visualizing latents.', choices=[ 'tsne', 'pca',])
# Other arguments
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--print_every', type=int, default=1, help='Print every i\'th epoch')
parser.add_argument('--which_gpu', type=int, default=0, help='Which GPU to use? 0 or 1')
parser.add_argument('--save_model', type=int, default=0, help='Option for saving model in directory')
parser.add_argument('--save_file', type=str, default='./clf_metrics.txt', help='File for saving results')
parser.add_argument('--eval_mode', type=str, default='test', help='Evaluation mode, val or test')
parser.add_argument('--feature_extractor_name', type=str, default='densenet', help='Feature extractor. densenet or resnet')

args = parser.parse_args() 

# Add argument for use of classifier decoder
views = args.model_name.split('_')[-1]
args.use_labels = True if 'y' in views else False
args.use_text = True if 'w' in views else False
args.use_iconic = True if 'i' in views else False
args.use_private = True if 'private' in args.model_name else False
args.use_batchnorm = True if args.use_batchnorm == 1 else False
print(args)

# Create directories and saving file
if not os.path.exists(args.save_file):
    os.mknod(args.save_file)
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

# Add directory for saving softmax classifier
if not args.use_labels:
    args.clf_dir = os.path.join(args.model_dir, 'saved_classifier')
    if not os.path.exists(args.clf_dir):
        os.mkdir(args.clf_dir)

# Reset graph and set seed
tf.reset_default_graph()
print("Random seed: ", args.seed)
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

### Load datasets
train_data, val_data, test_data = get_datasets(args)

### Get model
model = get_model(args, train_data)
elbo = model.loss

t_vars = tf.trainable_variables()
with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    # Minimize elbo
    grads_and_vars = optimizer.compute_gradients(-elbo, var_list=t_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): 
        train_op = optimizer.apply_gradients(grads_and_vars)

run_single_epoch = get_training_function(args.model_name)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)

config = tf.ConfigProto() 
config.gpu_options.visible_device_list = str(args.which_gpu) #only see the gpu 1
sess = tf.Session(config=config) 
sess.run(init)

elbos_train = []
elbos_val = []
accuracy_train = []
accuracy_val = []

# Store hyperparameters
train_hyperparams = {'batch_size': args.batch_size, 'is_training': True, 'dropout_rate': args.dropout_rate, 'kl_weight': 1.0}
val_hyperparams = {'batch_size': args.batch_size, 'is_training': False, 'dropout_rate': 1.0, 'kl_weight': 1.0}

print('Start training...')
for epoch in range(args.n_epochs):

    # Train step
    t0 = time.time()
    results = run_single_epoch(args, train_data, model, train_hyperparams, session=sess, 
                                train_op=train_op, shuffle=True, mode='train')

    if ((epoch+1) % args.print_every) == 0 or epoch == 0:
        print_results(args, results, mode='train', epoch=epoch+1, t_start=t0)

    elbos_train.append(results['total_loss']) # Store elbo loss
    if args.use_labels:
        accuracy_train.append(results['accuracy'])
    
    # Validation step
    results = run_single_epoch(args, val_data, model, val_hyperparams, session=sess, 
                               train_op=None, shuffle=False, mode='val')
    if ((epoch+1) % args.print_every) == 0 or epoch == 0:
        print_results(args, results, mode='val')

    elbos_val.append(results['total_loss']) # Store elbo loss

    if args.use_labels:
        accuracy_val.append(results['accuracy'])
    
    # Save model and elbos
    if args.save_model:
        saver.save(sess, os.path.join(args.model_dir, 'models_'+str(epoch)+'.ckpt'))

    np.save(args.model_dir + '/elbo_train.npy', elbos_train)
    np.save(args.model_dir + '/elbo_val.npy', elbos_val)
    if args.use_labels:
        np.save(args.model_dir + '/accuracy_train.npy', accuracy_train)
        np.save(args.model_dir + '/accuracy_val.npy', accuracy_val)        


# Evaluation
if args.eval_mode == 'val':
    eval_data = val_data
elif args.eval_mode == 'test':
    eval_data = test_data
else:
    ValueError('Unknown eval_mode: %s'.format(args.eval_mode))

if args.use_labels:
    model_type = args.model_name.split('_')[0]
    if model_type == 'vcca':
        tensors = {'x': model.x, 'labels': model.labels, 'scores': model.val_score,
                    'accuracy': model.val_accuracy, 'posterior_samples': model.K}
    elif model_type == 'splitae':
        tensors = {'x': model.x, 'labels': model.labels, 'scores': model.logits, 'accuracy': model.accuracy}

    accuracy, accuracy_coarse, predicted_labels = evaluate_class_label_decoder(args, sess, tensors, eval_data)
else:
    # Softmax classifier
    latent_rep_train = sess.run(model.latent_rep, feed_dict={model.x: train_data['features']} )
    latent_rep_eval = sess.run(model.latent_rep, feed_dict={model.x: eval_data['features']} )
    train_data['latent_features'] = latent_rep_train
    eval_data['latent_features'] = latent_rep_eval
    accuracy, accuracy_coarse, predicted_labels = evaluate_softmax_classifier(args, eval_data,
                                                                            train_data, train=True)
# Write accuracies to file
write_accuracy_to_file(args, accuracy, accuracy_coarse)

if args.visualize_latents:
    # Add latent representations to dataset dictionaries
    train_data = add_latents_to_dataset(args, sess, model, train_data)
    eval_data = add_latents_to_dataset(args, sess, model, eval_data)

    # Plot latents in 2D using PCA
    plot_latent_representation(args, train_data, eval_data, method='pca')